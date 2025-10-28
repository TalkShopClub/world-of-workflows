#!/usr/bin/env python3
"""
Predict states for trajectory files using a specified model.
Takes a folder of trajectory files and predicts states for the first k actions of each file.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import glob
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from rest_apis.world_model_scripts.world_model_agent import WorldModelAgent
from src.browsergym.workarena.instance import SNowInstance

def should_process_trajectory(trajectory_file, output_dir, k):
    """Check if trajectory file should be processed based on modification time"""
    output_file = output_dir / f"{trajectory_file.stem}_k{k}_state_predictions.json"
    
    # If prediction file doesn't exist, process the trajectory
    if not output_file.exists():
        return True, "No existing prediction file found"
    
    # Get modification times
    trajectory_mtime = trajectory_file.stat().st_mtime
    prediction_mtime = output_file.stat().st_mtime
    
    # If trajectory is newer than prediction, process it
    if trajectory_mtime > prediction_mtime:
        return True, f"Trajectory modified after last prediction (trajectory: {datetime.fromtimestamp(trajectory_mtime)}, prediction: {datetime.fromtimestamp(prediction_mtime)})"
    
    return False, f"Trajectory unchanged since last prediction (trajectory: {datetime.fromtimestamp(trajectory_mtime)}, prediction: {datetime.fromtimestamp(prediction_mtime)})"

class TrajectoryStatePredictor:
    """Predict states for a trajectory file using a world model agent."""
    
    def __init__(self, model: str, custom_schema_path: str = None):
        self.model = model
        self.custom_schema_path = custom_schema_path
        self.agent = None
        self.instance = None
    
    async def initialize(self):
        """Initialize the world model agent and ServiceNow instance."""
        print(f"🔧 Initializing TrajectoryStatePredictor with model: {self.model}")
        
        if self.custom_schema_path:
            print(f"📁 Using custom schema path: {self.custom_schema_path}")
            # Validate custom schema path exists
            if not Path(self.custom_schema_path).exists():
                raise FileNotFoundError(f"Custom schema path does not exist: {self.custom_schema_path}")
        else:
            print("📁 Using default schema (all_table_schemas.json)")
        
        # Initialize ServiceNow instance
        self.instance = SNowInstance(
            snow_url=os.getenv("SNOW_INSTANCE_URL"),
            snow_credentials=(os.getenv("SNOW_INSTANCE_UNAME"), os.getenv("SNOW_INSTANCE_PWD"))
        )
        
        # Initialize world model agent
        self.agent = WorldModelAgent(model=self.model)
        await self.agent.initialize_mcp_server("full")
        
        print("✅ Initialization complete")
    
    def load_trajectory(self, trajectory_file: str) -> List[Dict]:
        """Load trajectory file and extract actions."""
        print(f"📂 Loading trajectory from: {trajectory_file}")
        
        try:
            with open(trajectory_file, 'r') as f:
                trajectory = json.load(f)
            
            if not isinstance(trajectory, list):
                raise ValueError("Trajectory file must contain a list of actions")
            
            print(f"✅ Loaded {len(trajectory)} actions from trajectory")
            return trajectory
            
        except Exception as e:
            print(f"❌ Error loading trajectory: {e}")
            return []
    
    
    def extract_action_calls(self, trajectory: List[Dict], k: int) -> List[Dict]:
        """Extract action calls from trajectory for first k actions."""
        print(f"🔍 Extracting action calls for first {k} actions...")
        
        action_calls = []
        for i, action in enumerate(trajectory[:k]):
            if 'action' in action and 'tool_name' in action['action']:
                action_call = {
                    'tool_name': action['action']['tool_name'],
                    'parameters': action['action'].get('parameters', {})
                }
                action_calls.append(action_call)
                print(f"   Action {i+1}: {action_call['tool_name']}")
            else:
                print(f"   ⚠️ Skipping action {i+1}: Missing tool_name")
        
        print(f"✅ Extracted {len(action_calls)} action calls")
        return action_calls
    
    async def predict_states(self, action_calls: List[Dict], task_name: str = None) -> List[Dict]:
        """Predict states for the given action calls."""
        print(f"🔮 Predicting states for {len(action_calls)} actions...")
        
        predicted_states = []
        
        for i, action_call in enumerate(action_calls):
            print(f"   Predicting state for action {i+1}/{len(action_calls)}: {action_call['tool_name']}")
            
            try:
                # Use custom schema prediction if available, otherwise use default
                if self.custom_schema_path and task_name:
                    print(f"   📁 Using custom schema for task: {task_name}")
                    state_prediction = await self.agent.predict_states_custom([action_call], task_name, self.custom_schema_path)
                else:
                    state_prediction = await self.agent.predict_states([action_call])
                
                if state_prediction and len(state_prediction) > 0:
                    # Convert StateDiff object to dictionary
                    state_dict = {
                        'sysauditrecord': [record.dict() for record in state_prediction[0].sysauditrecord],
                        'additional_information': state_prediction[0].additional_information.dict() if hasattr(state_prediction[0].additional_information, 'dict') else state_prediction[0].additional_information
                    }
                    predicted_states.append(state_dict)
                    print(f"   ✅ Generated state prediction with {len(state_dict['sysauditrecord'])} audit records")
                else:
                    print(f"   ⚠️ No state prediction generated")
                    predicted_states.append({
                        'sysauditrecord': [],
                        'additional_information': {}
                    })
                    
            except Exception as e:
                print(f"   ❌ Error predicting state: {e}")
                predicted_states.append({
                    'sysauditrecord': [],
                    'additional_information': {'error': str(e)}
                })
        
        print(f"✅ Generated {len(predicted_states)} state predictions")
        return predicted_states
    
    def save_results(self, trajectory_file: str, action_calls: List[Dict], 
                    predicted_states: List[Dict], k: int, output_dir: Path) -> str:
        """Save prediction results to file organized by model name and k value."""

        # Extract task name from trajectory file (without extension)
        trajectory_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        output_file = output_dir / f"{trajectory_name}_k{k}.json"
        
        results = {
            'prediction_info': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'trajectory_file': trajectory_file,
                'k': k,
                'total_actions_predicted': len(action_calls),
                'total_states_generated': len(predicted_states)
            },
            'action_calls': action_calls,
            'predicted_states': predicted_states
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return ""
    
    async def process_single_trajectory(self, trajectory_file: str, k: int, output_dir: Path) -> Optional[Dict]:
        """Process a single trajectory file and return results."""
        print(f"\n🎯 Processing: {os.path.basename(trajectory_file)}")
        print("-" * 50)
        
        # Load trajectory
        trajectory = self.load_trajectory(trajectory_file)
        if not trajectory:
            print(f"❌ Failed to load trajectory: {trajectory_file}")
            return None
        
        # Extract task name from trajectory file (without extension)
        task_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        print(f"📋 Task name: {task_name}")
        
        # Adjust k if trajectory is shorter
        actual_k = min(k, len(trajectory))
        if actual_k < k:
            print(f"⚠️ Warning: Trajectory has only {len(trajectory)} actions, but k={k}")
            print(f"Will predict for all {len(trajectory)} available actions")
        
        # Extract action calls
        action_calls = self.extract_action_calls(trajectory, actual_k)
        if not action_calls:
            print(f"❌ No valid action calls found in: {trajectory_file}")
            return None
        
        # Predict states (pass task_name for custom schema support)
        predicted_states = await self.predict_states(action_calls, task_name)
        
        # Print summary for this file
        self.print_summary(action_calls, predicted_states)
        
        # Save results for this file
        output_file = self.save_results(trajectory_file, action_calls, predicted_states, actual_k, output_dir)
        
        return {
            'trajectory_file': trajectory_file,
            'action_calls': action_calls,
            'predicted_states': predicted_states,
            'output_file': output_file,
            'k': actual_k,
            'task_name': task_name
        }
    
    def print_summary(self, action_calls: List[Dict], predicted_states: List[Dict]):
        """Print a summary of the predictions."""
        print("\n📊 Prediction Summary:")
        print("=" * 50)
        
        total_audit_records = 0
        total_additional_info = 0
        
        for i, (action, state) in enumerate(zip(action_calls, predicted_states)):
            audit_count = len(state.get('sysauditrecord', []))
            additional_info = state.get('additional_information', {})
            additional_count = len(additional_info) if isinstance(additional_info, dict) else 0
            
            total_audit_records += audit_count
            total_additional_info += additional_count
            
            print(f"Action {i+1}: {action['tool_name']}")
            print(f"  - Audit records: {audit_count}")
            print(f"  - Additional info: {additional_count}")
        
        print(f"\nTotal audit records predicted: {total_audit_records}")
        print(f"Total additional information: {total_additional_info}")
        print(f"Average audit records per action: {total_audit_records / len(action_calls) if action_calls else 0:.1f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict states for trajectory files using specified LLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model with custom schema
  python predict_trajectory_states.py --model anthropic/claude-sonnet-4.5 --k 5 --custom-schema-path /path/to/schemas

  # Use specific model and force reprocessing
  python predict_trajectory_states.py --model anthropic/claude-sonnet-4.5 --k 3 --force

  # Use without custom schema (default behavior)
  python predict_trajectory_states.py --model anthropic/claude-sonnet-4.5 --k 5
        """
    )
    parser.add_argument("--model", help="Model name (e.g., 'anthropic/claude-sonnet-4', 'openai/gpt-4o')")
    parser.add_argument("--k", type=int, help="Number of actions to predict (1 to k)")
    parser.add_argument("--trajectory_folder", nargs='?', default="trajectories", 
                       help="Path to folder containing trajectory JSON files (default: trajectories)")
    parser.add_argument(
        "--custom-schema-path",
        type=str,
        default=None,
        help="Path to directory containing custom schema JSON files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all trajectories, ignoring timestamps"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

async def main():
    """Main function to process all trajectories"""
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model
    k = args.k
    force_reprocess = args.force
    custom_schema_path = args.custom_schema_path
    verbose = args.verbose
    
    print(f"🤖 Using model: {model_name}")
    print(f"🎯 Predicting states for first {k} actions")
    
    if custom_schema_path:
        print(f"📁 Using custom schema path: {custom_schema_path}")
    else:
        print("📁 Using default schema (all_table_schemas.json)")
    
    if force_reprocess:
        print("🔄 Force reprocessing enabled - will process all trajectories")
    
    if verbose:
        print("🔍 Verbose mode enabled")
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it in a .env file or export it in your shell")
        return

    # Setup paths
    script_dir = Path(__file__).parent
    trajectories_dir = script_dir / args.trajectory_folder
    base_output_dir = script_dir / "state_preds_new"
    base_output_dir.mkdir(exist_ok=True)
    
    # Create model-specific subdirectory
    model_name_clean = model_name.split("/")[-1].replace(":", "_")
    output_dir = base_output_dir / model_name_clean
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Trajectories directory: {trajectories_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Get all trajectory files
    trajectory_files = list(trajectories_dir.glob("*.json"))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    if not trajectory_files:
        print("❌ No trajectory files found!")
        return
    
    # Initialize predictor
    print(f"\n🤖 Initializing TrajectoryStatePredictor with {model_name}...")
    predictor = TrajectoryStatePredictor(model_name, custom_schema_path)
    
    try:
        # Initialize
        await predictor.initialize()
        
        # Process each trajectory
        results_summary = []
        successful = 0
        failed = 0
        skipped = 0
        
        for i, trajectory_file in enumerate(trajectory_files, 1):
            print(f"\n[{i}/{len(trajectory_files)}] Checking {trajectory_file.name}")
            
            # Check if trajectory needs processing (unless force reprocess is enabled)
            if not force_reprocess:
                should_process, reason = should_process_trajectory(trajectory_file, output_dir, k)
                
                if not should_process:
                    print(f"⏭️  Skipping {trajectory_file.name}: {reason}")
                    results_summary.append({
                        "file": trajectory_file.name,
                        "status": "skipped",
                        "reason": reason
                    })
                    skipped += 1
                    continue
            else:
                reason = "Force reprocessing enabled"
            
            print(f"🔄 Processing {trajectory_file.name}: {reason}")
            result = await predictor.process_single_trajectory(str(trajectory_file), k, output_dir)
            
            if result:
                results_summary.append({
                    "file": trajectory_file.name,
                    "status": "success",
                    "total_actions_predicted": len(result["action_calls"]),
                    "total_states_generated": len(result["predicted_states"]),
                    "k": result["k"]
                })
                successful += 1
                print(f"✅ Successfully processed: {trajectory_file.name}")
            else:
                results_summary.append({
                    "file": trajectory_file.name,
                    "status": "failed"
                })
                failed += 1
                print(f"❌ Failed to process: {trajectory_file.name}")
        
        # Save summary
        summary = {
            "model_used": model_name,
            "k": k,
            "custom_schema_path": custom_schema_path,
            "total_files": len(trajectory_files),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "force_reprocess": force_reprocess,
            "verbose": verbose,
            "timestamp": datetime.now().isoformat(),
            "results": results_summary
        }
        
        summary_file = output_dir / f"state_predictions_k{k}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Model used: {model_name}")
        print(f"K value: {k}")
        print(f"Total files checked: {len(trajectory_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"📁 Summary saved to: {summary_file}")
        print(f"📁 Individual results saved to: {output_dir}/k={k}/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
