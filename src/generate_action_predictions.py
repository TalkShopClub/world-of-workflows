#!/usr/bin/env python3
"""
Generate action predictions for all trajectories using specified LLM model.
Similar to test 2 in test.py but processes all trajectory files.
"""

import asyncio
import json
import os
import argparse
from pathlib import Path
try:
    from rest_apis.world_model_scripts.world_model_agent import WorldModelAgent
except ImportError:
    # Fallback for when running script directly
    from world_model_agent import WorldModelAgent
import time
from datetime import datetime

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    pass

def should_process_trajectory(trajectory_file, output_dir):
    """Check if trajectory file should be processed based on modification time"""
    output_file = output_dir / f"{trajectory_file.stem}_action_predictions.json"
    
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


def convert_audits_to_state_diff(audits):
    """Convert audit records to state_diff format"""
    try:
        from rest_apis.world_model_scripts.world_model_agent import StateDiff, SysAuditRecord, AdditionalInformation, operation
    except ImportError:
        # Fallback for when running script directly
        from world_model_agent import StateDiff, SysAuditRecord, AdditionalInformation, operation
    
    # Convert audits to SysAuditRecord format
    sysaudit_records = []
    for audit in audits:
        sysaudit_records.append(SysAuditRecord(
            fieldname=audit.get('fieldname', ''),
            newvalue=audit.get('newvalue', ''),
            tablename=audit.get('tablename', ''),
            oldvalue=audit.get('oldvalue', '')
        ))
    
    # Determine operation types
    operation_types = []
    for audit in audits:
        old_value = audit.get('oldvalue', '')
        new_value = audit.get('newvalue', '')
        
        if old_value == '' and new_value != '':
            operation_types.append(operation.post)
        elif old_value != '' and new_value != '' and new_value != 'DELETED':
            operation_types.append(operation.put)
        elif new_value == 'DELETED' and old_value != '':
            operation_types.append(operation.delete)
        else:
            operation_types.append(operation.get)
    
    # Count different types of changes
    num_created = len([a for a in audits if a.get('oldvalue', '') == '' and a.get('newvalue', '') != ''])
    num_modified = len([a for a in audits if a.get('oldvalue', '') != '' and a.get('newvalue', '') != '' and a.get('newvalue', '') != 'DELETED'])
    num_deleted = len([a for a in audits if a.get('newvalue', '') == 'DELETED' and a.get('oldvalue', '') != ''])
    
    # Get unique tables modified
    tables_modified = list(set([audit.get('tablename', '') for audit in audits if audit.get('tablename')]))
    
    additional_info = AdditionalInformation(
        num_audits=len(audits),
        num_modified_entries=num_modified,
        num_deleted_entries=num_deleted,
        num_created_entries=num_created,
        operation_type=operation_types,
        tables_modified=tables_modified
    )
    
    return StateDiff(
        sysauditrecord=sysaudit_records,
        additional_information=additional_info
    ).model_dump(mode='json')


async def process_trajectory(agent, trajectory_file, output_dir, custom_schema_path=None, verbose=False, max_state_diffs=20):
    """Process a single trajectory file and generate action predictions"""
    print(f"\n{'='*60}")
    print(f"Processing: {trajectory_file.name}")
    print(f"{'='*60}")
    
    try:
        # Load trajectory data
        with open(trajectory_file, "r") as f:
            trajectory = json.load(f)
        
        print(f"Loaded {len(trajectory)} steps from trajectory")
        
        # Extract task name from trajectory file (without extension)
        task_name = trajectory_file.stem
        if verbose:
            print(f"📋 Task name: {task_name}")
        
        # Extract state diffs for action prediction
        state_diffs = []
        for i, step in enumerate(trajectory):
            state_diff = None
            
            # Try different formats for state information
            if "state_diff" in step:
                # Format 1: Direct state_diff field
                try:
                    if isinstance(step["state_diff"], str):
                        state_diff = json.loads(step["state_diff"])
                    else:
                        state_diff = step["state_diff"]
                except (json.JSONDecodeError, TypeError) as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not parse state_diff in step {i+1}: {e}")
                    continue
                    
            elif "ground_truth_state" in step:
                # Format 2: ground_truth_state field
                try:
                    if isinstance(step["ground_truth_state"], str):
                        state_diff = json.loads(step["ground_truth_state"])
                    else:
                        state_diff = step["ground_truth_state"]
                except (json.JSONDecodeError, TypeError) as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not parse ground_truth_state in step {i+1}: {e}")
                    continue
                    
            elif "audits" in step and step["audits"]:
                # Format 3: Convert audits to state_diff format
                try:
                    state_diff = convert_audits_to_state_diff(step["audits"])
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not convert audits to state_diff in step {i+1}: {e}")
                    continue
            
            if state_diff:
                state_diffs.append(state_diff)
            elif verbose:
                print(f"⚠️  Warning: No state information found in step {i+1}")
        
        if not state_diffs:
            print(f"❌ No valid state diffs found in {trajectory_file.name}")
            return None
        
        print(f"Extracted {len(state_diffs)} valid state diffs")
        
        # Limit the number of state diffs to prevent context length issues
        if len(state_diffs) > max_state_diffs:
            print(f"⚠️  Warning: Limiting to first {max_state_diffs} state diffs to prevent context length issues")
            state_diffs = state_diffs[:max_state_diffs]
        
        # Generate action predictions
        print(f"🤖 Generating action predictions...")
        start_time = time.time()
        
        # Use custom schema prediction if available, otherwise use default
        if custom_schema_path and task_name:
            print(f"📁 Using custom schema for task: {task_name}")
            predicted_actions = await agent.predict_actions_custom(state_diffs, task_name, custom_schema_path)
        else:
            predicted_actions = await agent.predict_actions(state_diffs, task_name)
        
        end_time = time.time()
        print(f"✅ Generated {len(predicted_actions)} action predictions in {end_time - start_time:.2f} seconds")
        
        # Prepare results
        results = {
            "trajectory_file": trajectory_file.name,
            "task_name": task_name,
            "total_steps": len(trajectory),
            "valid_state_diffs": len(state_diffs),
            "predicted_actions": len(predicted_actions),
            "processing_time_seconds": end_time - start_time,
            "custom_schema_used": custom_schema_path is not None,
            "timestamp": datetime.now().isoformat(),
            "predictions": [action.model_dump() for action in predicted_actions]
        }
        
        # Save results
        output_file = output_dir / f"{trajectory_file.stem}_action_predictions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Saved results to: {output_file}")
        
        return results
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing {trajectory_file.name}: {e}")
        traceback.print_exc()
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate action predictions for trajectories using specified LLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model with custom schema
  python generate_action_predictions.py --model anthropic/claude-sonnet-4.5 --custom-schema-path /path/to/schemas

  # Use specific model and force reprocessing
  python generate_action_predictions.py --model anthropic/claude-sonnet-4.5 --force

  # Use without custom schema (default behavior)
  python generate_action_predictions.py --model anthropic/claude-sonnet-4.5
        """
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/gpt-4o-mini",
        help="LLM model to use for predictions (default: openai/gpt-4o-mini)"
    )
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
    parser.add_argument(
        "--max-state-diffs",
        type=int,
        default=100,
        help="Maximum number of state diffs to process per trajectory (default: 100)"
    )
    return parser.parse_args()

async def main():
    """Main function to process all trajectories"""
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model
    force_reprocess = args.force
    custom_schema_path = args.custom_schema_path
    verbose = args.verbose
    max_state_diffs = args.max_state_diffs
    
    print(f"🤖 Using model: {model_name}")
    
    if custom_schema_path:
        print(f"📁 Using custom schema path: {custom_schema_path}")
        # Validate custom schema path exists
        if not Path(custom_schema_path).exists():
            print(f"❌ Error: Custom schema path does not exist: {custom_schema_path}")
            return
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
    trajectories_dir = script_dir / "trajectories"
    base_output_dir = script_dir / "action_predictions_results"
    base_output_dir.mkdir(exist_ok=True)
    
    output_dir = base_output_dir / model_name.split("/")[-1].replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Trajectories directory: {trajectories_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Get all trajectory files
    trajectory_files = list(trajectories_dir.glob("*.json"))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    if not trajectory_files:
        print("❌ No trajectory files found!")
        return
    
    # Initialize agent
    print(f"\n🤖 Initializing WorldModelAgent with {model_name}...")
    agent = WorldModelAgent(model=model_name)
    
    # Process each trajectory
    results_summary = []
    successful = 0
    failed = 0
    skipped = 0
    
    for i, trajectory_file in enumerate(trajectory_files, 1):
        print(f"\n[{i}/{len(trajectory_files)}] Checking {trajectory_file.name}")
        
        # Check if trajectory needs processing (unless force reprocess is enabled)
        if not force_reprocess:
            should_process, reason = should_process_trajectory(trajectory_file, output_dir)
            
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
        result = await process_trajectory(agent, trajectory_file, output_dir, custom_schema_path, verbose, max_state_diffs)
        
        if result:
            results_summary.append({
                "file": trajectory_file.name,
                "status": "success",
                "total_steps": result["total_steps"],
                "valid_state_diffs": result["valid_state_diffs"],
                "predicted_actions": result["predicted_actions"],
                "processing_time": result["processing_time_seconds"]
            })
            successful += 1
        else:
            results_summary.append({
                "file": trajectory_file.name,
                "status": "failed"
            })
            failed += 1
    
    # Save summary
    summary = {
        "model_used": model_name,
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
    
    summary_file = output_dir / "action_predictions_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model used: {model_name}")
    print(f"Total files checked: {len(trajectory_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"📁 Summary saved to: {summary_file}")
    print(f"📁 Individual results saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
