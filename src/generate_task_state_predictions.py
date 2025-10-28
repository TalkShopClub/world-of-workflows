#!/usr/bin/env python3
"""
Generate state predictions for tasks defined in tasks.py

This script:
1. Runs each task from tasks.py to capture MCP calls
2. Converts MCP calls to ActionCall format
3. Uses WorldModelAgent to predict state changes
4. Saves results in format expected by eval_pipeline.py
"""

import asyncio
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.browsergym.workarena.instance import SNowInstance
from dotenv import load_dotenv

# Import task classes
from custom.tasks import (
    TransferAsset, UserGroupAsset, KnowledgeBaseArticle, Incident, CatalogItem,
    ApproveChangeRequest, RejectChangeRequest, AssignUserRole, ChangeUserInfo,
    CreateHierarchy, RemoveUserFromGroup, PublishKnowledgeBaseArticle,
    MoveCatalogItem, DeactivateUser, PromoteUser, CreateRequest, CreateHardwareAsset
)

# Import world model components
from rest_apis.world_model_scripts.world_model_agent import WorldModelAgent, ActionCall, StateDiff

# Load environment variables
load_dotenv()

class TaskStatePredictor:
    """Generates state predictions for ServiceNow tasks"""
    
    def __init__(self, model: str = "anthropic/claude-sonnet-4.5", custom_schema_path: str = None):
        self.model = model
        self.agent = None
        self.instance = None
        self.results = []
        self.custom_schema_path = custom_schema_path
        
    async def initialize(self):
        """Initialize the world model agent and ServiceNow instance"""
        print("🔧 Initializing TaskStatePredictor...")
        
        # Initialize ServiceNow instance
        self.instance = SNowInstance(
            snow_url=os.getenv("SNOW_INSTANCE_URL"),
            snow_credentials=(os.getenv("SNOW_INSTANCE_UNAME"), os.getenv("SNOW_INSTANCE_PWD"))
        )
        
        # Initialize world model agent
        self.agent = WorldModelAgent(model=self.model)
        await self.agent.initialize_mcp_server("full")
        
        print("✅ Initialization complete")
    
    def extract_action_calls(self, mcp_data: Dict) -> List[ActionCall]:
        """Extract ActionCall objects from task MCP data"""
        action_calls = []
        
        for mcp_call in mcp_data.get("mcp_calls", []):
            action = mcp_call.get("action", {})
            tool_name = action.get("tool_name")
            parameters = action.get("parameters", {})
            
            if tool_name:
                action_calls.append(ActionCall(
                    tool_name=tool_name,
                    parameters=parameters
                ))
        
        return action_calls
    
    async def run_task_and_predict_states(self, task_class, task_name: str) -> Dict:
        """Run a task and generate state predictions"""
        print(f"\n🔄 Running task: {task_name}")
        
        try:
            # Create and run task
            task = task_class(self.instance)
            task_result = task.run()
            
            if not task_result or not task_result.get("success"):
                print(f"❌ Task {task_name} failed to run successfully")
                return {
                    "task_name": task_name,
                    "success": False,
                    "error": "Task execution failed",
                    "mcp_data": task_result.get("mcp_data", {}) if task_result else {}
                }
            
            # Extract action calls from MCP data
            mcp_data = task_result.get("mcp_data", {})
            action_calls = self.extract_action_calls(mcp_data)
            
            if not action_calls:
                print(f"⚠️ No action calls found for task {task_name}")
                return {
                    "task_name": task_name,
                    "success": False,
                    "error": "No action calls found",
                    "mcp_data": mcp_data
                }
            
            print(f"📊 Found {len(action_calls)} action calls")
            
            # Convert ActionCall objects to dict format for world model
            actions_dict = []
            for action_call in action_calls:
                actions_dict.append({
                    "tool_name": action_call.tool_name,
                    "parameters": action_call.parameters
                })
            
            # Generate state predictions using world model
            print(f"🤖 Generating state predictions for {task_name}...")
            if self.custom_schema_path:
                print(f"📁 Using custom schema from: {self.custom_schema_path}")
                predicted_states = await self.agent.predict_states_custom(actions_dict, task_name, self.custom_schema_path)
            else:
                predicted_states = await self.agent.predict_states(actions_dict, task_name)
            
            # Convert predicted states to dict format for JSON serialization
            predicted_states_dict = []
            for state in predicted_states:
                predicted_states_dict.append(state.model_dump(mode='json'))
            
            print(f"✅ Generated {len(predicted_states)} state predictions")
            
            return {
                "task_name": task_name,
                "success": True,
                "action_calls": [action.model_dump(mode='json') for action in action_calls],
                "predicted_states": predicted_states_dict,
                "mcp_data": mcp_data,
                "num_actions": len(action_calls),
                "num_predictions": len(predicted_states)
            }
            
        except Exception as e:
            print(f"❌ Error running task {task_name}: {e}")
            traceback.print_exc()
            return {
                "task_name": task_name,
                "success": False,
                "error": str(e),
                "mcp_data": {}
            }
    
    async def generate_all_predictions(self):
        """Generate state predictions for all tasks"""
        print("🚀 Starting state prediction generation for all tasks...")
        
        # Define all task classes and their names
        task_classes = [
            (TransferAsset, "TransferAsset"),
            (UserGroupAsset, "UserGroupAsset"),
            (KnowledgeBaseArticle, "KnowledgeBaseArticle"),
            (Incident, "Incident"),
            (CatalogItem, "CatalogItem"),
            (ApproveChangeRequest, "ApproveChangeRequest"),
            (RejectChangeRequest, "RejectChangeRequest"),
            (AssignUserRole, "AssignUserRole"),
            (ChangeUserInfo, "ChangeUserInfo"),
            (CreateHierarchy, "CreateHierarchy"),
            (RemoveUserFromGroup, "RemoveUserFromGroup"),
            (PublishKnowledgeBaseArticle, "PublishKnowledgeBaseArticle"),
            (MoveCatalogItem, "MoveCatalogItem"),
            (DeactivateUser, "DeactivateUser"),
            (PromoteUser, "PromoteUser"),
            (CreateRequest, "CreateRequest"),
            (CreateHardwareAsset, "CreateHardwareAsset")
        ]
        
        # Process each task
        for task_class, task_name in task_classes:
            result = await self.run_task_and_predict_states(task_class, task_name)
            self.results.append(result)
            
            # Add small delay between tasks to avoid overwhelming the system
            await asyncio.sleep(2)
        
        print(f"\n✅ Completed processing {len(self.results)} tasks")
    
    def save_results(self, output_path: str = None):
        """Save results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"task_state_predictions_{timestamp}.json"
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare summary statistics
        successful_tasks = [r for r in self.results if r.get("success")]
        failed_tasks = [r for r in self.results if not r.get("success")]
        
        summary = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "total_tasks": len(self.results),
                "successful_tasks": len(successful_tasks),
                "failed_tasks": len(failed_tasks)
            },
            "task_results": self.results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total tasks: {len(self.results)}")
        print(f"   Successful: {len(successful_tasks)}")
        print(f"   Failed: {len(failed_tasks)}")
        
        if successful_tasks:
            total_actions = sum(r.get("num_actions", 0) for r in successful_tasks)
            total_predictions = sum(r.get("num_predictions", 0) for r in successful_tasks)
            print(f"   Total actions: {total_actions}")
            print(f"   Total predictions: {total_predictions}")
        
        if failed_tasks:
            print(f"\n❌ Failed tasks:")
            for task in failed_tasks:
                print(f"   - {task['task_name']}: {task.get('error', 'Unknown error')}")
        
        return output_path

def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="Generate state predictions for ServiceNow tasks using custom schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model with custom schema
  python generate_task_state_predictions.py --custom-schema-path /path/to/schemas

  # Use specific model and output file
  python generate_task_state_predictions.py --model anthropic/claude-sonnet-4.5 --output results.json

  # Use without custom schema (default behavior)
  python generate_task_state_predictions.py --model anthropic/claude-sonnet-4.5
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="LLM model to use for predictions (default: anthropic/claude-sonnet-4.5)"
    )
    
    parser.add_argument(
        "--custom-schema-path",
        type=str,
        default=None,
        help="Path to directory containing custom schema JSON files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

async def main():
    """Main execution function"""
    print("🎯 Task State Prediction Generator")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"🤖 Using model: {args.model}")
    
    if args.custom_schema_path:
        # Validate custom schema path exists
        if not Path(args.custom_schema_path).exists():
            print(f"❌ Custom schema path does not exist: {args.custom_schema_path}")
            return 1
        print(f"📁 Using custom schema path: {args.custom_schema_path}")
    else:
        print("📁 Using default schema (all_table_schemas.json)")
    
    if args.output:
        print(f"📁 Output path: {args.output}")
    
    if args.verbose:
        print("🔍 Verbose mode enabled")
    
    # Initialize predictor
    predictor = TaskStatePredictor(model=args.model, custom_schema_path=args.custom_schema_path)
    
    try:
        # Initialize components
        await predictor.initialize()
        
        # Generate predictions for all tasks
        await predictor.generate_all_predictions()
        
        # Save results
        output_file = predictor.save_results(args.output)
        
        print(f"\n🎉 State prediction generation complete!")
        print(f"📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
