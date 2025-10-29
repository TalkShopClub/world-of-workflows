#!/usr/bin/env python3
"""
Simple script to run task evaluation with different configurations

Usage examples:
    python run_task_evaluation.py --model gpt-4o --tasks TransferAsset,Incident
    python run_task_evaluation.py --model gpt-4o-mini --output results.json
    python run_task_evaluation.py --quick  # Run only a few tasks for testing
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from complete_task_evaluation import CompleteTaskEvaluator

async def run_evaluation(model: str, output_path: str = None, task_subset: list = None, quick: bool = False):
    """Run the complete task evaluation"""
    
    if quick:
        # Run only a few simple tasks for quick testing
        task_subset = ["DeactivateUser", "ChangeUserInfo", "CreateHardwareAsset"]
        print("🏃 Quick mode: Running only 3 simple tasks")
    
    print(f"🚀 Starting evaluation with model: {model}")
    if task_subset:
        print(f"📋 Task subset: {task_subset}")
    if output_path:
        print(f"📁 Output: {output_path}")
    
    # Initialize evaluator
    evaluator = CompleteTaskEvaluator(model=model)
    
    try:
        # Initialize components
        await evaluator.initialize()
        
        # Run evaluation
        await evaluator.evaluate_all_tasks(task_subset)
        
        # Save results
        output_file = evaluator.save_results(output_path)
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Run task evaluation pipeline")
    
    parser.add_argument(
        "--model", 
        default="anthropic/claude-sonnet-4",
        help="Model to use for predictions (default: anthropic/claude-sonnet-4)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results"
    )
    
    parser.add_argument(
        "--tasks", "-t",
        help="Comma-separated list of task names to evaluate"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only a few simple tasks for quick testing"
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available task names and exit"
    )
    
    args = parser.parse_args()
    
    # List available tasks
    if args.list_tasks:
        available_tasks = [
            "TransferAsset", "UserGroupAsset", "KnowledgeBaseArticle", "Incident",
            "CatalogItem", "ApproveChangeRequest", "RejectChangeRequest", "AssignUserRole",
            "ChangeUserInfo", "CreateHierarchy", "RemoveUserFromGroup", "PublishKnowledgeBaseArticle",
            "MoveCatalogItem", "DeactivateUser", "PromoteUser", "CreateRequest", "CreateHardwareAsset"
        ]
        print("Available tasks:")
        for task in available_tasks:
            print(f"  - {task}")
        return 0
    
    # Parse task subset
    task_subset = None
    if args.tasks:
        task_subset = [task.strip() for task in args.tasks.split(',')]
        print(f"Selected tasks: {task_subset}")
    
    # Run evaluation
    return asyncio.run(run_evaluation(
        model=args.model,
        output_path=args.output,
        task_subset=task_subset,
        quick=args.quick
    ))

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
