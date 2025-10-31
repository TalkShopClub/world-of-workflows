#!/usr/bin/env python3
"""
Example usage of the world-of-workflows package.

This demonstrates:
1. Generating state predictions from action sequences
2. Generating action predictions from state changes
3. Evaluating constraint-based task completion
4. Evaluating LLM-generated responses
"""

import asyncio
import json
from pathlib import Path
import os

from wow.environment.agent import WorldModelAgent, ActionCall, StateDiff
from wow.environment.generators.actions import generate_action_predictions
from wow.environment.generators.states import generate_task_state_predictions
from wow.environment.generators.constraint_tasks import generate_constraint_violation_predictions
from wow.environment.evaluation import (
    evaluate_constraint_predictions,
    StatePredictionEvaluator,
    ActionPredictionEvaluator
)


async def example_state_prediction():
    """Example: Generate state predictions from action sequence"""
    print("\n" + "="*60)
    print("EXAMPLE 1: State Prediction")
    print("="*60)

    agent = WorldModelAgent(model="anthropic/claude-sonnet-4.5")
    await agent.initialize_mcp_server("full")

    actions = [
        {"tool_name": "create_incident", "parameters": {"short_description": "Server down"}},
        {"tool_name": "update_incident", "parameters": {"sys_id": "inc123", "state": "2"}}
    ]

    predicted_states = await agent.predict_states(actions, task="create_and_update_incident")

    print(f"Generated {len(predicted_states)} state predictions")
    for i, state in enumerate(predicted_states):
        print(f"\nState {i+1}:")
        print(f"  Tables modified: {state.additional_information.tables_modified}")
        print(f"  Num audits: {state.additional_information.num_audits}")


async def example_action_prediction():
    """Example: Generate action predictions from state changes"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Action Prediction")
    print("="*60)

    agent = WorldModelAgent(model="openai/o3")

    trajectory_file = Path("src/wow/data_files/trajectories/assignuserrole.json") # Picking one of the trajectories 
    if not trajectory_file.exists(): 
        return print(f"Trajectory file not found: {trajectory_file}")

    os.makedirs('example_predictions', exist_ok=True)
    output_file = Path('example_predictions/assignuserrole_action_prediction.json')

    results = await generate_action_predictions(
        trajectory_file=trajectory_file,
        agent=agent,
        max_state_diffs=20, 
        output_file=output_file
    )

    print(f"Action predictions saved to: {results['output_file']}")


async def example_custom_schema_action_prediction():
    """Example: Generate action predictions using custom schema"""
    print("\n" + "="*60)
    print("EXAMPLE 2b: Custom Schema Action Prediction")
    print("="*60)

    agent = WorldModelAgent(model="openai/o3")

    trajectory_file = Path("src/wow/data_files/trajectories/assignuserrole.json")
    if not trajectory_file.exists(): 
        return print(f"Trajectory file not found: {trajectory_file}")

    # Use custom schema from anthropic/claude-sonnet-4.5 directory
    custom_schema_path = Path("src/wow/data_files/schemas/anthropic/claude-sonnet-4.5")
    if not custom_schema_path.exists():
        return print(f"Custom schema path not found: {custom_schema_path}")

    os.makedirs('example_predictions', exist_ok=True)
    output_file = Path('example_predictions/assignuserrole_custom_schema_action_prediction.json')

    results = await generate_action_predictions(
        trajectory_file=trajectory_file,
        agent=agent,
        max_state_diffs=10,  # Limit to 10 to avoid token overflow
        custom_schema_path=str(custom_schema_path),
        output_file=output_file
    )

    if results:
        print(f"✅ Custom schema action predictions saved to: {results['output_file']}")
        print(f"   Predicted {results['predicted_actions']} actions")
        print(f"   Processing time: {results['processing_time_seconds']:.2f} seconds")
        print(f"   Custom schema used: {results['custom_schema_used']}")
    else:
        print("❌ Failed to generate predictions")


async def example_constraint_prediction(output_file: Path):
    """Example: Generate constraint violation predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Constraint Violation Prediction")
    print("="*60)

    agent = WorldModelAgent(model="openai/o3")
    await agent.initialize_mcp_server("full")

    data_folder = Path("src/wow/data_files/constraint_violation_data")
    if not data_folder.exists():
        print(f"Constraint data folder not found: {data_folder}")
        return

    results = await generate_constraint_violation_predictions(
        agent=agent,
        data_folder=data_folder,
        trajectory_type="original",
        mode="state_action",
        perfect_schema=True,
        output_file=output_file
    )

    print(f"Processed {len(results['results'])} trajectories")
    print(f"Total cost: ${results['metadata']['total_cost']:.4f}")
    print(f"JSON errors: {results['metadata']['total_json_errors']}")


async def example_state_evaluation():
    """Example: Evaluate state predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: State Prediction Evaluation")
    print("="*60)

    model_name = "claude-sonnet-4.5"
    evaluator = StatePredictionEvaluator(model_name)
    
    # Evaluate state predictions for k=1, 2, 3, 4, 5
    results = evaluator.evaluate_state_predictions(k_values=[1, 2, 3, 4, 5])
    
    if 'error' in results:
        return print(f"❌ Error: {results.get('error')}")
    print("\n📊 State Evaluation Results:")
    for k_key, k_data in results.get('k_evaluations', {}).items():
        if k_data.get('status') != 'success': continue
        print(f"\n  {k_key}:")
        print(f"    Full Match Rate: {k_data.get('full_match_rate', 0):.3f}")
        print(f"    Avg SysAudit IoU: {k_data.get('avg_sysaudit_iou', 0):.3f}")
        print(f"    Total Steps: {k_data.get('total_steps', 0)}")
        print(f"    Files Evaluated: {k_data.get('num_files', 0)}")


async def example_action_evaluation():
    """Example: Evaluate action predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Action Prediction Evaluation")
    print("="*60)

    model_name = "claude-sonnet-4.5"
    evaluator = ActionPredictionEvaluator(model_name)
    results = evaluator.evaluate_action_predictions()
    
    if 'error' not in results:
        metrics = results.get('metrics', {})
        print("\n📊 Action Evaluation Results:")
        print(f"    Tool Name Accuracy: {metrics.get('tool_name_accuracy', 0):.3f}")
        print(f"    Perfect Match Accuracy: {metrics.get('perfect_match_accuracy', 0):.3f}")
        print(f"    Total Actions: {metrics.get('total_actions', 0)}")
        print(f"    Tool Name Matches: {metrics.get('tool_name_matches', 0)}")
        print(f"    Perfect Matches: {metrics.get('perfect_matches', 0)}")
    else:
        print(f"❌ Error: {results.get('error')}")

async def example_run_action(): 
    """Example: Run an action and save the result to a file"""

    agent = WorldModelAgent(model="openai/o3")
    await agent.initialize_mcp_server("full")

    out_file = Path('example_predictions/test_action.json')

    tool_name = "create_user"
    tool_params = {
        "user_name": "marcus.chen",
        "first_name": "Marcus",
        "last_name": "Chen",
        "email": "marcus.chen@example.com",
    }
    await agent.run_mcp_action(tool_name, tool_params, out_file)

async def main():
    """Run all examples"""

    os.makedirs('example_predictions', exist_ok=True)

    # await example_state_prediction()
    # await example_action_prediction()
    await example_custom_schema_action_prediction()

    # output_file = Path('example_predictions/original_constraint_prediction.json')
    # await example_constraint_prediction(output_file=output_file)

    # # Evaluate the constraint prediction
    # print("\n" + "="*60)
    # print("Evaluating Constraint Predictions")
    # print("="*60)

    # evaluation_results = evaluate_constraint_predictions(output_file)
    # print(f"Accuracy: {evaluation_results['accuracy']:.2%}")
    # await example_run_action()


if __name__ == "__main__":
    asyncio.run(main())
