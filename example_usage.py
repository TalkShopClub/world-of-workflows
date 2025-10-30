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
from src.eval_pipeline import constraint_violation_evaluation


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

    predicted_states = await agent.predict_states(actions, task_name="create_and_update_incident")

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
        print(f"Trajectory file not found: {trajectory_file}")
        return

    os.makedirs('example_predictions', exist_ok=True)
    output_file = Path('example_predictions/assignuserrole_action_prediction.json')

    results = await generate_action_predictions(
        trajectory_file=trajectory_file,
        agent=agent,
        max_state_diffs=20, 
        output_file=output_file
    )

    print(f"Action predictions saved to: {results['output_file']}")


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


async def main():
    """Run all examples"""

    # await example_state_prediction()
    # await example_action_prediction()

    os.makedirs('example_predictions', exist_ok=True)
    output_file = Path('example_predictions/original_constraint_prediction.json')
    await example_constraint_prediction(output_file=output_file)

    # Evaluate the constraint prediction 



if __name__ == "__main__":
    asyncio.run(main())
