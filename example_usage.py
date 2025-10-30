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

from src.wow.environment.agent import WorldModelAgent, ActionCall, StateDiff
from src.wow.environment.generators.actions import generate_action_predictions
from src.wow.environment.generators.states import generate_task_state_predictions
from src.wow.environment.generators.constraint_tasks import generate_constraint_violation_predictions
from src.eval_pipeline import (
    state_rollout_evaluation,
    inverse_action_rollout_evaluation,
    constraint_violation_evaluation
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

    agent = WorldModelAgent(model="anthropic/claude-sonnet-4.5")

    trajectory_file = Path("src/wow/data_files/trajectories/sample_trajectory.json")
    if not trajectory_file.exists():
        print(f"Trajectory file not found: {trajectory_file}")
        return

    results = await generate_action_predictions(
        trajectory_file=trajectory_file,
        agent=agent,
        max_state_diffs=20
    )

    if results:
        print(f"Predicted {results['predicted_actions']} actions")
        print(f"Processing time: {results['processing_time_seconds']:.2f}s")


async def example_constraint_prediction():
    """Example: Generate constraint violation predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Constraint Violation Prediction")
    print("="*60)

    agent = WorldModelAgent(model="anthropic/claude-sonnet-4.5")
    await agent.initialize_mcp_server("full")

    data_folder = Path("src/wow/data_files/constraint_violation_data")
    if not data_folder.exists():
        print(f"Constraint data folder not found: {data_folder}")
        return

    results = await generate_constraint_violation_predictions(
        agent=agent,
        data_folder=data_folder,
        trajectory_type="combined",
        mode="state_action",
        perfect_schema=True
    )

    print(f"Processed {len(results['results'])} trajectories")
    print(f"Total cost: ${results['metadata']['total_cost']:.4f}")
    print(f"JSON errors: {results['metadata']['total_json_errors']}")


async def example_constraint_evaluation():
    """Example: Evaluate constraint violation predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Constraint Violation Evaluation")
    print("="*60)

    pred_constraint_nums = [1, 3]
    pred_invalid_action_idxs = [2, 5]
    gt_constraint_nums = [1, 3]
    gt_invalid_action_idxs = [2, 5]

    accuracy = constraint_violation_evaluation(
        pred_constraint_nums,
        pred_invalid_action_idxs,
        gt_constraint_nums,
        gt_invalid_action_idxs
    )

    print(f"Constraint violation accuracy: {accuracy}")


async def example_state_evaluation():
    """Example: Evaluate state predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 5: State Prediction Evaluation")
    print("="*60)

    from src.wow.environment.agent import SysAuditRecord, AdditionalInformation, operation

    predicted_state = StateDiff(
        sysauditrecord=[
            SysAuditRecord(
                tablename="incident",
                fieldname="state",
                oldvalue="1",
                newvalue="2"
            )
        ],
        additional_information=AdditionalInformation(
            num_audits=1,
            num_modified_entries=1,
            num_deleted_entries=0,
            num_created_entries=0,
            operation_type=[operation.put],
            tables_modified=["incident"]
        )
    )

    gt_state = StateDiff(
        sysauditrecord=[
            SysAuditRecord(
                tablename="incident",
                fieldname="state",
                oldvalue="1",
                newvalue="2"
            )
        ],
        additional_information=AdditionalInformation(
            num_audits=1,
            num_modified_entries=1,
            num_deleted_entries=0,
            num_created_entries=0,
            operation_type=[operation.put],
            tables_modified=["incident"]
        )
    )

    full_acc, partial_acc, side_effects = await state_rollout_evaluation(
        [predicted_state],
        [gt_state]
    )

    print(f"Full accuracy: {full_acc}")
    print(f"Partial accuracy (sysaudit, additional_info): {partial_acc}")
    print(f"Side effects: {side_effects}")


async def example_action_evaluation():
    """Example: Evaluate action predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Action Prediction Evaluation")
    print("="*60)

    predicted_actions = [
        ActionCall(
            tool_name="create_incident",
            parameters={"short_description": "Server down"}
        )
    ]

    gt_actions = [
        ActionCall(
            tool_name="create_incident",
            parameters={"short_description": "Server down"}
        )
    ]

    metrics = inverse_action_rollout_evaluation(predicted_actions, gt_actions)

    for i, (tool_match, param_match, diffs) in enumerate(metrics):
        print(f"\nAction {i+1}:")
        print(f"  Tool match: {tool_match}")
        print(f"  Param match: {param_match:.2%}")
        print(f"  Diffs: {diffs}")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("World of Workflows - Example Usage")
    print("="*60)

    await example_state_prediction()
    await example_action_prediction()
    await example_constraint_prediction()
    await example_constraint_evaluation()
    await example_state_evaluation()
    await example_action_evaluation()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
