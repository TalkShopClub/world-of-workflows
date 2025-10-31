import json
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def _convert_audits_to_state_diff(audits):
    """Convert audit records to state_diff format"""
    from ..agent import StateDiff, SysAuditRecord, AdditionalInformation, operation

    sysaudit_records = []
    for audit in audits:
        sysaudit_records.append(SysAuditRecord(
            fieldname=audit.get('fieldname', ''),
            newvalue=audit.get('newvalue', ''),
            tablename=audit.get('tablename', ''),
            oldvalue=audit.get('oldvalue', '')
        ))

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

    num_created = len([a for a in audits if a.get('oldvalue', '') == '' and a.get('newvalue', '') != ''])
    num_modified = len([a for a in audits if a.get('oldvalue', '') != '' and a.get('newvalue', '') != '' and a.get('newvalue', '') != 'DELETED'])
    num_deleted = len([a for a in audits if a.get('newvalue', '') == 'DELETED' and a.get('oldvalue', '') != ''])

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


def _extract_state_diffs_from_trajectory(trajectory: List[Dict], max_state_diffs: int = 100) -> List[Dict]:
    """Extract state diffs from trajectory data"""
    state_diffs = []
    for step in trajectory:
        state_diff = None

        if "state_diff" in step:
            try:
                if isinstance(step["state_diff"], str):
                    state_diff = json.loads(step["state_diff"])
                else:
                    state_diff = step["state_diff"]
            except (json.JSONDecodeError, TypeError):
                continue

        elif "ground_truth_state" in step:
            try:
                if isinstance(step["ground_truth_state"], str):
                    state_diff = json.loads(step["ground_truth_state"])
                else:
                    state_diff = step["ground_truth_state"]
            except (json.JSONDecodeError, TypeError):
                continue

        elif "audits" in step and step["audits"]:
            try:
                state_diff = _convert_audits_to_state_diff(step["audits"])
            except Exception:
                continue

        if state_diff:
            state_diffs.append(state_diff)

    if len(state_diffs) > max_state_diffs:
        state_diffs = state_diffs[:max_state_diffs]

    return state_diffs


async def generate_action_predictions(
    trajectory_file: Path,
    agent,
    output_file: Path = None,
    custom_schema_path: str = None,
    max_state_diffs: int = 100
):
    """
    Generate action predictions for a single trajectory.

    Args:
        trajectory_file: Path to trajectory JSON file
        agent: WorldModelAgent instance
        output_file: Output file path. If None, auto-generates based on trajectory file
        custom_schema_path: Optional path to custom schemas directory
        max_state_diffs: Maximum number of state diffs to process
    """
    with open(trajectory_file, "r") as f:
        trajectory = json.load(f)

    task_name = trajectory_file.stem
    state_diffs = _extract_state_diffs_from_trajectory(trajectory, max_state_diffs)

    if not state_diffs:
        return None

    start_time = time.time()

    if custom_schema_path and task_name:
        predicted_actions = await agent.predict_actions_custom(state_diffs, task_name, custom_schema_path)
    else:
        predicted_actions = await agent.predict_actions(state_diffs, task_name)

    end_time = time.time()

    results = {
        "trajectory_file": trajectory_file.name,
        "task_name": task_name,
        "total_steps": len(trajectory),
        "valid_state_diffs": len(state_diffs),
        "predicted_actions": len(predicted_actions),
        "processing_time_seconds": end_time - start_time,
        "custom_schema_used": custom_schema_path is not None,
        "predictions": [action.model_dump() for action in predicted_actions]
    }

    if output_file is None:
        base_dir = Path(__file__).parent.parent.parent
        output_dir = base_dir / "action_predictions_results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{trajectory_file.stem}_action_predictions.json"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Add output_file to results for caller convenience
    results["output_file"] = str(output_file)

    return results
