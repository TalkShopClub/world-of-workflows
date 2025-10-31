import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def _extract_action_calls(mcp_data: Dict) -> List:
    """Extract ActionCall objects from task MCP data"""
    from ..agent import ActionCall

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


async def generate_task_state_predictions(
    task,
    task_name: str,
    agent,
    output_file: Path = None,
    custom_schema_path: str = None
):
    """
    Run a task and generate state predictions.

    Args:
        task: Task instance to run
        task_name: Name of the task
        agent: WorldModelAgent instance
        output_file: Output file path. If None, auto-generates
        custom_schema_path: Optional path to custom schemas directory
    """
    task_result = task.run()

    if not task_result or not task_result.get("success"):
        return {
            "task_name": task_name,
            "success": False,
            "error": "Task execution failed",
            "mcp_data": task_result.get("mcp_data", {}) if task_result else {}
        }

    mcp_data = task_result.get("mcp_data", {})
    action_calls = _extract_action_calls(mcp_data)

    if not action_calls:
        return {
            "task_name": task_name,
            "success": False,
            "error": "No action calls found",
            "mcp_data": mcp_data
        }

    actions_dict = []
    for action_call in action_calls:
        actions_dict.append({
            "tool_name": action_call.tool_name,
            "parameters": action_call.parameters
        })

    if custom_schema_path:
        predicted_states = await agent.predict_states_custom(actions_dict, task_name, custom_schema_path)
    else:
        predicted_states = await agent.predict_states(actions_dict, task_name)

    predicted_states_dict = []
    for state in predicted_states:
        predicted_states_dict.append(state.model_dump(mode='json'))

    results = {
        "task_name": task_name,
        "success": True,
        "action_calls": [action.model_dump(mode='json') for action in action_calls],
        "predicted_states": predicted_states_dict,
        "mcp_data": mcp_data,
        "num_actions": len(action_calls),
        "num_predictions": len(predicted_states)
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results
