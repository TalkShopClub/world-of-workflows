# Custom Schema Support for State Prediction

This document describes how to use custom task-specific schemas for state prediction in the WorkArena world model.

## Overview

The custom schema functionality allows you to use task-specific JSON schema files instead of the default `all_table_schemas.json` for more focused and efficient state predictions. This is particularly useful when you have specific task schemas that contain only the relevant tool calls and context for a particular task.

## Schema Format

Custom schema files should be JSON files with the following structure:

```json
{
  "trajectory_file": "task_name.json",
  "generated_at": "2025-10-14T15:48:19.961956",
  "num_actions": 4,
  "num_tool_calls": 100,
  "schema_context_size": 322197,
  "tool_calls": [
    {
      "observation_id": "270fa9ac84be30ba",
      "name": "list_incidents",
      "type": "TOOL",
      "input": "{'number': 'INC0010091', 'limit': 1}",
      "output": {
        "success": true,
        "message": "Found 1 incidents",
        "incidents": [...]
      },
      "start_time": "2025-10-14T19:47:54.886000+00:00",
      "end_time": "2025-10-14T19:47:55.056000+00:00"
    }
  ]
}
```

The key component is the `tool_calls` array, which contains the tool specifications that will be used for state prediction.

## Usage

### 1. Using the WorldModelAgent directly

```python
from rest_apis.world_model_scripts.world_model_agent import WorldModelAgent

# Initialize agent
agent = WorldModelAgent(model="anthropic/claude-sonnet-4")
await agent.initialize_mcp_server("full")

# Define actions
actions = [
    {
        "tool_name": "create_incident",
        "parameters": {
            "short_description": "Test incident",
            "description": "Testing custom schema"
        }
    }
]

# Use custom schema for prediction
custom_schema_path = "/path/to/custom/schemas"
predicted_states = await agent.predict_states_custom(
    actions=actions,
    task="incident",  # This will look for incident.json in the schema path
    custom_schema_path=custom_schema_path
)
```

### 2. Using the generate_task_state_predictions.py script

```bash
# Using custom schema
python generate_task_state_predictions.py \
    "anthropic/claude-sonnet-4" \
    "/path/to/custom/schemas" \
    "output_file.json"

# Without custom schema (uses default)
python generate_task_state_predictions.py \
    "anthropic/claude-sonnet-4" \
    "" \
    "output_file.json"
```

### 3. Command Line Arguments

The script accepts the following arguments:
1. `model` (optional): LLM model to use (default: "anthropic/claude-sonnet-4")
2. `custom_schema_path` (optional): Path to directory containing custom schema JSON files
3. `output_path` (optional): Path for output file

## Schema File Naming

Custom schema files should be named according to the task name in lowercase. For example:
- `incident.json` for incident-related tasks
- `catalogitem.json` for catalog item tasks
- `transferasset.json` for asset transfer tasks

## Benefits

1. **Focused Context**: Only includes tool calls relevant to the specific task
2. **Reduced Token Usage**: Smaller context means lower API costs
3. **Better Performance**: More focused predictions with task-specific context
4. **Flexibility**: Easy to customize schemas for different tasks

## Example Schema Directory Structure

```
custom_schemas/
├── incident.json
├── catalogitem.json
├── transferasset.json
├── createhardwareasset.json
└── ...
```

## Testing

Run the test script to verify the custom schema functionality:

```bash
python test_custom_schema.py
```

This will test the custom schema prompt generation and validate that the functionality works correctly.
