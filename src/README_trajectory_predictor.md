# Trajectory State Predictor

A script for predicting state changes for the first k actions in a trajectory file using different language models.

## Usage

```bash
python predict_trajectory_states.py <model_name> <k> <trajectory_file>
```

### Arguments

- `model_name`: The language model to use for predictions (e.g., "anthropic/claude-sonnet-4", "openai/gpt-4o")
- `k`: Number of actions to predict (predicts actions 1 to k)
- `trajectory_file`: Path to the trajectory JSON file

### Examples

```bash
# Predict first 3 actions using Claude Sonnet 4
python predict_trajectory_states.py anthropic/claude-sonnet-4 3 trajectories/incident.json

# Predict first 2 actions using GPT-4o
python predict_trajectory_states.py openai/gpt-4o 2 trajectories/transferasset.json

# Predict first 5 actions using Claude Sonnet 4
python predict_trajectory_states.py anthropic/claude-sonnet-4 5 trajectories/catalogitem.json
```

## Output

The script generates a JSON file organized by model name in the `rest_apis/world_model_scripts/state_prediction_results/` subfolder with the following structure:

```
state_prediction_results/
├── anthropic_claude-sonnet-4/
│   └── taskname_k2_20251006_011654.json
├── openai_gpt-4o/
│   └── taskname_k1_20251006_011711.json
└── ...
```

```json
{
  "metadata": {
    "model": "anthropic/claude-sonnet-4",
    "k": 3,
    "trajectory_file": "trajectories/incident.json",
    "timestamp": "2025-01-06T01:09:20.123456",
    "total_actions_in_trajectory": 4,
    "actions_predicted": 3
  },
  "action_calls": [
    {
      "tool_name": "create_incident",
      "parameters": {...}
    }
  ],
  "predicted_states": [
    {
      "sysauditrecord": [...],
      "additional_information": {...}
    }
  ]
}
```

## Features

- **Flexible Model Support**: Works with any model supported by the world model agent
- **Configurable Prediction Length**: Choose how many actions to predict (1 to k)
- **Detailed Logging**: Shows progress and statistics during prediction
- **JSON Output**: Saves results organized by model name in the `rest_apis/world_model_scripts/state_prediction_results/` subfolder in a structured format for further analysis
- **Error Handling**: Gracefully handles prediction failures

## Requirements

- ServiceNow instance credentials (SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME, SNOW_INSTANCE_PWD)
- Valid trajectory files in the expected format
- World model agent and MCP server initialized

## Notes

- The script predicts states for actions 1 through k (inclusive)
- If k is greater than the number of actions in the trajectory, it will predict all available actions
- Each prediction includes both audit records and additional information
- Results are saved with a timestamped filename for easy identification
