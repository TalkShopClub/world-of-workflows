# Task State Prediction and Evaluation

This directory contains scripts for generating state predictions for ServiceNow tasks and evaluating them using the world model evaluation pipeline.

## Overview

The evaluation pipeline consists of three main components:

1. **State Prediction Generation**: Generate state predictions for tasks using the WorldModelAgent
2. **Ground Truth Generation**: Execute tasks to capture actual state changes
3. **Evaluation**: Compare predictions with ground truth using metrics from `eval_pipeline.py`

## Scripts

### 1. `generate_task_state_predictions.py`
Generates state predictions for all tasks defined in `custom/tasks.py`.

**Usage:**
```bash
python generate_task_state_predictions.py [model] [output_file]
```

**Example:**
```bash
python generate_task_state_predictions.py openai/gpt-4o task_predictions.json
```

### 2. `evaluate_task_predictions.py`
Evaluates previously generated predictions (basic evaluation).

**Usage:**
```bash
python evaluate_task_predictions.py <predictions_file> [output_file]
```

**Example:**
```bash
python evaluate_task_predictions.py task_predictions.json evaluation_results.json
```

### 3. `complete_task_evaluation.py`
Complete evaluation pipeline that generates predictions, captures ground truth, and evaluates them.

**Usage:**
```bash
python complete_task_evaluation.py [model] [output_file] [task_subset]
```

**Example:**
```bash
python complete_task_evaluation.py openai/gpt-4o results.json TransferAsset,Incident
```

### 4. `run_task_evaluation.py`
Convenient wrapper script with command-line options.

**Usage:**
```bash
python run_task_evaluation.py --help
python run_task_evaluation.py --model gpt-4o --tasks TransferAsset,Incident
python run_task_evaluation.py --quick  # Run only 3 simple tasks
python run_task_evaluation.py --list-tasks  # List all available tasks
```

## Available Tasks

The following tasks are available for evaluation:

- `TransferAsset`: Transfer asset between users
- `UserGroupAsset`: User, group, and asset management
- `KnowledgeBaseArticle`: Knowledge base and article management
- `Incident`: Incident creation and management
- `CatalogItem`: Service catalog item management
- `ApproveChangeRequest`: Change request approval workflow
- `RejectChangeRequest`: Change request rejection workflow
- `AssignUserRole`: User role assignment
- `ChangeUserInfo`: User information updates
- `CreateHierarchy`: Group hierarchy creation
- `RemoveUserFromGroup`: Group membership management
- `PublishKnowledgeBaseArticle`: Knowledge article publishing
- `MoveCatalogItem`: Catalog item movement
- `DeactivateUser`: User deactivation
- `PromoteUser`: User promotion and management
- `CreateRequest`: Item request creation
- `CreateHardwareAsset`: Hardware asset creation

## Evaluation Metrics

The evaluation uses metrics from `eval_pipeline.py`:

1. **Full State Rollout Accuracy**: Whether all ground truth state changes are captured in predictions
2. **Partial State Rollout Accuracy**: Percentage of ground truth changes captured (separate for sysaudit records and additional information)
3. **Side Effects**: Number of predicted changes not present in ground truth

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "evaluation_info": {
    "timestamp": "2024-01-01T12:00:00",
    "model": "openai/gpt-4o",
    "total_tasks": 17,
    "successful_tasks": 15,
    "failed_tasks": 2
  },
  "overall_metrics": {
    "avg_full_accuracy": 0.85,
    "avg_sysaudit_accuracy": 0.92,
    "avg_additional_accuracy": 0.78,
    "total_side_effects": 12,
    "avg_side_effects_per_task": 0.8
  },
  "task_results": [
    {
      "task_name": "TransferAsset",
      "success": true,
      "num_actions": 7,
      "num_predictions": 7,
      "num_ground_truth": 7,
      "metrics": {
        "full_accuracy": [1.0, 0.0, 1.0, ...],
        "partial_accuracy": [(1.0, 1.0), (0.5, 0.8), ...],
        "side_effects": [(0, []), (2, [...]), ...],
        "avg_full_accuracy": 0.85,
        "avg_sysaudit_accuracy": 0.92,
        "avg_additional_accuracy": 0.78,
        "total_side_effects": 3
      },
      "action_calls": [...],
      "predicted_states": [...],
      "ground_truth_states": [...],
      "mcp_data": {...}
    }
  ]
}
```

## Prerequisites

1. **Environment Variables**: Ensure the following are set:
   - `SNOW_INSTANCE_URL`: ServiceNow instance URL
   - `SNOW_INSTANCE_UNAME`: ServiceNow username
   - `SNOW_INSTANCE_PWD`: ServiceNow password
   - `OPENROUTER_API_KEY`: API key for OpenRouter (for LLM access)

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **ServiceNow Instance**: Ensure you have access to a ServiceNow instance for task execution.

## Quick Start

1. **Run a quick test** (3 simple tasks):
   ```bash
   python run_task_evaluation.py --quick
   ```

2. **Run specific tasks**:
   ```bash
   python run_task_evaluation.py --tasks TransferAsset,Incident,DeactivateUser
   ```

3. **Run all tasks with custom model**:
   ```bash
   python run_task_evaluation.py --model openai/gpt-4o-mini --output my_results.json
   ```

## Troubleshooting

- **Task execution failures**: Check ServiceNow instance connectivity and permissions
- **Prediction generation failures**: Verify OpenRouter API key and model availability
- **Memory issues**: Use task subsets to reduce memory usage
- **Timeout issues**: Increase delays between actions in the scripts

## Integration with eval_pipeline.py

The generated predictions can be directly used with the existing `eval_pipeline.py` evaluation functions:

```python
from eval_pipeline import state_rollout_evaluation, compute_state_rollout_metrics

# Load your predictions
with open('task_predictions.json', 'r') as f:
    data = json.load(f)

# Extract predicted states and ground truth
predicted_states = [StateDiff(**state) for state in data['predicted_states']]
gt_states = [StateDiff(**state) for state in data['ground_truth_states']]

# Run evaluation
full_acc, partial_acc, side_effects = compute_state_rollout_metrics(
    predicted_states, gt_states
)
```
