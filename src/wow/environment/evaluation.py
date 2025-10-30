import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def constraint_violation_evaluation(
    pred_constraint_nums: List[int],
    pred_invalid_action_idxs: List[int],
    gt_constraint_nums: List[int],
    gt_invalid_action_idxs: List[int]
) -> int:
    """
    Evaluate world model's constraint violation accuracy for a given policy number and invalid action index.

    Args:
        pred_constraint_nums: Predicted constraint numbers that are violated (-1 if no constraint is violated)
        pred_invalid_action_idxs: Predicted invalid action indices within the sequence of actions that violates the constraint (-1 if no constraint is violated)
        gt_constraint_nums: Ground truth constraint numbers that are violated (-1 if no constraint is violated)
        gt_invalid_action_idxs: Ground truth invalid action indices within the sequence of actions that violates the constraint (-1 if no constraint is violated)

    Returns:
        constraint_violation_accuracy: Constraint violation accuracy (1 if the predicted constraint number and invalid action index match the ground truth, 0 otherwise)
    """
    pred_pairs = set(zip(pred_constraint_nums, pred_invalid_action_idxs))
    gt_pairs = set(zip(gt_constraint_nums, gt_invalid_action_idxs))

    return 1 if pred_pairs == gt_pairs else 0


def evaluate_constraint_predictions(predictions_file: Path) -> Dict[str, Any]:
    """
    Evaluate constraint violation predictions from a JSON file.

    Args:
        predictions_file: Path to JSON file containing constraint predictions
                         Expected format: {"results": [{"gt_answer_idxs": [...],
                                                        "gt_constraint_nums": [...],
                                                        "llm_answer_idxs": [...],
                                                        "llm_constraint_nums": [...]}]}

    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Overall constraint violation accuracy
        - total_predictions: Total number of predictions evaluated
        - accuracies_per_prediction: List of individual accuracies
    """
    with open(predictions_file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    if not results:
        return {
            "accuracy": 0.0,
            "total_predictions": 0,
            "accuracies_per_prediction": []
        }

    constraint_violation_accuracies = []

    for result in tqdm(results, desc="Evaluating constraint predictions"):
        pred_constraint_nums = result.get("llm_constraint_nums", [])
        pred_invalid_action_idxs = result.get("llm_answer_idxs", [])
        gt_constraint_nums = result.get("gt_constraint_nums", [])
        gt_invalid_action_idxs = result.get("gt_answer_idxs", [])

        accuracy = constraint_violation_evaluation(
            pred_constraint_nums,
            pred_invalid_action_idxs,
            gt_constraint_nums,
            gt_invalid_action_idxs
        )
        constraint_violation_accuracies.append(accuracy)

    return {
        "accuracy": float(np.mean(constraint_violation_accuracies)),
        "total_predictions": len(results),
        "accuracies_per_prediction": constraint_violation_accuracies
    }
