'''
Evaluation pipeline for the world model. Supports three evaluation metrics: 
1. Accuracy of the world model in predicting the state changes rollout for a given sequence of actions. 
2. Accuracy of the world model in inverse prediction of actions for a given state changes rollout. 
3. Accuracy of the world model for Q/A dataset where the world model is given a hard constraint and evaluated on whether it can identify which actions from a sequence of actions lead to bad states. 
''' 

from typing import List, Tuple, Optional
import os 
import asyncio
from datetime import datetime
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

# Import local modules (support both direct execution and module import)
try:
    from .world_model_agent import (
        StateDiff, 
        ActionCall, 
        WorldModelAgent, 
        SysAuditRecord, 
        operation
    )
except ImportError:
    from world_model_agent import (
        StateDiff, 
        ActionCall, 
        WorldModelAgent, 
        SysAuditRecord, 
        operation
    )

# Import states module if needed
try:
    from .states import get_sys_audit
except ImportError:
    try:
        from states import get_sys_audit
    except ImportError:
        def get_sys_audit(*args, **kwargs):
            return []


def compute_full_rollout_accuracy(pred_state_diff: StateDiff, gt_state_diff: StateDiff) -> float:
    """
    Compute full state rollout accuracy for a single step.
    Returns 1.0 if both sysaudit records and additional information match completely, 0.0 otherwise.
    """
    # Check if all GT sysaudit records are in predicted (order doesn't matter)
    gt_sysaudit_set = set()
    for record in gt_state_diff.sysauditrecord:
        gt_sysaudit_set.add((record.tablename, record.fieldname, record.oldvalue, record.newvalue))
    
    predicted_sysaudit_set = set()
    for record in pred_state_diff.sysauditrecord:
        predicted_sysaudit_set.add((record.tablename, record.fieldname, record.oldvalue, record.newvalue))
    
    # Check if all GT sysaudit records are captured in predicted
    sysaudit_match = gt_sysaudit_set.issubset(predicted_sysaudit_set)
    
    # Check additional information equality
    gt_additional = gt_state_diff.additional_information
    pred_additional = pred_state_diff.additional_information
    
    additional_info_match = (
        gt_additional.num_audits == pred_additional.num_audits and
        gt_additional.num_modified_entries == pred_additional.num_modified_entries and
        gt_additional.num_deleted_entries == pred_additional.num_deleted_entries and
        gt_additional.num_created_entries == pred_additional.num_created_entries and
        set(gt_additional.operation_type) == set(pred_additional.operation_type) and
        set(gt_additional.tables_modified) == set(pred_additional.tables_modified)
    )
    
    # Full accuracy is 1.0 if both sysaudit and additional info match, 0.0 otherwise
    return 1.0 if (sysaudit_match and additional_info_match) else 0.0

def compute_partial_rollout_accuracy(pred_state_diff: StateDiff, gt_state_diff: StateDiff) -> Tuple[float, float]:
    """
    Compute partial state rollout accuracy for a single step using IoU (Intersection over Union).
    Returns tuple of (sysaudit_iou, additional_info_iou).
    """
    # Get sysaudit sets for comparison
    gt_sysaudit_set = set()
    for record in gt_state_diff.sysauditrecord:
        gt_sysaudit_set.add((record.tablename, record.fieldname, record.oldvalue, record.newvalue))
    
    predicted_sysaudit_set = set()
    for record in pred_state_diff.sysauditrecord:
        predicted_sysaudit_set.add((record.tablename, record.fieldname, record.oldvalue, record.newvalue))
    
    # Calculate IoU for sysaudit records: intersection / union
    intersection = gt_sysaudit_set.intersection(predicted_sysaudit_set)
    union = gt_sysaudit_set.union(predicted_sysaudit_set)
    
    if len(union) > 0:
        sysaudit_iou = len(intersection) / len(union)
    else:
        sysaudit_iou = 1.0  # Perfect if no records to compare
    
    # For additional information, calculate IoU for each field
    gt_additional = gt_state_diff.additional_information
    pred_additional = pred_state_diff.additional_information
    
    # Create sets for each additional info field to calculate IoU
    gt_audits_set = {gt_additional.num_audits}
    pred_audits_set = {pred_additional.num_audits}
    audits_iou = len(gt_audits_set.intersection(pred_audits_set)) / len(gt_audits_set.union(pred_audits_set)) if len(gt_audits_set.union(pred_audits_set)) > 0 else 1.0
    
    gt_modified_set = {gt_additional.num_modified_entries}
    pred_modified_set = {pred_additional.num_modified_entries}
    modified_iou = len(gt_modified_set.intersection(pred_modified_set)) / len(gt_modified_set.union(pred_modified_set)) if len(gt_modified_set.union(pred_modified_set)) > 0 else 1.0
    
    gt_deleted_set = {gt_additional.num_deleted_entries}
    pred_deleted_set = {pred_additional.num_deleted_entries}
    deleted_iou = len(gt_deleted_set.intersection(pred_deleted_set)) / len(gt_deleted_set.union(pred_deleted_set)) if len(gt_deleted_set.union(pred_deleted_set)) > 0 else 1.0
    
    gt_created_set = {gt_additional.num_created_entries}
    pred_created_set = {pred_additional.num_created_entries}
    created_iou = len(gt_created_set.intersection(pred_created_set)) / len(gt_created_set.union(pred_created_set)) if len(gt_created_set.union(pred_created_set)) > 0 else 1.0
    
    # For operation_type and tables_modified, use set-based IoU
    gt_ops_set = set(gt_additional.operation_type)
    pred_ops_set = set(pred_additional.operation_type)
    ops_iou = len(gt_ops_set.intersection(pred_ops_set)) / len(gt_ops_set.union(pred_ops_set)) if len(gt_ops_set.union(pred_ops_set)) > 0 else 1.0
    
    gt_tables_set = set(gt_additional.tables_modified)
    pred_tables_set = set(pred_additional.tables_modified)
    tables_iou = len(gt_tables_set.intersection(pred_tables_set)) / len(gt_tables_set.union(pred_tables_set)) if len(gt_tables_set.union(pred_tables_set)) > 0 else 1.0
    
    # Average IoU across all additional info fields
    additional_info_iou = (audits_iou + modified_iou + deleted_iou + created_iou + ops_iou + tables_iou) / 6

    return sysaudit_iou, additional_info_iou

def compute_side_effects(pred_state_diff: StateDiff, gt_state_diff: StateDiff) -> Tuple[int, List[SysAuditRecord]]:
    """
    Compute side effects for a single step.
    Returns tuple of (num_side_effects, side_effect_records).
    Side effects are table-column pairs in predicted but not in ground truth.
    """
    # Find table-column pairs in predicted but not in GT
    gt_table_column = set((r.tablename, r.fieldname) for r in gt_state_diff.sysauditrecord)
    side_effs = [r for r in pred_state_diff.sysauditrecord if (r.tablename, r.fieldname) not in gt_table_column]
    return len(side_effs), side_effs

def compute_state_rollout_metrics(pred_state_diffs: List[StateDiff], gt_state_diffs: List[StateDiff]): 
    """
    Returns full state rollout accuracy, partial state rollout accuracy and number of side effects for each step in the action trajectory. 
    Side effects are defined as unique tables and columns that are in the predicted state diff but not in the ground truth state diff. 
    Full state rollout accuracy is whether all the ground truth state diff are in the predicted state diff. 
    Partial state rollout accuracy is percentage of ground truth state diff that are in the predicted state diff. This is computed separately for sysaudit records and additional information. 

    Returns: 
        full_state_rollout_accuracy: List[float] 
        partial_state_rollout_accuracy: List[Tuple[float, float]] 
        side_effects: List[Tuple[int, List[SysAuditRecord]]] 
    """ 

    full_rollout_acc = []
    part_rollout_acc = []
    side_effects = []

    for pred_state_diff, gt_state_diff in tqdm(zip(pred_state_diffs, gt_state_diffs), total=len(gt_state_diffs), desc="Computing state rollout metrics"):
        full_rollout_acc.append(compute_full_rollout_accuracy(pred_state_diff, gt_state_diff))
        part_rollout_acc.append(compute_partial_rollout_accuracy(pred_state_diff, gt_state_diff))
        side_effects.append(compute_side_effects(pred_state_diff, gt_state_diff))

    return full_rollout_acc, part_rollout_acc, side_effects

async def state_rollout_evaluation(predicted_state_diff: List[StateDiff], gt_state_diffs: List[StateDiff], verbose: bool = False) -> Tuple[List[float], List[Tuple[float, float]], List[Tuple[int, List[SysAuditRecord]]]]: 
    """
    Evaluate state rollout accuracy for each step in the action trajectory. 
    Ground truth states are obtained from the ground truth action trajectory. 
    """ 

    assert len(gt_state_diffs) == len(predicted_state_diff), "Number of ground truth states must match number of predicted states"

    if verbose: 
        # Store ground truth states in a file 
        with open("gt_state_diffs.json", "w") as f:
            # Convert Pydantic models to dictionaries for JSON serialization
            # Use mode='json' to properly serialize enums as strings
            gt_state_diffs_dict = [state_diff.model_dump(mode='json') for state_diff in gt_state_diffs]
            json.dump(gt_state_diffs_dict, f, indent=2)

    # Compute full and partial state rollout accuracy 
    return compute_state_rollout_metrics(predicted_state_diff, gt_state_diffs)

def inverse_action_rollout_evaluation(predicted_action_trajectory: List[ActionCall], gt_action_trajectory: List[ActionCall], verbose: bool = False) -> List[Tuple[int, float, Tuple[int, Tuple[str, List[str]]]]]: 
    """
    Evaluate inverse action rollout accuracy for each step in the action trajectory. 
    Ground truth actions are obtained from the ground truth action trajectory. 

    Returns list of tuples of (tool_match, param_match_percentage, action_diffs) for each step:
    - tool_match: 1 if tool names match, 0 otherwise
    - param_match_percentage: percentage of GT parameters captured in predicted parameters
    - action_diffs: list of tuples where each tuple is:
        * For tool mismatch: (1, (predicted_tool_name, [all_param_strings]))
        * For param mismatch: (num_extra_params, (predicted_tool_name, [extra_param_strings]))
    """ 

    action_rollout_metrics = [] 

    for predicted_action, gt_action in tqdm(
        zip(predicted_action_trajectory, gt_action_trajectory), 
        total=len(predicted_action_trajectory), 
        desc="Evaluating inverse action rollout"):
        
        # Compute tool matches
        tool_match = 1 if predicted_action.tool_name == gt_action.tool_name else 0

        if not tool_match:  # If tool name mismatch, no need to check if params match because it is irrelevant.
            param_match_percentage = 0.0 
        
        else: 
            # Compute percentage of params in gt action captured by predicted action params
            gt_param_pairs = set()
            for key, value in gt_action.parameters.items():
                # Convert value to string for consistent comparison
                gt_param_pairs.add((key, str(value)))
            
            predicted_param_pairs = set()
            for key, value in predicted_action.parameters.items():
                # Convert value to string for consistent comparison
                predicted_param_pairs.add((key, str(value)))
            
            # Calculate percentage of GT params captured in predicted params
            if len(gt_param_pairs) > 0:
                param_match_percentage = len(gt_param_pairs.intersection(predicted_param_pairs)) / len(gt_param_pairs)
            else:
                param_match_percentage = 1.0  # Perfect if no GT params to match
        
        # Compute action diffs - mismatches in tool name or parameters
        
        # Add tool name mismatch if different
        if not tool_match:
            # Include all parameters for the mismatched tool
            all_params = []
            for key, value in predicted_action.parameters.items():
                all_params.append(f"{key}={value}")
            action_diffs = (len(all_params), (predicted_action.tool_name, all_params))

        else: 
            # Add parameters that are in predicted but not in GT (extra parameters)
            gt_param_keys = set(gt_action.parameters.keys())
            extra_params = []
            for key, value in predicted_action.parameters.items():
                if key not in gt_param_keys:
                    extra_params.append(f"{key}={value}")
            
            # If there are extra parameters, add them as a single diff entry
            if extra_params:
                action_diffs = (len(extra_params), (predicted_action.tool_name, extra_params))
            else:
                # No diffs - perfect match
                action_diffs = (0, (predicted_action.tool_name, []))
        
        # Store metrics for this step
        action_rollout_metrics.append((tool_match, param_match_percentage, action_diffs))
    
    return action_rollout_metrics

def constraint_violation_evaluation(pred_constraint_nums: List[int], pred_invalid_action_idxs: List[int], gt_constraint_nums: List[int], gt_invalid_action_idxs: List[int]) -> Tuple[float, float, float]:
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

    # Create pairs of (constraint_num, invalid_action_idx) and convert to sets for comparison
    # This preserves the pairing relationship while allowing for different ordering
    pred_pairs = set(zip(pred_constraint_nums, pred_invalid_action_idxs))
    gt_pairs = set(zip(gt_constraint_nums, gt_invalid_action_idxs))

    # print(f"Predicted (constraint_num, invalid_action_idx) pairs: {sorted(pred_pairs)}")
    # print(f"Ground truth (constraint_num, invalid_action_idx) pairs: {sorted(gt_pairs)}")

    print("Predicted constraint numbers: ", sorted(pred_constraint_nums))
    print("Ground truth constraint numbers: ", sorted(gt_constraint_nums))
    print("Predicted invalid action indices: ", sorted(pred_invalid_action_idxs))
    print("Ground truth invalid action indices: ", sorted(gt_invalid_action_idxs))

    # Check if the sets of pairs are identical
    return 1 if pred_pairs == gt_pairs else 0

if __name__ == "__main__":
 
    # Setup the world model agent 
    # models = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/o3", "anthropic/claude-sonnet-4", "openai/gpt-5", "anthropic/claude-sonnet-4.5"]

    # Accept command line arguments for the model, perfect_schema, and mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4")
    parser.add_argument("--perfect_schema", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="state_action", choices=["action_only", "state_action", "state_only"])
    parser.add_argument("--trajectory_type", type=str, default="combined", choices=["original", "combined", "masked", "all"])
    args = parser.parse_args()

    perfect_schema = args.perfect_schema
    mode = args.mode
    model = args.model
    
    # agent = WorldModelAgent(model=model)
    # asyncio.run(agent.initialize_mcp_server("full"))

    # # Load the QA dataset 
    # data_folder = Path(__file__).parent / "qa_data"
    # policies = [] 
    
    # # Read all the policies from the data folder > constraint<number> > answer.json 
    # for constraint_num in range(1, 13):
    #     with open(data_folder / f"constraint{constraint_num}" / "answer.json", "r") as f:
    #         policy = json.load(f)
    #         policies.append(policy["policy_name"]) 

    # results = []
    # gt_trajectories = [] 
    # all_gt_answer_idxs = [] 
    # all_gt_constraint_nums = [] 
    # # Load all gt trajectories (original trajectory, combined trajectory, all masked perturbations) and their answer indices and constraint numbers
    # for constraint_num in tqdm(range(1, 13), desc=f"Loading all trajectories"):
        
    #     if args.trajectory_type in ["original", "all"]:
    #         # Load single constraint violation trajectory (original trajectory)
    #         with open(data_folder / f"constraint{constraint_num}" / "trajectory.json", "r") as f:
    #             gt_trajectory = json.load(f) 
    #             gt_trajectories.append(gt_trajectory) 

    #         # Load the answer for the original trajectory
    #         with open(data_folder / f"constraint{constraint_num}" / "answer.json", "r") as f:
    #             gt_answer_idx = json.load(f)["invalid_action_idx"]
    #             gt_constraint_num = constraint_num - 1 
    #             all_gt_answer_idxs.append([gt_answer_idx]) # Evaluation is done on lists of answer indices and constraint numbers so that combined perturbations can be evaluated as well. 
    #             all_gt_constraint_nums.append([gt_constraint_num])

    #     if args.trajectory_type in ["combined", "all"]:
    #         combined_trajectory_folders = []
    #         for folder in sorted(os.listdir(data_folder)): 
    #             if folder.startswith(f"combined_trajectory_{constraint_num}_"):
    #                 combined_trajectory_folders.append(folder)
            
    #         for folder in sorted(combined_trajectory_folders, key=lambda x: int(x.split("_")[-2])):
    #             # Load combined constraint violation trajectories that includes the original trajectory
    #             with open(data_folder / folder / "trajectory.json", "r") as f:
    #                 combined_trajectory = json.load(f) 
    #                 gt_trajectories.append(combined_trajectory) 

    #             # Load the answer for the combined trajectory
    #             with open(data_folder / folder / "answer.json", "r") as f:
    #                 combined_answer = json.load(f)
    #                 all_gt_answer_idxs.append(combined_answer["invalid_action_idxs"]) 
    #                 all_gt_constraint_nums.append(combined_answer["invalid_policy_nums"])

    #     if args.trajectory_type in ["masked", "all"]:
    #         # Load all masked perturbations of the original trajectory that avoid violating any constraint
    #         for file in sorted(os.listdir(data_folder / f"perturbed_trajectory_{constraint_num}")):
    #             if file != "answer.json":
    #                 with open(data_folder / f"perturbed_trajectory_{constraint_num}" / file, "r") as f:
    #                     perturbed_trajectory = json.load(f) 
    #                     gt_trajectories.append(perturbed_trajectory)
    #                     all_gt_answer_idxs.append([-1])
    #                     all_gt_constraint_nums.append([-1])

    # # Predict constraint violation by LLM 
    # total_input_tokens = 0
    # total_cost = 0 
    # total_json_errors = 0 
    # context_window_errors = 0 

    # for trajectory, gt_answer_idxs, gt_constraint_nums in tqdm(zip(gt_trajectories, all_gt_answer_idxs, all_gt_constraint_nums), desc=f"Predicting constraint violation for {model.split('/')[-1]}", total=len(gt_trajectories)):
    #     print(f"Length of trajectory: {len(trajectory)}")
    #     print(f"Number of combined trajectories: {len(gt_constraint_nums)}")
    #     try: 
    #         pred_constraint_violation, usage = asyncio.run(agent.predict_constraint_violation(trajectory, policies, mode=mode, perfect_schema=perfect_schema))
    #     except Exception as e:
    #         context_window_errors += 1
    #         continue

    #     total_input_tokens += usage.prompt_tokens
    #     total_cost += usage.cost
    #     print(f"Total input tokens so far: {total_input_tokens}")
    #     print(f"Total cost so far: {total_cost}")

    #     if pred_constraint_violation is None:
    #         print(f"{model.split('/')[-1]} failed to output structured JSON")
    #         total_json_errors += 1

    #     result = {
    #         "gt_answer_idxs": gt_answer_idxs,
    #         "gt_constraint_nums": gt_constraint_nums,
    #         "llm_constraint_nums": pred_constraint_violation["violated_policy_idxs"] if pred_constraint_violation is not None else [],
    #         "llm_answer_idxs": pred_constraint_violation["invalid_action_idxs"] if pred_constraint_violation is not None else [],
    #         "llm_reason": pred_constraint_violation["reason"] if pred_constraint_violation is not None else "",
    #     }

    #     results.append(result) 

    # # Save results 
    # save_folder_name = f"{"perfect_schema" if perfect_schema else "non_perfect_schema"}_{mode}_{args.trajectory_type}_constraint_evals"
    # save_path = Path(__file__).parent / "llm_evals" / "with_perturbations" /save_folder_name / f"{model.split('/')[-1]}_llm_violation_predictions.json"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(save_path, "w") as f:
    #     json.dump(results, f, indent=2)

    # Read results from the save path
    save_folder_name = f"{"perfect_schema" if perfect_schema else "non_perfect_schema"}_{mode}_{args.trajectory_type}_constraint_evals"
    save_path = Path(__file__).parent / "llm_evals" / "with_perturbations" /save_folder_name / f"{model.split('/')[-1]}_llm_violation_predictions.json"
    with open(save_path, "r") as f:
        results = json.load(f)

    # Evaluate the LLM violation predictions
    results_path = Path(__file__).parent / "llm_evals" / "with_perturbations" /save_folder_name / "results" / f"{model.split('/')[-1]}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    constraint_violation_accuracies = []

    for i, result in tqdm(enumerate(results), desc="Evaluating constraint violation predictions"): 
        print(f"Result {i} --------------------------------")
        pred_constraint_nums = result["llm_constraint_nums"]
        pred_invalid_action_idxs = result["llm_answer_idxs"]
        gt_constraint_nums = result["gt_constraint_nums"]
        gt_invalid_action_idxs = result["gt_answer_idxs"]
        
        constraint_violation_accuracy = constraint_violation_evaluation(pred_constraint_nums, pred_invalid_action_idxs, gt_constraint_nums, gt_invalid_action_idxs)
        constraint_violation_accuracies.append(constraint_violation_accuracy)

    with open(Path(__file__).parent / "qa_data" / f"combined_trajectory_1_3" / "trajectory.json", "r") as f:
        temp_trajectory = json.load(f)
    
    temp_actions = [step["action"] for step in temp_trajectory]
    with open("temp_actions.json", "w") as f:
        json.dump(temp_actions, f, indent=2)


    # with open(results_path, "w") as f:
    #     json.dump({
    #         "accuracy": np.mean(constraint_violation_accuracies), 
    #         "total_input_tokens": total_input_tokens,
    #         "total_cost": total_cost,
    #         "total_json_errors": total_json_errors,
    #         "context_window_errors": context_window_errors
    #     }, f, indent=2)

    print(f"Constraint violation accuracy for {model}: {np.mean(constraint_violation_accuracies)}")
    # print(f"Total input tokens: {total_input_tokens}")
    # print(f"Total cost: {total_cost}")
    # print(f"Total JSON errors: {total_json_errors}")
