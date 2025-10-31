import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from wow.environment.agent import StateDiff, SysAuditRecord, AdditionalInformation


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


# State evaluation functions
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


def compute_state_rollout_metrics(pred_state_diffs: List[StateDiff], gt_state_diffs: List[StateDiff]): 
    """
    Returns full state rollout accuracy, partial state rollout accuracy for each step in the action trajectory. 

    Returns: 
        full_state_rollout_accuracy: List[float] 
        partial_state_rollout_accuracy: List[Tuple[float, float]] 
        side_effects: List[Tuple[int, List[SysAuditRecord]]] (unused, kept for compatibility)
    """ 

    full_rollout_acc = []
    part_rollout_acc = []
    side_effects = []

    for pred_state_diff, gt_state_diff in tqdm(zip(pred_state_diffs, gt_state_diffs), total=len(gt_state_diffs), desc="Computing state rollout metrics"):
        full_rollout_acc.append(compute_full_rollout_accuracy(pred_state_diff, gt_state_diff))
        part_rollout_acc.append(compute_partial_rollout_accuracy(pred_state_diff, gt_state_diff))
        # Side effects not used but kept for compatibility
        gt_table_column = set((r.tablename, r.fieldname) for r in gt_state_diff.sysauditrecord)
        side_effs = [r for r in pred_state_diff.sysauditrecord if (r.tablename, r.fieldname) not in gt_table_column]
        side_effects.append((len(side_effs), side_effs))

    return full_rollout_acc, part_rollout_acc, side_effects


class StatePredictionEvaluator:
    """Evaluate state prediction accuracy"""
    
    def __init__(self, model_name: str, base_dir: Optional[Path] = None):
        self.model_name = model_name
        if base_dir is None:
            # Default to project root (parent of src directory)
            self.base_dir = Path(__file__).parent.parent.parent.parent
        else:
            self.base_dir = base_dir
        
    def evaluate_state_predictions(self, k_values: List[int] = [1, 2, 3, 4, 5]) -> Dict[str, Any]:
        """Evaluate state predictions for given k values"""
        
        print(f"\n{'='*80}\nSTATE PREDICTION EVALUATION\n{'='*80}")
        print(f"Model: {self.model_name}\nK values: {k_values}")
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'state_prediction',
            'timestamp': datetime.now().isoformat(),
            'k_evaluations': {}
        }
        
        # Load state predictions from state_preds directory
        state_pred_dir = self.base_dir / "src" / "wow" / "data_files" / "state_preds" / self.model_name
        
        if not state_pred_dir.exists():
            print(f"❌ State prediction directory not found: {state_pred_dir}")
            return {
                'error': f"Directory not found: {state_pred_dir}",
                'success': False
            }
        
        for k in k_values:
            print(f"\n📊 Evaluating k={k}...")
            results['k_evaluations'][f'k={k}'] = self._evaluate_k(state_pred_dir, k)
        
        return results
    
    def _evaluate_k(self, pred_dir: Path, k: int) -> Dict[str, Any]:
        """Evaluate state predictions for a specific k value"""
        # Load all prediction files and filter those with file_k >= requested_k
        all_prediction_files = list(pred_dir.glob("*_k[0-9].json"))
        
        # Filter files where file_k >= requested_k
        prediction_files = []
        for pred_file in all_prediction_files:
            match = re.search(r'_k(\d+)\.json$', pred_file.name)
            if match:
                file_k = int(match.group(1))
                if file_k >= k:
                    prediction_files.append((pred_file, file_k))
        
        if not prediction_files:
            return {
                'status': 'no_files',
                'message': f'No prediction files found with k>={k}'
            }
        
        all_evaluations = []
        
        for pred_file, file_k in tqdm(prediction_files, desc=f"Evaluating k={k}"):
            try:
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
                
                traj_filename = re.sub(r'_k\d+\.json$', '.json', pred_file.name)
                traj_file = self.base_dir / "src" / "wow" / "data_files" / "trajectories" / traj_filename
                
                if not traj_file.exists():
                    print(f"⚠️ Warning: Trajectory file not found: {traj_file}")
                    continue
                
                with open(traj_file, 'r') as tf:
                    traj_data = json.load(tf)
                
                eval_result = self._evaluate_single(pred_data, traj_data, k)
                if eval_result:
                    all_evaluations.append(eval_result)
                    
            except Exception as e:
                print(f"⚠️ Warning: Error evaluating {pred_file}: {e}")
                continue
        
        if all_evaluations:
            total_steps = sum(e.get('num_steps', 0) for e in all_evaluations)
            total_full_matches = sum(e.get('num_full_matches', 0) for e in all_evaluations)
            avg_sysaudit_iou = np.mean([e.get('avg_sysaudit_iou', 0) for e in all_evaluations])
            avg_additional_info_iou = np.mean([e.get('avg_additional_info_iou', 0) for e in all_evaluations])
            
            full_match_rate = total_full_matches / total_steps if total_steps > 0 else 0.0
            
            return {
                'status': 'success',
                'num_files': len(all_evaluations),
                'total_steps': total_steps,
                'full_match_rate': full_match_rate,
                'avg_sysaudit_iou': avg_sysaudit_iou,
                'avg_additional_info_iou': avg_additional_info_iou,
                'details': all_evaluations
            }
        else:
            return {
                'status': 'no_valid_evaluations',
                'num_files': 0
            }
    
    def _dict_to_state_diff(self, state_dict: Dict) -> Optional[StateDiff]:
        """Convert dictionary to StateDiff object"""
        try:
            sysaudit_records = []
            for record_dict in state_dict.get('sysauditrecord', []):
                sysaudit_records.append(SysAuditRecord(**record_dict))
            
            additional_info_dict = state_dict.get('additional_information', {})
            if 'operation_type' in additional_info_dict:
                op_types = additional_info_dict['operation_type']
                if isinstance(op_types, list):
                    from wow.environment.agent import operation
                    additional_info_dict['operation_type'] = [
                        op if isinstance(op, operation) else operation[op] if isinstance(op, str) and op in operation.__members__ else op
                        for op in op_types
                    ]
            
            additional_info = AdditionalInformation(**additional_info_dict)
            
            return StateDiff(
                sysauditrecord=sysaudit_records,
                additional_information=additional_info
            )
        except Exception as e:
            print(f"⚠️ Warning: Error converting state dict to StateDiff: {e}")
            return None
    
    def _extract_gt_state_diffs(self, trajectory: List[Dict], k: int) -> List[StateDiff]:
        """Extract ground truth state diffs from trajectory"""
        gt_state_diffs = []
        
        for step in trajectory[:k]:
            state_diff_dict = None
            
            if "state_diff" in step:
                if isinstance(step["state_diff"], str):
                    try:
                        state_diff_dict = json.loads(step["state_diff"])
                    except json.JSONDecodeError:
                        continue
                else:
                    state_diff_dict = step["state_diff"]
            elif "ground_truth_state" in step:
                if isinstance(step["ground_truth_state"], str):
                    try:
                        state_diff_dict = json.loads(step["ground_truth_state"])
                    except json.JSONDecodeError:
                        continue
                else:
                    state_diff_dict = step["ground_truth_state"]
            elif "audits" in step and step["audits"]:
                state_diff_dict = self._convert_audits_to_state_diff(step["audits"])
            
            if state_diff_dict:
                state_diff = self._dict_to_state_diff(state_diff_dict)
                if state_diff:
                    gt_state_diffs.append(state_diff)
        
        return gt_state_diffs
    
    def _convert_audits_to_state_diff(self, audits: List[Dict]) -> Dict:
        """Convert audit records to state_diff format"""
        from wow.environment.agent import operation
        
        sysaudit_records = []
        tables_modified = set()
        operation_types = set()
        num_modified = 0
        num_deleted = 0
        num_created = 0
        
        for audit in audits:
            sysaudit_records.append({
                'tablename': audit.get('tablename', ''),
                'fieldname': audit.get('fieldname', ''),
                'oldvalue': audit.get('oldvalue', ''),
                'newvalue': audit.get('newvalue', '')
            })
            
            table = audit.get('tablename', '')
            if table:
                tables_modified.add(table)
            
            old_val = audit.get('oldvalue', '')
            new_val = audit.get('newvalue', '')
            
            if old_val == '' and new_val != '':
                operation_types.add('post')
                num_created += 1
            elif old_val != '' and new_val == '':
                operation_types.add('delete')
                num_deleted += 1
            elif old_val != '' and new_val != '':
                operation_types.add('put')
                num_modified += 1
        
        return {
            'sysauditrecord': sysaudit_records,
            'additional_information': {
                'num_audits': len(audits),
                'num_modified_entries': num_modified,
                'num_deleted_entries': num_deleted,
                'num_created_entries': num_created,
                'operation_type': list(operation_types),
                'tables_modified': list(tables_modified)
            }
        }
    
    def _evaluate_single(self, pred_data: Dict, traj_data: List[Dict], k: int) -> Optional[Dict[str, Any]]:
        """Evaluate a single state prediction against ground truth"""
        predicted_states_dict = pred_data.get('predicted_states', [])
        gt_state_diffs = self._extract_gt_state_diffs(traj_data, k)
        
        pred_state_diffs = []
        for state_dict in predicted_states_dict[:k]:
            state_diff = self._dict_to_state_diff(state_dict)
            if state_diff:
                pred_state_diffs.append(state_diff)
        
        min_len = min(len(pred_state_diffs), len(gt_state_diffs))
        if min_len == 0:
            return None
        
        pred_state_diffs = pred_state_diffs[:min_len]
        gt_state_diffs = gt_state_diffs[:min_len]
        
        full_rollout_acc, part_rollout_acc, _ = compute_state_rollout_metrics(
            pred_state_diffs, gt_state_diffs
        )
        
        num_steps = len(full_rollout_acc)
        num_full_matches = sum(1 for acc in full_rollout_acc if acc == 1.0)
        
        sysaudit_ious = [iou[0] for iou in part_rollout_acc]
        additional_info_ious = [iou[1] for iou in part_rollout_acc]
        avg_sysaudit_iou = np.mean(sysaudit_ious) if sysaudit_ious else 0.0
        avg_additional_info_iou = np.mean(additional_info_ious) if additional_info_ious else 0.0
        
        return {
            'num_steps': num_steps,
            'num_full_matches': num_full_matches,
            'full_match_rate': num_full_matches / num_steps if num_steps > 0 else 0.0,
            'avg_sysaudit_iou': avg_sysaudit_iou,
            'avg_additional_info_iou': avg_additional_info_iou,
            'full_rollout_acc': full_rollout_acc,
            'partial_rollout_acc': [(float(iou[0]), float(iou[1])) for iou in part_rollout_acc]
        }


class ActionPredictionEvaluator:
    """Evaluate action prediction accuracy"""
    
    def __init__(self, model_name: str, base_dir: Optional[Path] = None):
        self.model_name = model_name
        if base_dir is None:
            # Default to project root (parent of src directory)
            self.base_dir = Path(__file__).parent.parent.parent.parent
        else:
            self.base_dir = base_dir
    
    def evaluate_action_predictions(self) -> Dict[str, Any]:
        """Evaluate action prediction accuracy"""
        print(f"\n{'='*80}")
        print("ACTION PREDICTION EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        
        action_pred_dir = self.base_dir / "src" / "wow" / "data_files" / "action_preds" / self.model_name
        trajectories_dir = self.base_dir / "src" / "wow" / "data_files" / "trajectories"
        
        if not action_pred_dir.exists():
            print(f"❌ Action prediction directory not found: {action_pred_dir}")
            return {'error': f"Directory not found: {action_pred_dir}", 'success': False}
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'action_prediction',
            'timestamp': datetime.now().isoformat()
        }
        
        prediction_files = [f for f in action_pred_dir.glob("*.json") if not f.name.endswith("_summary.json")]
        
        tool_name_matches = 0
        perfect_matches = 0
        total_actions = 0
        
        def normalize_param_value(val):
            """Normalize parameter value for comparison"""
            if val is None:
                return None
            if isinstance(val, bool):
                return bool(val)
            if isinstance(val, (int, float)):
                return val
            return str(val).strip()
        
        for pred_file in tqdm(prediction_files, desc="Evaluating actions"):
            traj_filename = pred_file.name.replace("_action_predictions.json", ".json")
            traj_file = trajectories_dir / traj_filename
            
            if not traj_file.exists():
                continue
            
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            
            gt_actions = [step.get('action', {}) for step in traj_data if 'action' in step]
            pred_actions = pred_data.get('predictions', [])
            
            for gt_action, pred_action in zip(gt_actions[:len(pred_actions)], pred_actions):
                total_actions += 1
                if gt_action.get('tool_name') == pred_action.get('tool_name'):
                    tool_name_matches += 1
                    gt_params = gt_action.get('parameters', {})
                    pred_params = pred_action.get('parameters', {})
                    
                    params_match = True
                    for key, gt_value in gt_params.items():
                        pred_value = pred_params.get(key)
                        if normalize_param_value(gt_value) != normalize_param_value(pred_value):
                            params_match = False
                            break
                    perfect_matches += params_match
        
        tool_name_accuracy = tool_name_matches / total_actions if total_actions > 0 else 0
        perfect_match_accuracy = perfect_matches / total_actions if total_actions > 0 else 0
        
        results['metrics'] = {
            'tool_name_accuracy': tool_name_accuracy,
            'perfect_match_accuracy': perfect_match_accuracy,
            'total_actions': total_actions,
            'tool_name_matches': tool_name_matches,
            'perfect_matches': perfect_matches
        }
        
        print(f"✅ Tool name accuracy: {tool_name_accuracy:.3f}")
        print(f"✅ Perfect match accuracy (tool name + all parameters): {perfect_match_accuracy:.3f}")
        
        return results
