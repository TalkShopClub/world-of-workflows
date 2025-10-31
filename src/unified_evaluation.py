#!/usr/bin/env python3
"""
Unified Evaluation Script

Evaluates three aspects of world model performance:
1. State Prediction - Accuracy of predicting state changes from actions
2. Action Prediction - Accuracy of predicting actions from state changes (inverse prediction)
3. Constraint-Based Task Completion - Accuracy of identifying constraint violations

Usage:
    python unified_evaluation.py --model gpt-4o --evaluation-type all
    python unified_evaluation.py --model claude-sonnet-4 --evaluation-type state --k-values 1 2 3
    python unified_evaluation.py --model o3 --evaluation-type constraint --mode action_only
"""

import json
import os
import argparse
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import numpy as np

from wow.environment.agent import StateDiff, SysAuditRecord, AdditionalInformation
from eval_pipeline import compute_state_rollout_metrics

class StatePredictionEvaluator:
    """Evaluate state prediction accuracy"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent
        
    def evaluate_state_predictions(self, k_values: List[int] = None) -> Dict[str, Any]:
        """Evaluate state predictions for given k values"""
        if k_values is None:
            k_values = [1, 2, 3, 4, 5]
        
        print(f"\n{'='*80}")
        print("STATE PREDICTION EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"K values: {k_values}")
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'state_prediction',
            'timestamp': datetime.now().isoformat(),
            'k_evaluations': {}
        }
        
        # Load state predictions from state_preds directory
        state_pred_dir = self.base_dir / "wow" / "data_files" / "state_preds" / self.model_name
        
        if not state_pred_dir.exists():
            print(f"❌ State prediction directory not found: {state_pred_dir}")
            return {
                'error': f"Directory not found: {state_pred_dir}",
                'success': False
            }
        
        for k in k_values:
            print(f"\n📊 Evaluating k={k}...")
            k_results = self._evaluate_k(state_pred_dir, k)
            results['k_evaluations'][f'k={k}'] = k_results
        
        return results
    
    def _evaluate_k(self, pred_dir: Path, k: int) -> Dict[str, Any]:
        """Evaluate state predictions for a specific k value"""
        # Load all prediction files and filter those with file_k >= requested_k
        # Files are named like: approvechangerequest_k4.json
        # A file with k=5 can be used for k=1,2,3,4,5 evaluations
        all_prediction_files = list(pred_dir.glob("*_k*.json"))
        
        # Exclude summary files
        all_prediction_files = [f for f in all_prediction_files if not f.name.endswith("_summary.json")]
        
        # Filter files where file_k >= requested_k
        prediction_files = []
        for pred_file in all_prediction_files:
            # Extract k value from filename (e.g., approvechangerequest_k4.json -> 4)
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
                
                # Extract trajectory name from filename (e.g., approvechangerequest_k4.json -> approvechangerequest)
                # Remove _k{digits}.json pattern
                traj_filename = re.sub(r'_k\d+\.json$', '.json', pred_file.name)
                traj_file = self.base_dir / "wow" / "data_files" / "trajectories" / traj_filename
                
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
                import traceback
                traceback.print_exc()
                continue
        
        if all_evaluations:
            # Aggregate metrics across all files
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
            # Convert sysaudit records
            sysaudit_records = []
            for record_dict in state_dict.get('sysauditrecord', []):
                sysaudit_records.append(SysAuditRecord(**record_dict))
            
            # Convert additional information
            additional_info_dict = state_dict.get('additional_information', {})
            # Handle operation_type enum values
            if 'operation_type' in additional_info_dict:
                op_types = additional_info_dict['operation_type']
                if isinstance(op_types, list):
                    # Convert string values to operation enum if needed
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
            
            # Try different formats for state information
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
                # Convert audits to state_diff format
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
            # Create sysaudit record
            sysaudit_records.append({
                'tablename': audit.get('tablename', ''),
                'fieldname': audit.get('fieldname', ''),
                'oldvalue': audit.get('oldvalue', ''),
                'newvalue': audit.get('newvalue', '')
            })
            
            table = audit.get('tablename', '')
            if table:
                tables_modified.add(table)
            
            # Infer operation type from old/new values
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
        """
        Evaluate a single state prediction against ground truth.
        
        Args:
            pred_data: Prediction data from state_preds (contains 'predicted_states' list)
            traj_data: Ground truth trajectory data (list of steps)
            k: k value (number of steps to evaluate)
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract predicted states
        predicted_states_dict = pred_data.get('predicted_states', [])
        
        # Extract ground truth state diffs from trajectory
        gt_state_diffs = self._extract_gt_state_diffs(traj_data, k)
        
        # Convert predicted states to StateDiff objects
        pred_state_diffs = []
        for state_dict in predicted_states_dict[:k]:
            state_diff = self._dict_to_state_diff(state_dict)
            if state_diff:
                pred_state_diffs.append(state_diff)
        
        # Match lengths - pad or truncate as needed
        min_len = min(len(pred_state_diffs), len(gt_state_diffs))
        if min_len == 0:
            return None
        
        pred_state_diffs = pred_state_diffs[:min_len]
        gt_state_diffs = gt_state_diffs[:min_len]
        
        # Use compute_state_rollout_metrics from eval_pipeline
        full_rollout_acc, part_rollout_acc, _ = compute_state_rollout_metrics(
            pred_state_diffs, gt_state_diffs
        )
        
        # Calculate aggregate metrics
        num_steps = len(full_rollout_acc)
        num_full_matches = sum(1 for acc in full_rollout_acc if acc == 1.0)
        
        # Calculate average IoU for partial matches
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
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent
    
    def evaluate_action_predictions(self) -> Dict[str, Any]:
        """Evaluate action prediction accuracy"""
        print(f"\n{'='*80}")
        print("ACTION PREDICTION EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        
        action_pred_dir = self.base_dir / "wow" / "data_files" / "action_preds" / self.model_name
        trajectories_dir = self.base_dir / "wow" / "data_files" / "trajectories"
        
        if not action_pred_dir.exists():
            print(f"❌ Action prediction directory not found: {action_pred_dir}")
            return {'error': f"Directory not found: {action_pred_dir}", 'success': False}
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'action_prediction',
            'timestamp': datetime.now().isoformat()
        }
        
        # Find matching files (exclude summary files)
        prediction_files = [f for f in action_pred_dir.glob("*.json") if not f.name.endswith("_summary.json")]
        trajectory_files = list(trajectories_dir.glob("*.json"))
        
        evaluations = []
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
            # Find corresponding trajectory (strip _action_predictions suffix)
            traj_filename = pred_file.name.replace("_action_predictions.json", ".json")
            traj_file = trajectories_dir / traj_filename
            
            if not traj_file.exists():
                continue
            
            # Load predictions and ground truth
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            
            # Extract actions
            gt_actions = [step.get('action', {}) for step in traj_data if 'action' in step]
            pred_actions = pred_data.get('predictions', [])
            
            # Compare actions
            for gt_action, pred_action in zip(gt_actions[:len(pred_actions)], pred_actions):
                total_actions += 1
                if gt_action.get('tool_name') == pred_action.get('tool_name'):
                    tool_name_matches += 1
                    gt_params = gt_action.get('parameters', {})
                    pred_params = pred_action.get('parameters', {})
                    
                    # Check if all GT parameters match predicted parameters
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


class ConstraintEvaluator:
    """Evaluate constraint-based task completion"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent
    
    def evaluate_constraint_predictions(
        self, 
        mode: str = "state_action",
        perfect_schema: bool = True,
        trajectory_type: str = "combined"
    ) -> Dict[str, Any]:
        """Evaluate constraint violation predictions"""

        assert mode in ["action_only", "state_action", "state_only"], "Invalid mode. Specify one of the following: action_only, state_action, state_only"
        assert trajectory_type in ["original", "combined", "masked", "all"], "Invalid trajectory type. Specify one of the following: original, combined, masked, all"

        print(f"\n{'='*80}")
        print(f"Evaluating constraint predictions for model: {self.model_name}")
        
        # Look for evaluation results
        eval_folder = f"{'perfect_schema' if perfect_schema else 'non_perfect_schema'}_{mode}_{trajectory_type}_constraint_evals"
        results_dir = self.base_dir / "llm_evals" / "with_perturbations" / eval_folder
        
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return {
                'error': f"Directory not found: {results_dir}",
                'success': False
            }
        
        # Load predictions
        pred_file = results_dir / f"{self.model_name.split('/')[-1]}_llm_violation_predictions.json"
        
        if not pred_file.exists():
            print(f"❌ Prediction file for {self.model_name.split('/')[-1]} not found: {pred_file}")
            return {
                'error': f"File not found: {pred_file}",
                'success': False
            }
        
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        
        # Evaluate predictions
        accuracies = []
        for result in pred_data:
            gt_constraint_nums = result.get('gt_constraint_nums', [])
            gt_invalid_action_idxs = result.get('gt_answer_idxs', [])
            pred_constraint_nums = result.get('llm_constraint_nums', [])
            pred_invalid_action_idxs = result.get('llm_answer_idxs', [])
            
            # Create pairs for comparison
            gt_pairs = set(zip(gt_constraint_nums, gt_invalid_action_idxs))
            pred_pairs = set(zip(pred_constraint_nums, pred_invalid_action_idxs))
            
            accuracy = 1.0 if gt_pairs == pred_pairs else 0.0
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'constraint_prediction',
            'mode': mode,
            'perfect_schema': perfect_schema,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'average_accuracy': avg_accuracy,
                'total_cases': len(accuracies),
                'correct_cases': sum(accuracies)
            }
        }
        
        print(f"✅ Average accuracy: {avg_accuracy:.3f}")
        print(f"✅ Total cases: {len(accuracies)}")
        
        return results

class UnifiedEvaluator:
    """Main evaluator that coordinates all three evaluation types"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.state_evaluator = StatePredictionEvaluator(model_name)
        self.action_evaluator = ActionPredictionEvaluator(model_name)
        self.constraint_evaluator = ConstraintEvaluator(model_name)
    
    def evaluate_all(
        self,
        evaluation_type: str = "all",
        k_values: List[int] = None,
        constraint_mode: str = "state_action",
        perfect_schema: bool = True,
        trajectory_type: str = "combined"
    ) -> Dict[str, Any]:
        """
        Run unified evaluation for all or specific evaluation types
        
        Args:
            evaluation_type: 'all', 'state', 'action', or 'constraint'
            k_values: List of k values for state evaluation
            constraint_mode: Mode for constraint evaluation
            perfect_schema: Whether to use perfect schema for constraints
        """
        print(f"\n{'#'*80}")
        print("UNIFIED WORLD MODEL EVALUATION")
        print(f"{'#'*80}")
        print(f"Model: {self.model_name}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        results = {
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'evaluations': {}
        }
        
        # State prediction evaluation
        if evaluation_type in ['all', 'state']:
            try:
                state_results = self.state_evaluator.evaluate_state_predictions(k_values)
                results['evaluations']['state_prediction'] = state_results
            except Exception as e:
                print(f"❌ State evaluation failed: {e}")
                results['evaluations']['state_prediction'] = {'error': str(e)}
        
        # Action prediction evaluation
        if evaluation_type in ['all', 'action']:
            try:
                action_results = self.action_evaluator.evaluate_action_predictions()
                results['evaluations']['action_prediction'] = action_results
            except Exception as e:
                print(f"❌ Action evaluation failed: {e}")
                results['evaluations']['action_prediction'] = {'error': str(e)}
        
        # Constraint evaluation
        if evaluation_type in ['all', 'constraint']:
            try:
                constraint_results = self.constraint_evaluator.evaluate_constraint_predictions(
                    mode=constraint_mode,
                    perfect_schema=perfect_schema,
                    trajectory_type=trajectory_type
                )
                results['evaluations']['constraint_prediction'] = constraint_results
            except Exception as e:
                print(f"❌ Constraint evaluation failed: {e}")
                results['evaluations']['constraint_prediction'] = {'error': str(e)}
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        output_dir = Path(__file__).parent / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_evaluation_{self.model_name.replace('/', '_')}_{timestamp}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        for eval_type, eval_results in results['evaluations'].items():
            print(f"\n📊 {eval_type.upper().replace('_', ' ')}:")
            
            if 'error' in eval_results:
                print(f"  ❌ Error: {eval_results['error']}")
            elif 'metrics' in eval_results:
                metrics = eval_results['metrics']
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            elif 'k_evaluations' in eval_results:
                for k_key, k_data in eval_results['k_evaluations'].items():
                    if k_data.get('status') == 'success':
                        print(f"  {k_key}:")
                        print(f"    Full Match Rate: {k_data.get('full_match_rate', 0):.3f}")
                        print(f"    Avg SysAudit IoU: {k_data.get('avg_sysaudit_iou', 0):.3f}")
                        print(f"    Avg Additional Info IoU: {k_data.get('avg_additional_info_iou', 0):.3f}")
                        print(f"    Total Steps: {k_data.get('total_steps', 0)}")
                        print(f"    Files Evaluated: {k_data.get('num_files', 0)}")
                    else:
                        print(f"  {k_key}: {k_data.get('status', 'Unknown')}")
            elif 'average_accuracy' in eval_results:
                print(f"  Average accuracy: {eval_results['average_accuracy']:.3f}")
            else:
                print(f"  Status: {eval_results.get('status', 'Unknown')}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for world model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all aspects
  python unified_evaluation.py --model gpt-4o --evaluation-type all
  
  # Evaluate only state predictions
  python unified_evaluation.py --model claude-sonnet-4 --evaluation-type state --k-values 1 2 3
  
  # Evaluate only action predictions
  python unified_evaluation.py --model o3 --evaluation-type action
  
  # Evaluate only constraint predictions
  python unified_evaluation.py --model gpt-5 --evaluation-type constraint --mode action_only
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name to evaluate (e.g., gpt-4o, claude-sonnet-4)'
    )
    
    parser.add_argument(
        '--evaluation-type',
        type=str,
        choices=['all', 'state', 'action', 'constraint'],
        default='all',
        help='Type of evaluation to run (default: all)'
    )
    
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5],
        help='K values for state evaluation (default: 1 2 3 4 5)'
    )
    
    parser.add_argument(
        '--constraint-mode',
        type=str,
        choices=['action_only', 'state_action', 'state_only'],
        default='state_action',
        help='Mode for constraint evaluation (default: state_action)'
    )

    parser.add_argument(
        '--trajectory-type',
        type=str,
        choices=['original', 'combined', 'masked', 'all'],
        default='combined',
        help='Trajectory type for constraint evaluation (default: combined)'
    )
    
    parser.add_argument(
        '--perfect-schema',
        action='store_true',
        default=True,
        help='Use perfect schema for constraint evaluation (default: True)'
    )
    
    parser.add_argument(
        '--no-perfect-schema',
        dest='perfect_schema',
        action='store_false',
        help='Disable perfect schema for constraint evaluation'
    )
    
    args = parser.parse_args()
    
    evaluator = UnifiedEvaluator(args.model)
    
    evaluator.evaluate_all(
        evaluation_type=args.evaluation_type,
        k_values=args.k_values,
        constraint_mode=args.constraint_mode,
        perfect_schema=args.perfect_schema,
        trajectory_type=args.trajectory_type
    )


if __name__ == "__main__":
    main()

