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
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Import local modules
try:
    from .world_model_agent import (
        WorldModelAgent, 
        StateDiff, 
        ActionCall, 
        SysAuditRecord, 
        AdditionalInformation, 
        operation
    )
except ImportError:
    # Fallback for direct script execution
    from world_model_agent import (
        WorldModelAgent, 
        StateDiff, 
        ActionCall, 
        SysAuditRecord, 
        AdditionalInformation, 
        operation
    )


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
        
        state_pred_dir = self.base_dir / "state_preds_new" / self.model_name
        
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
        """Evaluate predictions for a specific k value"""
        prediction_files = list(pred_dir.glob(f"*_k{k}.json"))
        
        if not prediction_files:
            return {
                'status': 'no_files',
                'message': f'No prediction files found for k={k}'
            }
        
        trajectories = []
        predictions = []
        
        for pred_file in tqdm(prediction_files, desc=f"Loading k={k}"):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                predictions.append(pred_data)
                
                # Find corresponding trajectory
                traj_file = self.base_dir / "trajectories" / pred_file.name.replace(f'_k{k}.json', '.json')
                if traj_file.exists():
                    with open(traj_file, 'r') as tf:
                        trajectories.append(json.load(tf))
                else:
                    trajectories.append(None)
        
        # Evaluate predictions
        evaluations = []
        for pred, traj in zip(predictions, trajectories):
            if traj is None:
                continue
            eval_result = self._evaluate_single(k, pred, traj)
            evaluations.append(eval_result)
        
        if evaluations:
            avg_accuracy = np.mean([e.get('accuracy', 0) for e in evaluations])
            return {
                'status': 'success',
                'num_files': len(evaluations),
                'average_accuracy': avg_accuracy,
                'details': evaluations
            }
        else:
            return {
                'status': 'no_valid_evaluations',
                'num_files': 0
            }
    
    def _evaluate_single(self, k: int, pred_data: Dict, traj_data: Dict) -> Dict[str, Any]:
        """Evaluate a single prediction against ground truth"""
        # Simplified evaluation - in practice, you'd compare state diffs
        return {
            'accuracy': 0.0,  # Placeholder
            'notes': 'Evaluation implementation needed'
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
        
        action_pred_dir = self.base_dir / "action_predictions_results" / self.model_name
        trajectories_dir = self.base_dir / "trajectories"
        
        if not action_pred_dir.exists():
            print(f"❌ Action prediction directory not found: {action_pred_dir}")
            return {'error': f"Directory not found: {action_pred_dir}", 'success': False}
        
        results = {
            'model': self.model_name,
            'evaluation_type': 'action_prediction',
            'timestamp': datetime.now().isoformat()
        }
        
        # Find matching files
        prediction_files = list(action_pred_dir.glob("*.json"))
        trajectory_files = list(trajectories_dir.glob("*.json"))
        
        evaluations = []
        tool_name_matches = 0
        param_matches = 0
        total_actions = 0
        
        for pred_file in tqdm(prediction_files, desc="Evaluating actions"):
            # Find corresponding trajectory
            traj_file = trajectories_dir / pred_file.name
            
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
                
                # Compare parameters
                gt_params = gt_action.get('parameters', {})
                pred_params = pred_action.get('parameters', {})
                
                for key in gt_params:
                    if key in pred_params and gt_params[key] == pred_params[key]:
                        param_matches += 1
        
        tool_name_accuracy = tool_name_matches / total_actions if total_actions > 0 else 0
        param_accuracy = param_matches / total_actions if total_actions > 0 else 0
        
        results['metrics'] = {
            'tool_name_accuracy': tool_name_accuracy,
            'parameter_accuracy': param_accuracy,
            'total_actions': total_actions,
            'tool_name_matches': tool_name_matches,
            'parameter_matches': param_matches
        }
        
        print(f"✅ Tool name accuracy: {tool_name_accuracy:.3f}")
        print(f"✅ Parameter accuracy: {param_accuracy:.3f}")
        
        return results


class ConstraintEvaluator:
    """Evaluate constraint-based task completion"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent
    
    def evaluate_constraint_predictions(
        self, 
        mode: str = "state_action",
        perfect_schema: bool = True
    ) -> Dict[str, Any]:
        """Evaluate constraint violation predictions"""
        print(f"\n{'='*80}")
        print("CONSTRAINT EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Mode: {mode}")
        print(f"Perfect schema: {perfect_schema}")
        
        # Look for evaluation results
        eval_folder = f"{'perfect_schema' if perfect_schema else 'non_perfect_schema'}_{mode}_constraint_evals"
        results_dir = self.base_dir / "llm_evals" / "with_perturbations" / eval_folder
        
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return {
                'error': f"Directory not found: {results_dir}",
                'success': False
            }
        
        # Load predictions
        pred_file = results_dir / f"{self.model_name.split('/')[-1]}_llm_violation_predictions.json"
        
        if not pred_file.exists():
            print(f"❌ Prediction file not found: {pred_file}")
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
        perfect_schema: bool = True
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
                    perfect_schema=perfect_schema
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
        perfect_schema=args.perfect_schema
    )


if __name__ == "__main__":
    main()

