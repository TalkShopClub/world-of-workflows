#!/usr/bin/env python3
"""
Simplified state prediction evaluation for k=1 to 5.
Generates metrics for k=1,2,3,4,5 all from the predicted_states of k=5,
since predicted states are iteratively generated and contain all intermediary audits at each step.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
import numpy as np
from tqdm import tqdm

class SimpleStateEvaluator:
    """Simplified state prediction evaluator."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
    
    def load_trajectory_ground_truth(self, trajectory_file: str, k: int) -> List[Dict[str, Any]]:
        """Load ground truth state diffs from trajectory file for first k actions."""
        print(f"📂 Loading ground truth from: {trajectory_file}")
        
        try:
            with open(trajectory_file, 'r') as f:
                trajectory = json.load(f)
            
            if not isinstance(trajectory, list):
                raise ValueError("Trajectory file must contain a list of actions")
            
            # Extract state diffs for first k actions
            state_diffs = []
            for i, step in enumerate(trajectory[:k]):
                if 'audits' in step and step['audits']:
                    # Convert audits to simple format
                    sysaudit_records = []
                    for audit in step['audits']:
                        record = {
                            'tablename': audit['tablename'],
                            'fieldname': audit['fieldname'],
                            'oldvalue': audit['oldvalue'],
                            'newvalue': audit['newvalue']
                        }
                        sysaudit_records.append(record)
                    
                    # Create additional information
                    modified_entries = len([r for r in sysaudit_records if r['oldvalue'] != ""])
                    created_entries = len([r for r in sysaudit_records if r['oldvalue'] == ""])
                    tables_modified = list(set(r['tablename'] for r in sysaudit_records))
                    
                    additional_info = {
                        'num_audits': len(sysaudit_records),
                        'num_modified_entries': modified_entries,
                        'num_deleted_entries': 0,  # Not tracked in current format
                        'num_created_entries': created_entries,
                        'operation_type': ['post' if r['oldvalue'] == "" else 'put' for r in sysaudit_records],
                        'tables_modified': tables_modified
                    }
                    
                    state_diff = {
                        'sysauditrecord': sysaudit_records,
                        'additional_information': additional_info
                    }
                    state_diffs.append(state_diff)
                else:
                    # Create empty state diff if no audits
                    state_diff = {
                        'sysauditrecord': [],
                        'additional_information': {
                            'num_audits': 0,
                            'num_modified_entries': 0,
                            'num_deleted_entries': 0,
                            'num_created_entries': 0,
                            'operation_type': [],
                            'tables_modified': []
                        }
                    }
                    state_diffs.append(state_diff)
            
            print(f"✅ Loaded {len(state_diffs)} ground truth state diffs")
            return state_diffs
            
        except Exception as e:
            print(f"❌ Error loading trajectory: {e}")
            return []
    
    def load_predicted_states(self, prediction_file: str, k: int) -> List[Dict[str, Any]]:
        """Load predicted state diffs from k=5 prediction file for specific k value."""
        print(f"📂 Loading predictions from: {prediction_file} (extracting first {k} states)")
        
        try:
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            predicted_states = []
            # Extract only the first k states from the k=5 prediction
            # Each predicted state already contains audits for that specific action only
            for state_data in prediction_data['predicted_states'][:k]:
                # Convert to simple format
                state_diff = {
                    'sysauditrecord': state_data.get('sysauditrecord', []),
                    'additional_information': state_data.get('additional_information', {})
                }
                predicted_states.append(state_diff)
            
            print(f"✅ Loaded {len(predicted_states)} predicted state diffs for k={k}")
            return predicted_states
            
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return []
    
    def compute_full_rollout_accuracy(self, pred_state_diff: Dict[str, Any], gt_state_diff: Dict[str, Any]) -> float:
        """Compute full state rollout accuracy for a single step."""
        # Check if all GT sysaudit records are in predicted (order doesn't matter)
        gt_sysaudit_set = set()
        for record in gt_state_diff['sysauditrecord']:
            gt_sysaudit_set.add((record['tablename'], record['fieldname'], record['oldvalue'], record['newvalue']))
        
        predicted_sysaudit_set = set()
        for record in pred_state_diff['sysauditrecord']:
            predicted_sysaudit_set.add((record['tablename'], record['fieldname'], record['oldvalue'], record['newvalue']))
        
        # Check if all GT sysaudit records are captured in predicted
        sysaudit_match = gt_sysaudit_set.issubset(predicted_sysaudit_set)
        
        # Check additional information equality
        gt_additional = gt_state_diff['additional_information']
        pred_additional = pred_state_diff['additional_information']
        
        additional_info_match = (
            gt_additional.get('num_audits', 0) == pred_additional.get('num_audits', 0) and
            gt_additional.get('num_modified_entries', 0) == pred_additional.get('num_modified_entries', 0) and
            gt_additional.get('num_deleted_entries', 0) == pred_additional.get('num_deleted_entries', 0) and
            gt_additional.get('num_created_entries', 0) == pred_additional.get('num_created_entries', 0) and
            set(gt_additional.get('operation_type', [])) == set(pred_additional.get('operation_type', [])) and
            set(gt_additional.get('tables_modified', [])) == set(pred_additional.get('tables_modified', []))
        )
        
        # Full accuracy is 1.0 if both sysaudit and additional info match, 0.0 otherwise
        return 1.0 if (sysaudit_match and additional_info_match) else 0.0
    
    def compute_partial_rollout_accuracy(self, pred_state_diff: Dict[str, Any], gt_state_diff: Dict[str, Any]) -> Tuple[float, float]:
        """Compute partial state rollout accuracy for a single step."""
        # Get sysaudit sets for comparison
        gt_sysaudit_set = set()
        for record in gt_state_diff['sysauditrecord']:
            gt_sysaudit_set.add((record['tablename'], record['fieldname'], record['oldvalue'], record['newvalue']))
        
        predicted_sysaudit_set = set()
        for record in pred_state_diff['sysauditrecord']:
            predicted_sysaudit_set.add((record['tablename'], record['fieldname'], record['oldvalue'], record['newvalue']))
        
        # Calculate percentage of GT sysaudit records that are in predicted
        if len(gt_sysaudit_set) > 0:
            sysaudit_partial_accuracy = len(gt_sysaudit_set.intersection(predicted_sysaudit_set)) / len(gt_sysaudit_set)
        else:
            sysaudit_partial_accuracy = 1.0  # Perfect if no GT records to match
        
        # For additional information, count how many fields match
        gt_additional = gt_state_diff['additional_information']
        pred_additional = pred_state_diff['additional_information']
        
        additional_info_matches = [
            gt_additional.get('num_audits', 0) == pred_additional.get('num_audits', 0),
            gt_additional.get('num_modified_entries', 0) == pred_additional.get('num_modified_entries', 0),
            gt_additional.get('num_deleted_entries', 0) == pred_additional.get('num_deleted_entries', 0),
            gt_additional.get('num_created_entries', 0) == pred_additional.get('num_created_entries', 0),
            set(gt_additional.get('operation_type', [])) == set(pred_additional.get('operation_type', [])),
            set(gt_additional.get('tables_modified', [])) == set(pred_additional.get('tables_modified', []))
        ]
        
        return sysaudit_partial_accuracy, sum(additional_info_matches) / len(additional_info_matches)
    
    def compute_side_effects(self, pred_state_diff: Dict[str, Any], gt_state_diff: Dict[str, Any]) -> Tuple[int, List[str]]:
        """Compute side effects for a single step."""
        # Find table-column pairs in predicted but not in GT
        gt_table_column = set((r['tablename'], r['fieldname']) for r in gt_state_diff['sysauditrecord'])
        side_effs = [f"{r['tablename']}.{r['fieldname']}" for r in pred_state_diff['sysauditrecord'] 
                    if (r['tablename'], r['fieldname']) not in gt_table_column]
        return len(side_effs), side_effs
    
    def evaluate_single_file(self, prediction_file: str, trajectory_file: str, k: int) -> Dict[str, Any]:
        """Evaluate a single prediction file against its corresponding trajectory."""
        print(f"\n🎯 Evaluating: {os.path.basename(prediction_file)}")
        print("-" * 60)
        
        # Load ground truth and predictions
        gt_states = self.load_trajectory_ground_truth(trajectory_file, k)
        pred_states = self.load_predicted_states(prediction_file, k)
        
        if not gt_states or not pred_states:
            print("❌ Failed to load ground truth or predictions")
            return {
                'file': os.path.basename(prediction_file),
                'status': 'failed',
                'error': 'Failed to load data'
            }
        
        if len(gt_states) != len(pred_states):
            print(f"⚠️ Warning: Mismatch in number of states (GT: {len(gt_states)}, Pred: {len(pred_states)})")
            # Truncate to minimum length
            min_len = min(len(gt_states), len(pred_states))
            gt_states = gt_states[:min_len]
            pred_states = pred_states[:min_len]
        
        try:
            # Run evaluation
            full_accuracy = []
            partial_accuracy = []
            side_effects = []
            
            for pred_state_diff, gt_state_diff in zip(pred_states, gt_states):
                full_acc = self.compute_full_rollout_accuracy(pred_state_diff, gt_state_diff)
                partial_acc = self.compute_partial_rollout_accuracy(pred_state_diff, gt_state_diff)
                side_eff = self.compute_side_effects(pred_state_diff, gt_state_diff)
                
                full_accuracy.append(full_acc)
                partial_accuracy.append(partial_acc)
                side_effects.append(side_eff)
            
            # Calculate summary statistics
            avg_full_accuracy = np.mean(full_accuracy)
            avg_sysaudit_accuracy = np.mean([acc[0] for acc in partial_accuracy])
            avg_additional_accuracy = np.mean([acc[1] for acc in partial_accuracy])
            total_side_effects = sum([se[0] for se in side_effects])
            avg_side_effects = np.mean([se[0] for se in side_effects])
            
            result = {
                'file': os.path.basename(prediction_file),
                'status': 'success',
                'k': k,
                'num_steps': len(gt_states),
                'metrics': {
                    'full_accuracy': avg_full_accuracy,
                    'sysaudit_accuracy': avg_sysaudit_accuracy,
                    'additional_info_accuracy': avg_additional_accuracy,
                    'total_side_effects': total_side_effects,
                    'avg_side_effects_per_step': avg_side_effects
                },
                'step_by_step': {
                    'full_accuracy': full_accuracy,
                    'partial_accuracy': partial_accuracy,
                    'side_effects': [(se[0], se[1]) for se in side_effects]
                }
            }
            
            print(f"✅ Evaluation complete:")
            print(f"   Full Accuracy: {avg_full_accuracy:.3f}")
            print(f"   SysAudit Accuracy: {avg_sysaudit_accuracy:.3f}")
            print(f"   Additional Info Accuracy: {avg_additional_accuracy:.3f}")
            print(f"   Total Side Effects: {total_side_effects}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return {
                'file': os.path.basename(prediction_file),
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_k_value(self, k: int, prediction_dir: Path, trajectory_dir: Path) -> Dict[str, Any]:
        """Evaluate all prediction files for a specific k value using k=5 prediction files."""
        print(f"\n🚀 Evaluating k={k} (using k=5 prediction files)")
        print("=" * 60)
        
        # Look for k=5 prediction files
        k5_dir = prediction_dir / "k=5"
        if not k5_dir.exists():
            print(f"❌ No k=5 predictions found")
            return {'k': k, 'status': 'no_data'}
        
        # Get all k=5 prediction files
        prediction_files = list(k5_dir.glob("*_k5_state_predictions.json"))
        print(f"Found {len(prediction_files)} k=5 prediction files")
        
        if not prediction_files:
            print(f"❌ No k=5 prediction files found")
            return {'k': k, 'status': 'no_files'}
        
        results = []
        successful = 0
        failed = 0
        
        for prediction_file in tqdm(prediction_files, desc=f"Evaluating k={k} from k=5 files"):
            # Extract trajectory name from prediction file
            trajectory_name = prediction_file.stem.replace("_k5_state_predictions", "")
            trajectory_file = trajectory_dir / f"{trajectory_name}.json"
            
            if not trajectory_file.exists():
                print(f"⚠️ Trajectory file not found: {trajectory_file}")
                failed += 1
                continue
            
            result = self.evaluate_single_file(str(prediction_file), str(trajectory_file), k)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            aggregate_metrics = {
                'full_accuracy': np.mean([r['metrics']['full_accuracy'] for r in successful_results]),
                'sysaudit_accuracy': np.mean([r['metrics']['sysaudit_accuracy'] for r in successful_results]),
                'additional_info_accuracy': np.mean([r['metrics']['additional_info_accuracy'] for r in successful_results]),
                'total_side_effects': np.sum([r['metrics']['total_side_effects'] for r in successful_results]),
                'avg_side_effects_per_file': np.mean([r['metrics']['avg_side_effects_per_step'] for r in successful_results])
            }
        else:
            aggregate_metrics = {}
        
        return {
            'k': k,
            'total_files': len(prediction_files),
            'successful': successful,
            'failed': failed,
            'aggregate_metrics': aggregate_metrics,
            'file_results': results
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file."""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            results_converted = convert_numpy_types(results)
            
            with open(output_file, 'w') as f:
                json.dump(results_converted, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate state predictions for k=1 to 5 using k=5 prediction files")
    parser.add_argument("model", help="Model name (e.g., 'gemini-2.5-pro', 'google_gemini-2.5-pro')")
    parser.add_argument("--k-values", nargs='+', type=int, default=[1, 2, 3, 4, 5], 
                       help="K values to evaluate (default: 1 2 3 4 5)")
    parser.add_argument("--trajectory-dir", default="trajectories", 
                       help="Directory containing trajectory files (default: trajectories)")
    parser.add_argument("--output-dir", default="evaluation_results", 
                       help="Output directory for results (default: evaluation_results)")
    
    args = parser.parse_args()
    
    print("🎯 Simple State Prediction Evaluator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"K values: {args.k_values}")
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Output directory: {args.output_dir}")
    print("📝 Note: Using k=5 prediction files for all evaluations")
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    prediction_dir = script_dir / "state_prediction_results" / args.model
    trajectory_dir = Path(args.trajectory_dir) if os.path.isabs(args.trajectory_dir) else script_dir / args.trajectory_dir
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not prediction_dir.exists():
        print(f"❌ Prediction directory not found: {prediction_dir}")
        return
    
    if not trajectory_dir.exists():
        print(f"❌ Trajectory directory not found: {trajectory_dir}")
        return
    
    # Check if k=5 directory exists
    k5_dir = prediction_dir / "k=5"
    if not k5_dir.exists():
        print(f"❌ k=5 prediction directory not found: {k5_dir}")
        print("   This evaluator requires k=5 prediction files to extract states for k=1,2,3,4,5")
        return
    
    # Initialize evaluator
    evaluator = SimpleStateEvaluator(args.model)
    
    # Evaluate each k value
    all_results = {
        'model': args.model,
        'k_values': args.k_values,
        'timestamp': datetime.now().isoformat(),
        'evaluations': {}
    }
    
    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"EVALUATING K={k}")
        print(f"{'='*60}")
        
        k_results = evaluator.evaluate_k_value(k, prediction_dir, trajectory_dir)
        all_results['evaluations'][f'k={k}'] = k_results
        
        if k_results.get('status') != 'no_data' and k_results.get('status') != 'no_files':
            metrics = k_results.get('aggregate_metrics', {})
            print(f"\n📊 K={k} Summary:")
            print(f"   Files processed: {k_results.get('successful', 0)}/{k_results.get('total_files', 0)}")
            print(f"   Full Accuracy: {metrics.get('full_accuracy', 0):.3f}")
            print(f"   SysAudit Accuracy: {metrics.get('sysaudit_accuracy', 0):.3f}")
            print(f"   Additional Info Accuracy: {metrics.get('additional_info_accuracy', 0):.3f}")
            print(f"   Total Side Effects: {metrics.get('total_side_effects', 0)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"state_prediction_evaluation_{args.model}_{timestamp}.json"
    evaluator.save_results(all_results, str(output_file))
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for k in args.k_values:
        k_key = f'k={k}'
        if k_key in all_results['evaluations']:
            k_results = all_results['evaluations'][k_key]
            if k_results.get('status') == 'success' and 'aggregate_metrics' in k_results:
                metrics = k_results['aggregate_metrics']
                print(f"K={k}: Full Acc={metrics.get('full_accuracy', 0):.3f}, "
                      f"SysAudit Acc={metrics.get('sysaudit_accuracy', 0):.3f}, "
                      f"Side Effects={metrics.get('total_side_effects', 0)}")
            else:
                print(f"K={k}: {k_results.get('status', 'unknown')}")
    
    print(f"\n🎉 Evaluation complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
