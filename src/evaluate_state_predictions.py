#!/usr/bin/env python3
"""
Evaluate state predictions for k=1 to 5 using the evaluation pipeline.
Compares predicted state diffs with ground truth from trajectory files.
"""

import json
import os
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm

# Import local modules (support both direct execution and module import)
try:
    from .world_model_agent import StateDiff, ActionCall, WorldModelAgent, SysAuditRecord, AdditionalInformation, operation
    from .eval_pipeline import compute_state_rollout_metrics, state_rollout_evaluation
except ImportError:
    from world_model_agent import StateDiff, ActionCall, WorldModelAgent, SysAuditRecord, AdditionalInformation, operation
    try:
        from eval_pipeline import compute_state_rollout_metrics, state_rollout_evaluation
    except ImportError:
        def compute_state_rollout_metrics(pred, gt):
            return [], [], []
        def state_rollout_evaluation(pred, gt, verbose=False):
            return compute_state_rollout_metrics(pred, gt)

class StatePredictionEvaluator:
    """Evaluate state predictions against ground truth trajectory data."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        self.table_schemas = self.load_table_schemas()
    
    def load_table_schemas(self) -> Dict[str, Dict[str, str]]:
        """Load table schemas and create lookup dictionary for field types."""
        print("📋 Loading table schemas for filtering...")
        
        try:
            schema_file = Path(__file__).parent / "prompts" / "all_table_schemas.json"
            with open(schema_file, 'r') as f:
                all_schemas = json.load(f)
            
            # Create lookup dictionary: {table_name: {field_name: internal_type}}
            table_schemas = {}
            for table_name, columns in all_schemas.items():
                table_schemas[table_name] = {}
                for column in columns:
                    field_name = column.get('element', '')
                    internal_type = column.get('internal_type', '')
                    if field_name and internal_type:
                        table_schemas[table_name][field_name] = internal_type
            
            print(f"✅ Loaded schemas for {len(table_schemas)} tables")
            return table_schemas
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load table schemas: {e}")
            return {}
    
    def should_include_audit_record(self, tablename: str, fieldname: str) -> bool:
        """Check if an audit record should be included in evaluation."""
        # Filter out workflow tables
        if tablename.startswith('wf_'):
            return False
        
        # Check field internal type
        if tablename in self.table_schemas:
            field_type = self.table_schemas[tablename].get(fieldname)
            if not field_type or field_type in ['glide_date_time', 'glide_duration', 'glide_time']:
                return False
        else:
            # Log warning if table not found in schema
            print(f"⚠️ Warning: Table '{tablename}' not found in schema, including audit record")
        
        return True
    
    def load_trajectory_ground_truth(self, trajectory_file: str, k: int) -> List[StateDiff]:
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
                    # Convert audits to StateDiff format
                    sysaudit_records = []
                    filtered_count = 0
                    for audit in step['audits']:
                        # Apply filtering logic
                        if self.should_include_audit_record(audit['tablename'], audit['fieldname']):
                            record = SysAuditRecord(
                                tablename=audit['tablename'],
                                fieldname=audit['fieldname'],
                                oldvalue=audit['oldvalue'],
                                newvalue=audit['newvalue']
                            )
                            sysaudit_records.append(record)
                        else:
                            filtered_count += 1
                    
                    if filtered_count > 0:
                        print(f"🔍 Filtered out {filtered_count} audit records from step {i+1}")
                    
                    # Create additional information
                    additional_info = AdditionalInformation(
                        num_audits=len(sysaudit_records),
                        num_modified_entries=len([r for r in sysaudit_records if r.oldvalue != ""]),
                        num_deleted_entries=0,  # Not tracked in current format
                        num_created_entries=len([r for r in sysaudit_records if r.oldvalue == ""]),
                        operation_type=[operation.post if r.oldvalue == "" else operation.put for r in sysaudit_records],
                        tables_modified=list(set(r.tablename for r in sysaudit_records))
                    )
                    
                    state_diff = StateDiff(
                        sysauditrecord=sysaudit_records,
                        additional_information=additional_info
                    )
                    state_diffs.append(state_diff)
                else:
                    # Create empty state diff if no audits
                    state_diff = StateDiff(
                        sysauditrecord=[],
                        additional_information=AdditionalInformation(
                            num_audits=0,
                            num_modified_entries=0,
                            num_deleted_entries=0,
                            num_created_entries=0,
                            operation_type=[],
                            tables_modified=[]
                        )
                    )
                    state_diffs.append(state_diff)
            
            print(f"✅ Loaded {len(state_diffs)} ground truth state diffs")
            return state_diffs
            
        except Exception as e:
            print(f"❌ Error loading trajectory: {e}")
            return []
    
    def load_predicted_states(self, prediction_file: str) -> Tuple[List[StateDiff], bool, str]:
        """Load predicted state diffs from prediction file.
        
        Returns:
            Tuple of (predicted_states, has_schema_error, error_message)
        """
        print(f"📂 Loading predictions from: {prediction_file}")
        
        try:
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            predicted_states = []
            has_schema_error = False
            error_message = ""
            
            for state_data in prediction_data['predicted_states']:
                # Check for schema errors in additional_information
                additional_data = state_data.get('additional_information', {})
                if 'error' in additional_data and 'schema file not found' in additional_data['error'].lower():
                    has_schema_error = True
                    error_message = additional_data['error']
                    print(f"⚠️ Schema error detected: {error_message}")
                    break
                
                # Convert sysaudit records
                sysaudit_records = []
                for record_data in state_data.get('sysauditrecord', []):
                    record = SysAuditRecord(
                        tablename=record_data['tablename'],
                        fieldname=record_data['fieldname'],
                        oldvalue=record_data['oldvalue'],
                        newvalue=record_data['newvalue']
                    )
                    sysaudit_records.append(record)
                
                # Convert additional information
                additional_info = AdditionalInformation(
                    num_audits=additional_data.get('num_audits', 0),
                    num_modified_entries=additional_data.get('num_modified_entries', 0),
                    num_deleted_entries=additional_data.get('num_deleted_entries', 0),
                    num_created_entries=additional_data.get('num_created_entries', 0),
                    operation_type=[operation(op) for op in additional_data.get('operation_type', [])],
                    tables_modified=additional_data.get('tables_modified', [])
                )
                
                state_diff = StateDiff(
                    sysauditrecord=sysaudit_records,
                    additional_information=additional_info
                )
                predicted_states.append(state_diff)
            
            if not has_schema_error:
                print(f"✅ Loaded {len(predicted_states)} predicted state diffs")
            else:
                print(f"❌ Skipping due to schema error: {error_message}")
            
            return predicted_states, has_schema_error, error_message
            
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return [], True, str(e)
    
    def evaluate_single_file(self, prediction_file: str, trajectory_file: str, k: int) -> Dict[str, Any]:
        """Evaluate a single prediction file against its corresponding trajectory."""
        print(f"\n🎯 Evaluating: {os.path.basename(prediction_file)}")
        print("-" * 60)
        
        # Load ground truth and predictions
        gt_states = self.load_trajectory_ground_truth(trajectory_file, k)
        pred_states, has_schema_error, error_message = self.load_predicted_states(prediction_file)
        
        # Check for schema errors
        if has_schema_error:
            print(f"❌ Skipping evaluation due to schema error: {error_message}")
            return {
                'file': os.path.basename(prediction_file),
                'status': 'skipped_schema_error',
                'error': error_message,
                'k': k
            }
        
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
            full_accuracy, iou_metrics, side_effects = compute_state_rollout_metrics(pred_states, gt_states)
            
            # Calculate summary statistics
            avg_full_accuracy = sum(full_accuracy) / len(full_accuracy) if full_accuracy else 0
            avg_sysaudit_iou = sum([iou[0] for iou in iou_metrics]) / len(iou_metrics) if iou_metrics else 0
            avg_additional_info_iou = sum([iou[1] for iou in iou_metrics]) / len(iou_metrics) if iou_metrics else 0
            total_side_effects = sum([se[0] for se in side_effects])
            avg_side_effects = sum([se[0] for se in side_effects]) / len(side_effects) if side_effects else 0
            
            result = {
                'file': os.path.basename(prediction_file),
                'status': 'success',
                'k': k,
                'num_steps': len(gt_states),
                'metrics': {
                    'full_accuracy': avg_full_accuracy,
                    'sysaudit_iou': avg_sysaudit_iou,
                    'additional_info_iou': avg_additional_info_iou,
                    'total_side_effects': total_side_effects,
                    'avg_side_effects_per_step': avg_side_effects
                },
                'step_by_step': {
                    'full_accuracy': full_accuracy,
                    'iou_metrics': iou_metrics,
                    'side_effects': [(se[0], [f"{r.tablename}.{r.fieldname}" for r in se[1]]) for se in side_effects]
                }
            }
            
            print(f"✅ Evaluation complete:")
            print(f"   Full Accuracy: {avg_full_accuracy:.3f}")
            print(f"   SysAudit IoU: {avg_sysaudit_iou:.3f}")
            print(f"   Additional Info IoU: {avg_additional_info_iou:.3f}")
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
        """Evaluate all prediction files for a specific k value."""
        print(f"\n🚀 Evaluating k={k}")
        print("=" * 60)
        
        if not prediction_dir.exists():
            print(f"❌ No predictions found for k={k}")
            return {'k': k, 'status': 'no_data'}
        
        # Get all prediction files
        prediction_files = list(prediction_dir.glob(f"*_k*.json"))
        print(f"Found {len(prediction_files)} prediction files for k={k}")
        
        if not prediction_files:
            print(f"❌ No prediction files found for k={k}")
            return {'k': k, 'status': 'no_files'}
        
        results = []
        successful = 0
        failed = 0
        skipped_schema_error = 0
        
        for prediction_file in tqdm(prediction_files, desc=f"Evaluating k={k}"):
            trajectory_file = trajectory_dir / f"{prediction_file.stem[:-3]}.json"
            
            if not trajectory_file.exists():
                print(f"⚠️ Trajectory file not found: {trajectory_file}")
                failed += 1
                continue
            
            result = self.evaluate_single_file(str(prediction_file), str(trajectory_file), k)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            elif result['status'] == 'skipped_schema_error':
                skipped_schema_error += 1
            else:
                failed += 1
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            full_accuracies = [r['metrics']['full_accuracy'] for r in successful_results]
            sysaudit_ious = [r['metrics']['sysaudit_iou'] for r in successful_results]
            additional_info_ious = [r['metrics']['additional_info_iou'] for r in successful_results]
            side_effects_list = [r['metrics']['total_side_effects'] for r in successful_results]
            avg_side_effects_list = [r['metrics']['avg_side_effects_per_step'] for r in successful_results]
            
            aggregate_metrics = {
                'full_accuracy': sum(full_accuracies) / len(full_accuracies),
                'sysaudit_iou': sum(sysaudit_ious) / len(sysaudit_ious),
                'additional_info_iou': sum(additional_info_ious) / len(additional_info_ious),
                'total_side_effects': sum(side_effects_list),
                'avg_side_effects_per_file': sum(avg_side_effects_list) / len(avg_side_effects_list)
            }
        else:
            aggregate_metrics = {}
        
        return {
            'k': k,
            'total_files': len(prediction_files),
            'successful': successful,
            'failed': failed,
            'skipped_schema_error': skipped_schema_error,
            'aggregate_metrics': aggregate_metrics,
            'file_results': results
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate state predictions for k=1 to 5")
    parser.add_argument("--state-pred-path", required=True, 
                       help="Path to directory containing state prediction files (model name will be inferred from innermost folder)")
    parser.add_argument("--k-values", nargs='+', type=int, default=[1, 2, 3, 4, 5], 
                       help="K values to evaluate (default: 1 2 3 4 5)")
    parser.add_argument("--trajectory-dir", default="trajectories", 
                       help="Directory containing trajectory files (default: trajectories)")
    parser.add_argument("--output-dir", default="evaluation_results", 
                       help="Output directory for results (default: evaluation_results)")
    
    args = parser.parse_args()
    
    # Infer model name from the innermost folder name
    state_pred_path = Path(args.state_pred_path)
    model_name = state_pred_path.name
    
    print("🎯 State Prediction Evaluator")
    print("=" * 60)
    print(f"State prediction path: {args.state_pred_path}")
    print(f"Inferred model name: {model_name}")
    print(f"K values: {args.k_values}")
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    prediction_dir = state_pred_path
    trajectory_dir = script_dir / args.trajectory_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not prediction_dir.exists():
        print(f"❌ Prediction directory not found: {prediction_dir}")
        return
    
    if not trajectory_dir.exists():
        print(f"❌ Trajectory directory not found: {trajectory_dir}")
        return
    
    # Initialize evaluator
    evaluator = StatePredictionEvaluator(model_name)
    
    # Evaluate each k value
    all_results = {
        'model': model_name,
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
        
        if k_results.get('status') == 'success':
            metrics = k_results['aggregate_metrics']
            print(f"\n📊 K={k} Summary:")
            print(f"   Files processed: {k_results['successful']}/{k_results['total_files']}")
            print(f"   Skipped (schema error): {k_results.get('skipped_schema_error', 0)}")
            print(f"   Failed: {k_results.get('failed', 0)}")
            if metrics:
                print(f"   Full Accuracy: {metrics.get('full_accuracy', 0):.3f}")
                print(f"   SysAudit IoU: {metrics.get('sysaudit_iou', 0):.3f}")
                print(f"   Additional Info IoU: {metrics.get('additional_info_iou', 0):.3f}")
                print(f"   Total Side Effects: {metrics.get('total_side_effects', 0)}")
            else:
                print("   No valid predictions to evaluate")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"state_prediction_evaluation_{model_name}_{timestamp}.json"
    evaluator.save_results(all_results, str(output_file))
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for k in args.k_values:
        k_key = f'k={k}'
        if k_key in all_results['evaluations']:
            k_results = all_results['evaluations'][k_key]
            if 'aggregate_metrics' in k_results and k_results['aggregate_metrics']:
                metrics = k_results['aggregate_metrics']
                print(f"K={k}: Files={k_results.get('successful', 0)}/{k_results.get('total_files', 0)}, "
                      f"Skipped={k_results.get('skipped_schema_error', 0)}, "
                      f"Full Acc={metrics.get('full_accuracy', 0):.3f}, "
                      f"SysAudit IoU={metrics.get('sysaudit_iou', 0):.3f}, "
                      f"Side Effects={metrics.get('total_side_effects', 0)}")
            else:
                print(f"K={k}: Files={k_results.get('successful', 0)}/{k_results.get('total_files', 0)}, "
                      f"Skipped={k_results.get('skipped_schema_error', 0)}, "
                      f"Status={k_results.get('status', 'no_data')}")
    
    print(f"\n🎉 Evaluation complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
