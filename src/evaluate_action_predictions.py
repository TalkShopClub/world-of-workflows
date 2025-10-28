#!/usr/bin/env python3
"""
Evaluate action prediction accuracy by comparing model predictions with ground truth actions.
Takes model name as parameter and compares predictions with trajectory files.
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import difflib

def load_trajectory_actions(trajectory_file: Path) -> List[Dict[str, Any]]:
    """Load ground truth actions from trajectory file"""
    with open(trajectory_file, "r") as f:
        trajectory = json.load(f)
    
    actions = []
    for step in trajectory:
        if "action" in step:
            actions.append(step["action"])
    
    return actions

def load_predicted_actions(prediction_file: Path) -> List[Dict[str, Any]]:
    """Load predicted actions from prediction file"""
    with open(prediction_file, "r") as f:
        prediction_data = json.load(f)
    
    return prediction_data.get("predictions", [])

def compare_actions(ground_truth: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a single ground truth action with predicted action"""
    comparison = {
        "tool_name_match": ground_truth.get("tool_name") == predicted.get("tool_name"),
        "ground_truth_tool": ground_truth.get("tool_name"),
        "predicted_tool": predicted.get("tool_name"),
        "parameter_matches": 0,
        "parameter_mismatches": 0,
        "parameter_details": {}
    }
    
    gt_params = ground_truth.get("parameters", {})
    pred_params = predicted.get("parameters", {})
    
    # Get all unique parameter keys
    all_keys = set(gt_params.keys()) | set(pred_params.keys())
    
    for key in all_keys:
        gt_value = gt_params.get(key)
        pred_value = pred_params.get(key)
        
        if gt_value == pred_value:
            comparison["parameter_matches"] += 1
            comparison["parameter_details"][key] = {
                "match": True,
                "ground_truth": gt_value,
                "predicted": pred_value
            }
        else:
            comparison["parameter_mismatches"] += 1
            comparison["parameter_details"][key] = {
                "match": False,
                "ground_truth": gt_value,
                "predicted": pred_value
            }
    
    return comparison

def calculate_accuracy_metrics(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics from comparisons"""
    if not comparisons:
        return {
            "tool_name_accuracy": 0.0,
            "parameter_accuracy": 0.0,
            "exact_match_accuracy": 0.0,
            "total_actions": 0,
            "total_parameters": 0
        }
    
    total_actions = len(comparisons)
    tool_name_matches = sum(1 for c in comparisons if c["tool_name_match"])
    total_parameters = sum(c["parameter_matches"] + c["parameter_mismatches"] for c in comparisons)
    parameter_matches = sum(c["parameter_matches"] for c in comparisons)
    exact_matches = sum(1 for c in comparisons if c["tool_name_match"] and c["parameter_mismatches"] == 0)
    
    return {
        "tool_name_accuracy": tool_name_matches / total_actions if total_actions > 0 else 0.0,
        "parameter_accuracy": parameter_matches / total_parameters if total_parameters > 0 else 0.0,
        "exact_match_accuracy": exact_matches / total_actions if total_actions > 0 else 0.0,
        "total_actions": total_actions,
        "total_parameters": total_parameters,
        "tool_name_matches": tool_name_matches,
        "parameter_matches": parameter_matches,
        "exact_matches": exact_matches
    }

def evaluate_trajectory(trajectory_file: Path, prediction_file: Path) -> Dict[str, Any]:
    """Evaluate a single trajectory file against its predictions"""
    try:
        # Load ground truth and predictions
        ground_truth_actions = load_trajectory_actions(trajectory_file)
        predicted_actions = load_predicted_actions(prediction_file)
        
        # Ensure we have the same number of actions
        min_length = min(len(ground_truth_actions), len(predicted_actions))
        ground_truth_actions = ground_truth_actions[:min_length]
        predicted_actions = predicted_actions[:min_length]
        
        # Compare each action pair
        comparisons = []
        for i, (gt_action, pred_action) in enumerate(zip(ground_truth_actions, predicted_actions)):
            comparison = compare_actions(gt_action, pred_action)
            comparison["step_index"] = i
            comparisons.append(comparison)
        
        # Calculate metrics
        metrics = calculate_accuracy_metrics(comparisons)
        
        return {
            "trajectory_file": trajectory_file.name,
            "prediction_file": prediction_file.name,
            "success": True,
            "metrics": metrics,
            "comparisons": comparisons,
            "ground_truth_count": len(ground_truth_actions),
            "predicted_count": len(predicted_actions)
        }
        
    except Exception as e:
        return {
            "trajectory_file": trajectory_file.name,
            "prediction_file": prediction_file.name,
            "success": False,
            "error": str(e),
            "metrics": calculate_accuracy_metrics([])
        }

def find_matching_files(trajectories_dir: Path, predictions_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching trajectory and prediction files"""
    trajectory_files = list(trajectories_dir.glob("*.json"))
    matching_pairs = []
    
    for traj_file in trajectory_files:
        # Look for corresponding prediction file
        pred_file = predictions_dir / f"{traj_file.stem}_action_predictions.json"
        if pred_file.exists():
            matching_pairs.append((traj_file, pred_file))
    
    return matching_pairs

def aggregate_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all trajectory evaluations"""
    successful_results = [r for r in evaluation_results if r["success"]]
    
    if not successful_results:
        return {
            "overall_tool_name_accuracy": 0.0,
            "overall_parameter_accuracy": 0.0,
            "overall_exact_match_accuracy": 0.0,
            "total_trajectories": len(evaluation_results),
            "successful_trajectories": 0,
            "total_actions": 0,
            "total_parameters": 0
        }
    
    # Calculate weighted averages
    total_actions = sum(r["metrics"]["total_actions"] for r in successful_results)
    total_parameters = sum(r["metrics"]["total_parameters"] for r in successful_results)
    
    if total_actions == 0:
        return {
            "overall_tool_name_accuracy": 0.0,
            "overall_parameter_accuracy": 0.0,
            "overall_exact_match_accuracy": 0.0,
            "total_trajectories": len(evaluation_results),
            "successful_trajectories": len(successful_results),
            "total_actions": 0,
            "total_parameters": 0
        }
    
    # Weighted by number of actions
    weighted_tool_accuracy = sum(
        r["metrics"]["tool_name_accuracy"] * r["metrics"]["total_actions"] 
        for r in successful_results
    ) / total_actions
    
    weighted_parameter_accuracy = sum(
        r["metrics"]["parameter_accuracy"] * r["metrics"]["total_parameters"] 
        for r in successful_results
    ) / total_parameters if total_parameters > 0 else 0.0
    
    weighted_exact_accuracy = sum(
        r["metrics"]["exact_match_accuracy"] * r["metrics"]["total_actions"] 
        for r in successful_results
    ) / total_actions
    
    return {
        "overall_tool_name_accuracy": weighted_tool_accuracy,
        "overall_parameter_accuracy": weighted_parameter_accuracy,
        "overall_exact_match_accuracy": weighted_exact_accuracy,
        "total_trajectories": len(evaluation_results),
        "successful_trajectories": len(successful_results),
        "total_actions": total_actions,
        "total_parameters": total_parameters,
        "per_trajectory_metrics": {
            r["trajectory_file"]: r["metrics"] for r in successful_results
        }
    }

def print_detailed_report(evaluation_results: List[Dict[str, Any]], aggregated_metrics: Dict[str, Any]):
    """Print detailed evaluation report"""
    print(f"\n{'='*80}")
    print(f"DETAILED EVALUATION REPORT")
    print(f"{'='*80}")
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Tool Name Accuracy: {aggregated_metrics['overall_tool_name_accuracy']:.3f}")
    print(f"  Parameter Accuracy: {aggregated_metrics['overall_parameter_accuracy']:.3f}")
    print(f"  Exact Match Accuracy: {aggregated_metrics['overall_exact_match_accuracy']:.3f}")
    print(f"  Total Trajectories: {aggregated_metrics['total_trajectories']}")
    print(f"  Successful Evaluations: {aggregated_metrics['successful_trajectories']}")
    print(f"  Total Actions: {aggregated_metrics['total_actions']}")
    print(f"  Total Parameters: {aggregated_metrics['total_parameters']}")
    
    print(f"\n📋 PER-TRAJECTORY BREAKDOWN:")
    for result in evaluation_results:
        if result["success"]:
            metrics = result["metrics"]
            print(f"  {result['trajectory_file']}:")
            print(f"    Tool Name: {metrics['tool_name_accuracy']:.3f} ({metrics['tool_name_matches']}/{metrics['total_actions']})")
            print(f"    Parameters: {metrics['parameter_accuracy']:.3f} ({metrics['parameter_matches']}/{metrics['total_parameters']})")
            print(f"    Exact Match: {metrics['exact_match_accuracy']:.3f} ({metrics['exact_matches']}/{metrics['total_actions']})")
        else:
            print(f"  {result['trajectory_file']}: ❌ FAILED - {result.get('error', 'Unknown error')}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate action prediction accuracy")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model name to evaluate (e.g., 'gpt-4o', 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed per-action comparisons in output"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    trajectories_dir = script_dir / "trajectories"
    predictions_dir = script_dir / "action_predictions_results" / args.model
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Evaluating model: {args.model}")
    print(f"📁 Trajectories directory: {trajectories_dir}")
    print(f"📁 Predictions directory: {predictions_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if directories exist
    if not trajectories_dir.exists():
        print(f"❌ Trajectories directory not found: {trajectories_dir}")
        return
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    # Find matching files
    matching_pairs = find_matching_files(trajectories_dir, predictions_dir)
    print(f"📋 Found {len(matching_pairs)} matching trajectory-prediction pairs")
    
    if not matching_pairs:
        print("❌ No matching files found!")
        return
    
    # Evaluate each trajectory
    evaluation_results = []
    for i, (traj_file, pred_file) in enumerate(matching_pairs, 1):
        print(f"\n[{i}/{len(matching_pairs)}] Evaluating {traj_file.name}")
        result = evaluate_trajectory(traj_file, pred_file)
        evaluation_results.append(result)
        
        if result["success"]:
            metrics = result["metrics"]
            print(f"  ✅ Tool: {metrics['tool_name_accuracy']:.3f}, Params: {metrics['parameter_accuracy']:.3f}, Exact: {metrics['exact_match_accuracy']:.3f}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Aggregate metrics
    aggregated_metrics = aggregate_metrics(evaluation_results)
    
    # Print detailed report
    print_detailed_report(evaluation_results, aggregated_metrics)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_{args.model}_{timestamp}.json"
    
    output_data = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "aggregated_metrics": aggregated_metrics,
        "trajectory_results": evaluation_results if args.detailed else [
            {
                "trajectory_file": r["trajectory_file"],
                "success": r["success"],
                "metrics": r["metrics"] if r["success"] else None,
                "error": r.get("error") if not r["success"] else None
            }
            for r in evaluation_results
        ]
    }
    
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    main()
