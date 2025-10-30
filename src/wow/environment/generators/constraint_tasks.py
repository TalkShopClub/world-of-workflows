import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def _load_policies(data_folder: Path) -> List[str]:
    """Load policy names from constraint violation data folder"""
    policies = []
    for constraint_num in range(1, 100):
        try:
            with open(data_folder / f"constraint{constraint_num}" / "answer.json", "r") as f:
                policy = json.load(f)
                policies.append(policy["policy_name"])
        except FileNotFoundError:
            break
    return policies


def _load_trajectories(
    data_folder: Path,
    trajectory_type: str
) -> Tuple[List[Dict], List[List[int]], List[List[int]]]:
    """
    Load trajectories and ground truth data based on trajectory type.

    Args:
        data_folder: Path to constraint violation data folder
        trajectory_type: Type of trajectories to load ("original", "combined", "masked", "all")

    Returns:
        Tuple of (trajectories, gt_answer_idxs, gt_constraint_nums)
    """
    gt_trajectories = []
    all_gt_answer_idxs = []
    all_gt_constraint_nums = []

    for constraint_num in tqdm(range(1, 100), desc="Loading trajectories"):
        if not os.path.exists(data_folder / f"constraint{constraint_num}" / "trajectory.json"):
            break

        if trajectory_type in ["original", "all"]:
            with open(data_folder / f"constraint{constraint_num}" / "trajectory.json", "r") as f:
                gt_trajectory = json.load(f)
                gt_trajectories.append(gt_trajectory)

            with open(data_folder / f"constraint{constraint_num}" / "answer.json", "r") as f:
                answer = json.load(f)
                all_gt_constraint_nums.append(answer["invalid_policy_nums"])
                all_gt_answer_idxs.append(answer["invalid_action_idxs"])

        if trajectory_type in ["combined", "all"]:
            combined_trajectory_folders = []
            for folder in sorted(os.listdir(data_folder)):
                if folder.startswith(f"combined_trajectory_{constraint_num}_"):
                    combined_trajectory_folders.append(folder)

            for folder in sorted(combined_trajectory_folders, key=lambda x: int(x.split("_")[-2])):
                with open(data_folder / folder / "trajectory.json", "r") as f:
                    combined_trajectory = json.load(f)
                    gt_trajectories.append(combined_trajectory)

                with open(data_folder / folder / "answer.json", "r") as f:
                    combined_answer = json.load(f)
                    all_gt_answer_idxs.append(combined_answer["invalid_action_idxs"])
                    all_gt_constraint_nums.append(combined_answer["invalid_policy_nums"])

        if trajectory_type in ["masked", "all"]:
            if os.path.exists(data_folder / f"perturbed_trajectory_{constraint_num}" / "trajectory.json"):
                with open(data_folder / f"perturbed_trajectory_{constraint_num}" / "trajectory.json", "r") as f:
                    perturbed_trajectory = json.load(f)
                    gt_trajectories.append(perturbed_trajectory)
                    all_gt_answer_idxs.append([-1])
                    all_gt_constraint_nums.append([-1])

    return gt_trajectories, all_gt_answer_idxs, all_gt_constraint_nums


async def generate_constraint_violation_predictions(
    agent,
    data_folder: Path = None,
    trajectory_type: str = "combined",
    mode: str = "state_action",
    perfect_schema: bool = True,
    output_file: Path = None
):
    """
    Generate constraint violation predictions for trajectories.

    Args:
        agent: WorldModelAgent instance
        data_folder: Path to constraint violation data folder
        trajectory_type: Type of trajectories ("original", "combined", "masked", "all")
        mode: Prediction mode ("action_only", "state_action", "state_only")
        perfect_schema: Whether to use perfect schema
        output_file: Output file path for results
    """
    if data_folder is None:
        base_dir = Path(__file__).parent.parent.parent.parent
        data_folder = base_dir / "constraint_violation_data"

    policies = _load_policies(data_folder)
    gt_trajectories, all_gt_answer_idxs, all_gt_constraint_nums = _load_trajectories(
        data_folder, trajectory_type
    )

    results = []
    total_input_tokens = 0
    total_cost = 0
    total_json_errors = 0
    context_window_errors = 0

    for trajectory, gt_answer_idxs, gt_constraint_nums in tqdm(
        zip(gt_trajectories, all_gt_answer_idxs, all_gt_constraint_nums),
        desc="Predicting constraint violations",
        total=len(gt_trajectories)
    ):
        try:
            pred_constraint_violation, usage = await agent.predict_constraint_violation(
                trajectory, policies, mode=mode, perfect_schema=perfect_schema
            )
        except Exception:
            context_window_errors += 1
            continue

        total_input_tokens += usage.prompt_tokens
        total_cost += usage.cost

        if pred_constraint_violation is None:
            total_json_errors += 1

        result = {
            "gt_answer_idxs": gt_answer_idxs,
            "gt_constraint_nums": gt_constraint_nums,
            "llm_constraint_nums": pred_constraint_violation["violated_policy_idxs"] if pred_constraint_violation is not None else [],
            "llm_answer_idxs": pred_constraint_violation["invalid_action_idxs"] if pred_constraint_violation is not None else [],
            "llm_reason": pred_constraint_violation["reason"] if pred_constraint_violation is not None else "",
        }

        results.append(result)

    output_data = {
        "results": results,
        "metadata": {
            "total_input_tokens": total_input_tokens,
            "total_cost": total_cost,
            "total_json_errors": total_json_errors,
            "context_window_errors": context_window_errors,
            "trajectory_type": trajectory_type,
            "mode": mode,
            "perfect_schema": perfect_schema
        }
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    return output_data
