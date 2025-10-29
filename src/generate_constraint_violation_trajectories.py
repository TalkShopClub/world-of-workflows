"""
This script assigns 3 MCP action-trajectory for each constraint. Each function in this file corresponds to a different constraint.
For each trajectory, MCP actions will be executed, the state changes will be recorded and this (action, state) pair for the entire trajectory will be saved to file. 
""" 

from pathlib import Path
import json
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import asyncio
from datetime import datetime
import pytz
from rest_apis.world_model_scripts.states import get_sys_audit
from rest_apis.world_model_scripts.world_model_agent import WorldModelAgent
from browsergym.workarena.instance import SNowInstance
import random
from tqdm import tqdm

def save_invalid_action(policy_name, invalid_action_idx, policy_num):
    """Append to the trajectory file""" 
    save_dir = Path(__file__).parent / "qa_data" / f"constraint{policy_num}"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "answer.json"

    policy_action = {
        "policy_name": policy_name,
        "invalid_action_idx": invalid_action_idx
    }
    with open(output_path, "w") as f:
        json.dump(policy_action, f, indent=2)

def generate_high_priority_incident_user_edit_violation():
    """Generate trajectories that violate the constraint: Editing the assigned user for a high priority incident ticket is not allowed"""
    policy = "Editing the assigned user for priority 1 incident ticket is not allowed unless there is no user assigned to the incident ticket"
    invalid_action_idx = 5 # 0-indexed

    trajectory = [
        # Created an incident ticket with priority 1 already (c3d609bc53103210f8bb51a0a0490e2a) 
        # Created a new incident ticket with priority 2 and assign to user (user_id: 005d500b536073005e0addeeff7b12f4) (incident sys_id: 7b18c93453503210f8bb51a0a0490e95)
        # Created a new incident ticket with priority 1 and assign to user (user_id: 02826bf03710200044e0bfc8bcbe5d55) (incident sys_id: f42909b453503210f8bb51a0a0490e90)
        # Assigned the incident ticket c3d609bc53103210f8bb51a0a0490e2a to user 02826bf03710200044e0bfc8bcbe5d5e 
        # Updated the incident ticket 7b18c93453503210f8bb51a0a0490e95 to priority 1 
        # Edited the assigned user for the incident ticket 7b18c93453503210f8bb51a0a0490e95 to user 02826bf03710200044e0bfc8bcbe5d55 (INVALID ACTION) 
    ]  
    save_invalid_action(policy, invalid_action_idx, 1) 

def generate_incident_resolution_assigned_user_violation():
    """Generate trajectories that violate the constraint: Ticket being resolved must have someone assigned to the ticket."""
    policy = "Ticket being resolved must have someone assigned to the ticket."
    invalid_action_idx = 7 # 0-indexed
    trajectory = [
        # Created an incident ticket (sys_id: e52f7e515354f210f8bb51a0a0490ebc)
        # Created another incident ticket with 88279497538c7210f8bb51a0a0490eff assigned to the ticket (sys_id: ffff7e915354f210f8bb51a0a0490e98)
        # Created another incident ticket with no one assigned to the ticket (sys_id: f28003d15354f210f8bb51a0a0490e93)
        # Assign user 46bac3d6a9fe1981005f299d979b8869 to first incident ticket 
        # Remove user from 2nd incident ticket 
        # Assign previous user 88279497538c7210f8bb51a0a0490eff assigned to 2nd incident ticket to third incident ticket (sys_id: f28003d15354f210f8bb51a0a0490e93) 
        # Change user for first incident ticket to 46b87022a9fe198101a78787e40d7547
        # Resolve the 2nd incident ticket (INVALID ACTION) 
    ] 
    save_invalid_action(policy, invalid_action_idx, 2)

def generate_priority_one_incident_assignment_violation():
    """Generate trajectories that violate the constraint: Incidents with priority 1 cannot be assigned to users who already have more than 3 priority 1 incidents"""
    policy = "Incidents with priority 1 cannot be assigned to users who already have more than 3 priority 1 incidents"
    invalid_action_idx = 8 # 0-indexed
    trajectory = [
        # Create first priority 1 incident and assign to first user 5137153cc611227c000bbd1bd8cd2005 (sys_id: 18a680e95398f210f8bb51a0a0490e45)
        # Create second priority 1 incident and assign to first user  (sys_id: 9f278c2153d8f210f8bb51a0a0490ead)
        # Create third priority 1 incident and assign to first user (now first user has 3 priority 1 incidents) (sys_id: fe08886153d8f210f8bb51a0a0490ed2)
        # Create priority 2 incident and assign to second user 681ccaf9c0a8016400b98a06818d57c7 (sys_id: 21984ca153d8f210f8bb51a0a0490e47)
        # Create priority 3 incident and assign to third user 9ee1b13dc6112271007f9d0efdb69cd0 (sys_id: 7a3908e153d8f210f8bb51a0a0490ea5)
        # Update priority 2 incident to priority 1 (second user now has 1 priority 1 incident) 
        # Create another priority 1 incident and assign to second user (second user now has 2 priority 1 incidents) (sys_id: 7a7a846553d8f210f8bb51a0a0490e1d)
        # Update the assigned user for priority 3 incident to first user
        # Update priority 3 incident to priority 1 (INVALID ACTION - first user now has 4 priority 1 incidents)
        # Edit the first incident ticket to priority 2 (first user now has 3 priority 1 incidents; VALID)
    ]
    save_invalid_action(policy, invalid_action_idx, 3) 


def generate_catalog_category_deletion_violation():
    """Generate trajectories that violate the constraint: Catalog categories cannot be deleted if they contain active items"""
    policy = "Catalog categories cannot be deleted if they contain active items"
    invalid_action_idx = 10 # 0-indexed
    trajectory = [
        # Create first catalog category for "Data Analytics Software" (sys_id: ef92d8ed53d8f210f8bb51a0a0490e4c)
        # Create second catalog category for "Business Intelligence Tools" (sys_id: 15131c21531cf210f8bb51a0a0490eee)
        # Create third catalog category for "Remote Work Accessories" (sys_id: 7d431861531cf210f8bb51a0a0490e48)
        # Create catalog item "Statistical Analysis Suite" in Data Analytics Software category (active by default) (sys_id: 356694a1531cf210f8bb51a0a0490ebb)
        # Create catalog item "Data Visualization Tool License" in Data Analytics Software category (active by default) (sys_id: a1d65c65531cf210f8bb51a0a0490e96)
        # Create catalog item "Executive Dashboard Platform" in Business Intelligence Tools category (active by default) (sys_id: e467d4a5531cf210f8bb51a0a0490e09)
        # Create catalog item "Ergonomic Desk Converter" in Remote Work Accessories category and set to inactive (sys_id: 34d7d0e553d8f210f8bb51a0a0490e80)
        # Move "Executive Dashboard Platform" item from Business Intelligence Tools to Data Analytics Software category
        # Set "Data Visualization Tool License" item to inactive
        # Delete the Remote Work Accessories category (VALID action -- desk converter is inactive)
        # Delete the Data Analytics Software category (INVALID ACTION - it still contains the "Executive Dashboard Platform" item which is active)
        # Delete the Business Intelligence Tools category (VALID action -- It has no items left)
    ]
    save_invalid_action(policy, invalid_action_idx, 4)


def generate_catalog_item_inactive_category_violation():
    """Generate trajectories that violate the constraint: Catalog items cannot be moved to categories that are inactive"""
    policy = "Catalog items cannot be moved to categories that are inactive"
    invalid_action_idx = 7 # 0-indexed
    trajectory = [
        # Create catalog category "Data Analytics Software" (active by default) (sys_id: 193f9821535cf210f8bb51a0a0490ef6)
        # Create catalog category "Business Intelligence Tools" (active by default) (sys_id: 5e6fd861535cf210f8bb51a0a0490ee6)
        # Create catalog category "Remote Work Accessories" (active by default) (sys_id: 0a8fd861535cf210f8bb51a0a0490ef0)
        # Create catalog item "Statistical Analysis Suite" in Data Analytics Software category (sys_id: 19bf9c61535cf210f8bb51a0a0490e67)
        # Create catalog item "Executive Dashboard Platform" in Business Intelligence Tools category (sys_id: 3820e4a1535cf210f8bb51a0a0490e86)
        # Set "Remote Work Accessories" category to inactive status
        # Move "Executive Dashboard Platform" from Business Intelligence Tools to Data Analytics Software category
        # Move "Statistical Analysis Suite" from Data Analytics Software to Remote Work Accessories category (INVALID ACTION - Remote Work Accessories is inactive)
    ]
    save_invalid_action(policy, invalid_action_idx, 5)


def generate_high_risk_change_friday_violation():
    """Generate trajectories that violate the constraint: Change requests with risk level 'High' cannot be scheduled on Fridays"""
    policy = "Change requests with risk level 'Moderate' or higher cannot be scheduled on Fridays"
    invalid_action_idx = 6 # 0-indexed
    trajectory = [
        # Create change request for "Reboot Windows Server" with Low risk, scheduled for Friday (VALID ACTION - Low risk change) (sys_id: 73252c29535cf210f8bb51a0a0490e26)
        # Create change request for "Database Migration" with Moderate risk, scheduled for Monday (sys_id: b3b5a469535cf210f8bb51a0a0490e4e)
        # Create change request for "Server Upgrade" with Low risk, scheduled for Wednesday (sys_id: be0660a9535cf210f8bb51a0a0490e6d)
        # Create change request for "Network Maintenance" with High risk, scheduled for Tuesday (sys_id: e75628a9535cf210f8bb51a0a0490e6c)
        # Update "Database Migration" change request risk level to High
        # Reschedule "Server Upgrade" to Thursday
        # Reschedule "Network Maintenance" to Friday (INVALID ACTION - Network Maintenance is a High risk change)
        # Reschedule "Database Migration" for Wednesday (VALID ACTION - Database Migration is a Moderate risk change)
    ]
    save_invalid_action(policy, invalid_action_idx, 6)


def generate_expensive_asset_transfer_violation():
    """Generate trajectories that violate the constraint: Assets that cost over $10000 cannot be transferred"""
    policy = "Assets that cost over $10000 cannot be transferred"
    invalid_action_idx = 6 # 0-indexed
    trajectory = [
        # Create asset "Dell Inc. XPS 15z" with cost $4500, asset tag D123 assigned to first user 5137153cc611227c000bbd1bd8cd2005 (sys_id: a355b9e953d03610f8bb51a0a0490e02)
        # Create asset "Iris 5875" with cost $12000, asset tag I120 assigned to second user 681ccaf9c0a8016400b98a06818d57c7 (sys_id: d8d5352553143610f8bb51a0a0490e43)
        # Create asset "Acer Notebook Battery" with cost $2000, asset tag A120 assigned to third user 9ee1b13dc6112271007f9d0efdb69cd0 (sys_id: c206352153143610f8bb51a0a0490ec2)
        # Update "Dell Inc. XPS 15z" cost to $6500
        # Transfer "Acer Notebook Battery" from third user to first user
        # Create asset "Dell Inc. XPS 15z" with cost $12000, asset tag D120 assigned to second user (sys_id: 67577da553143610f8bb51a0a0490eb1)
        # Transfer "Dell Inc. XPS 15z" with asset tag D120 from second user to fourth user 46b87022a9fe198101a78787e40d7547 (INVALID ACTION - cost is $12000, over $10000 limit)
        # Transfer "Dell Inc. XPS 15z" with asset tag D123 from first user to third user (VALID ACTION - cost is $4500, under $10000 limit)
    ]
    save_invalid_action(policy, invalid_action_idx, 7)


def generate_in_use_asset_serial_modification_violation():
    """Generate trajectories that violate the constraint: Serial numbers cannot be modified once asset is marked as 'In Use'"""
    policy = "Serial numbers cannot be modified once asset is marked as 'In Use'"
    invalid_action_idx = 8 # 0-indexed
    trajectory = [
        # Create asset "Dell Inc. XPS 15z" with cost $4500, asset tag D123, serial number "DL-4A7B-9C2D-E5F8-1234" in "In Stock" state (sys_id: 1b2a71e953143610f8bb51a0a0490e9c)
        # Create asset "Iris 5875" with cost $12000, asset tag I120, serial number "IR-8E3F-2A1B-C6D9-5678" in "In Stock" state (sys_id: ba8a71e953143610f8bb51a0a0490ea3)
        # Create asset "Acer Notebook Battery" with cost $2000, asset tag A120, serial number "AC-1F4A-7B8C-D2E5-9012" in "In Stock" state (sys_id: 7ebaf9e953143610f8bb51a0a0490e18)
        # Change "Dell Inc. XPS 15z" state to "In Use"
        # Update "Iris 5875" serial number to "IR-9F4G-3B2C-D7E0-6789"
        # Change "Iris 5875" state to "In Use"
        # Change "Acer Notebook Battery" state to "In Maintenance"
        # Update "Acer Notebook Battery" serial number to "AC-2G5B-8C9D-E3F6-0123" (VALID ACTION - asset is "In Maintenance")
        # Update "Dell Inc. XPS 15z" serial number to "DL-5B8C-0D3E-F6G9-2345" (INVALID ACTION - asset is "In Use")
    ]
    save_invalid_action(policy, invalid_action_idx, 8)


def generate_user_asset_limit_violation():
    """Generate trajectories that violate the constraint: Assets cannot be assigned to users who already have 5+ active assets"""
    policy = "Assets cannot be assigned to users who already have more than 5 active assets"
    invalid_action_idx = 10 # 0-indexed
    trajectory = [
        # Create asset "Dell Inc. XPS 15z" with asset tag D001 and assign to first user 5137153cc611227c000bbd1bd8cd2005 (first user: 1 asset) (sys_id: 4b8e316153543610f8bb51a0a0490e4d)
        # Create asset "Iris 5875" with asset tag I001 and assign to second user 681ccaf9c0a8016400b98a06818d57c7 (sys_id: 85ceb56d53143610f8bb51a0a0490e35)
        # Create asset "Acer Notebook Battery" with asset tag A001 and assign to first user 5137153cc611227c000bbd1bd8cd2005 (first user: 2 assets) (sys_id: 13ee396153543610f8bb51a0a0490e83)
        # Create asset "Dell Inc. XPS 15z" with asset tag D002 and assign to third user 9ee1b13dc6112271007f9d0efdb69cd0 (sys_id: 8e1f396153543610f8bb51a0a0490e8b)
        # Create asset "Acer Notebook Battery" with asset tag A003 initially unassigned (sys_id: 30af3d6153543610f8bb51a0a0490e46)
        # Create asset "Iris 5875" with asset tag I002 and assign to first user 5137153cc611227c000bbd1bd8cd2005 (first user: 3 assets) (sys_id: 13cff5a153543610f8bb51a0a0490eb9)
        # Create asset "Acer Notebook Battery" with asset tag A002 and assign to second user 681ccaf9c0a8016400b98a06818d57c7 (sys_id: c5ff7ded53143610f8bb51a0a0490e3d)
        # Create asset "Dell Inc. XPS 15z" with asset tag D003 and assign to first user 5137153cc611227c000bbd1bd8cd2005 (first user: 4 assets) (sys_id: c9208ea153543610f8bb51a0a0490e0f)
        # Create asset "Iris 5875" with asset tag I003 and assign to first user 5137153cc611227c000bbd1bd8cd2005 (first user: 5 assets) (sys_id: 964082e153543610f8bb51a0a0490e84)
        # Transfer "Acer Notebook Battery" with asset tag A002 from second user to third user 9ee1b13dc6112271007f9d0efdb69cd0
        # Transfer "Acer Notebook Battery" with asset tag A003 to first user 5137153cc611227c000bbd1bd8cd2005 (INVALID ACTION - first user already has 5 active assets)
    ]
    save_invalid_action(policy, invalid_action_idx, 9)


def generate_group_manager_removal_violation():
    """Generate trajectories that violate the constraint: Users cannot be removed from groups they manage"""
    policy = "Users cannot be removed from groups they manage"
    invalid_action_idx = 11 # 0-indexed
    trajectory = [
        # Create group "IT Support Team" (sys_id: 3f7706a953543610f8bb51a0a0490ebe)
        # Create group "HR Department" (sys_id: a1a7c26d53543610f8bb51a0a0490e07)
        # Create group "Finance Team" (sys_id: 27c7c26d53543610f8bb51a0a0490e0f)
        # Add first user 5137153cc611227c000bbd1bd8cd2005 to IT Support Team 
        # Set first user as manager of IT Support Team
        # Add second user 681ccaf9c0a8016400b98a06818d57c7 to HR Department and set as regular member
        # Add third user 9ee1b13dc6112271007f9d0efdb69cd0 to IT Support Team as regular member
        # Add first user 5137153cc611227c000bbd1bd8cd2005 to HR Department as regular member
        # Add fourth user 46b87022a9fe198101a78787e40d7547 to HR Department 
        # Set fourth user as manager of HR Department
        # Set manager of Finance Team to third user
        # Remove first user from IT Support Team (INVALID ACTION - first user is the manager of IT Support Team)
        # Remove first user from HR Department (VALID ACTION - first user is not the manager of HR Department)
        # Remove second user from HR Department (VALID ACTION - second user is not the manager of HR Department)
    ]
    save_invalid_action(policy, invalid_action_idx, 10)


def generate_inactive_user_manager_assignment_violation():
    """Generate trajectories that violate the constraint: Inactive users cannot be assigned as managers of active users"""
    policy = "Inactive users cannot be assigned as managers of active users"
    invalid_action_idx = 13 # 0-indexed
    trajectory = [
        # Create first user (active by default) (sys_id: 2423743153d43210f8bb51a0a0490e65)
        # Create second user (active by default) (sys_id: 0e53b43153d43210f8bb51a0a0490ed6)
        # Create group "HR Department" (sys_id: b3e3bcb153143210f8bb51a0a0490ea3)
        # Create group "Finance Team" (sys_id: a014f87153d43210f8bb51a0a0490ee3)
        # Create third user (active by default) (sys_id: 1fb4fc7153d43210f8bb51a0a0490e51)
        # Add first and third user to HR Department 
        # Add second user to Finance Team
        # Set third user as manager of HR Department
        # Remove first user from HR Department
        # Set second user as manager of third user
        # Set first user as manager of second user
        # Set second user status to inactive
        # Create fourth user (active by default) (sys_id: 6727787553d43210f8bb51a0a0490e67)
        # Set second user as manager of fourth user (INVALID ACTION - second user is inactive, fourth user is active)
    ]
    save_invalid_action(policy, invalid_action_idx, 11)


def generate_external_domain_admin_assignment_violation():
    """Generate trajectories that violate the constraint: Users with email domains outside company domain cannot be assigned admin roles"""
    policy = "Users with email domains outside company domain cannot be assigned admin roles. The company domain is skyfall.ai"
    invalid_action_idx = 8 # 0-indexed
    trajectory = [
        # Create first user with email "ime.vaqif@skyfall.ai" (internal domain) (sys_id: 15aafc3953d43210f8bb51a0a0490e5b)
        # Create second user with email "wolter.detlev@skyfall.ai" (internal domain) (sys_id: fcdaf8b953d43210f8bb51a0a0490e22)
        # Create third user with email "petko.axton@servicenow.com" (external domain) (sys_id: 3ffaf8b953d43210f8bb51a0a0490e2b)
        # Assign admin role to first user
        # Create fourth user with email "cash.gerard@partner.org" (external domain) (sys_id: 82eef43553d43210f8bb51a0a0490e5f)
        # Update second user email to "wolter.detlev@newcompany.com" (now external domain, maybe user left company)
        # Assign web_analytics_viewer role to third user
        # Assign license_viewer role to fourth user
        # Assign license_viewer and admin role to second user (wolter.detlev@newcompany.com) (INVALID ACTION - external domain user getting admin role)
    ]
    save_invalid_action(policy, invalid_action_idx, 12)

async def test_any_mcp_call(tool_name, params):
    agent = WorldModelAgent()
    await agent.initialize_mcp_server("full")

    start_time = datetime.now(pytz.timezone('GMT')).strftime("%Y-%m-%d %H:%M:%S")
    await asyncio.sleep(2)  
    resp = await agent.mcp_server._call_tool_impl(tool_name, params)
    await asyncio.sleep(7)  
    end_time = datetime.now(pytz.timezone('GMT')).strftime("%Y-%m-%d %H:%M:%S")

    audits = get_sys_audit(start_time, end_time)
    ground_truth_state = agent._generate_ground_truth_state(audits, resp[0].text, tool_name)

    return ground_truth_state, resp[0].text

def generate_action_for_trajectory(tool_name, params,trajectory_num):
    """ 
    Save the action, tool response and state changes from effect of action to a file. 
    By running this function with the same trajectory number, you can append to an existing trajectory. 
    """
    ground_truth_state, resp = asyncio.run(test_any_mcp_call(tool_name, params))

    # Convert tool response to json 
    resp = json.loads(resp)

    out_dir = Path(__file__).parent / "qa_data" / f"constraint{trajectory_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_path = out_dir / "trajectory.json" 
    
    # Load existing data or create new list
    if trajectory_path.exists() and trajectory_path.stat().st_size > 0:
        try:
            with open(trajectory_path, "r") as f:
                data = json.load(f)
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            # File exists but is corrupted, start fresh
            data = []
    else:
        data = []
    
    # Append new entry
    data.append({
        "action": {
            "tool_name": tool_name,
            "parameters": params
        },
        "tool_response": resp,
        "ground_truth_state": ground_truth_state.model_dump(mode='json')
    })
    
    # Write back to file
    with open(trajectory_path, "w") as f:
        json.dump(data, f, indent=2)

def generate_masked_trajectories():
    """
    Generate a masked perturbation of trajectory by keeping random records from the trajectory before the ground truth invalid action index. 
    """

    for trajectory_num in tqdm(range(1, 13), desc="Generating masked trajectories"):

        # Load the full trajectory 
        with open(Path(__file__).parent / "qa_data" / f"constraint{trajectory_num}" / "trajectory.json", "r") as f:
            trajectory = json.load(f)

        # Load the ground truth invalid action index
        with open(Path(__file__).parent / "qa_data" / f"constraint{trajectory_num}" / "answer.json", "r") as f:
            file = json.load(f)
        invalid_action_idx = file["invalid_action_idx"] 
        
        # Iteratively keep all records from the trajectory before the ground truth invalid action index
        full_trajectory_before_invalid_action = trajectory[:invalid_action_idx]
        perturbed_trajectories = [] 
        save_dir = Path(__file__).parent / "qa_data" / f"perturbed_trajectory_{trajectory_num}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Pick a random index from filtered trajectory to create a perturbed trajectory
        perturbed_index = random.randint(0, len(full_trajectory_before_invalid_action) - 1)
        perturbed_trajectory = full_trajectory_before_invalid_action[:perturbed_index]
        perturbed_trajectories.append(perturbed_trajectory)

        # Save the perturbed trajectory
        with open(save_dir / f"trajectory.json", "w") as f:
            json.dump(perturbed_trajectory, f, indent=2)

        # Save the answer for the perturbed trajectories
        answer = {
            "policy_name": "No policy violated",
            "invalid_action_idx": -1
        }
        with open(save_dir / "answer.json", "w") as f:
            json.dump(answer, f, indent=2)

def generate_combined_trajectories(): 
    """
    Randomly combines the full trajectory of different constraints into a single trajectory. 
    """

    # Load the full trajectories and invalid action indices of all constraints
    full_trajectories = []
    trajectory_lengths = [] 
    invalid_action_idxs = []
    invalid_policy_names = []
    for constraint_num in range(1, 13):
        with open(Path(__file__).parent / "qa_data" / f"constraint{constraint_num}" / "trajectory.json", "r") as f:
            file = json.load(f)
            trajectory_lengths.append(len(file))
            full_trajectories.append(file)

        with open(Path(__file__).parent / "qa_data" / f"constraint{constraint_num}" / "answer.json", "r") as f:
            file = json.load(f)
            invalid_action_idxs.append(file["invalid_action_idx"])
            invalid_policy_names.append(file["policy_name"])

    # print('Original Invalid action indices:', invalid_action_idxs)
    # print('Original Trajectory lengths:', trajectory_lengths)

    # Randomly combine the full trajectories. Each trajectory is combined with a random number of other trajectories. 
    # Must ensure that the trajectory is not combined with itself. 
    for i, trajectory in tqdm(enumerate(full_trajectories), desc="Generating combined trajectories", total=len(full_trajectories)):
        
        # Create 4 combined trajectories for each constraint
        for t in range(4): 
            num_other_trajectories = random.randint(1, 5) # Minimum 1 other trajectory, maximum 5 other trajectories combined
            
            # Get indices of other trajectories (excluding current one)
            other_indices = [j for j in range(len(full_trajectories)) if j != i]
            selected_indices = random.sample(other_indices, num_other_trajectories)
            
            # Combine current trajectory with selected other trajectories
            combined_trajectory = trajectory.copy() 
            combined_trajectory_lengths = [trajectory_lengths[i]]
            for idx in selected_indices:
                combined_trajectory.extend(full_trajectories[idx]) 
                # Get list of all trajectory lengths for trajectories that are being combined
                combined_trajectory_lengths.append(trajectory_lengths[idx])

            # print('Selected indices:', [i] + selected_indices)
            # print('Combined trajectory lengths:', combined_trajectory_lengths)

            # Update the invalid action indices for the combined trajectory
            answer_idxs = [invalid_action_idxs[i]]
            answer_policy_names = [invalid_policy_names[i]] 
            answer_policy_nums = [i]
            for j in range(1, len(combined_trajectory_lengths)):
                answer_idxs.append(invalid_action_idxs[selected_indices[j-1]] + sum(combined_trajectory_lengths[:j]))

            for idx in selected_indices:
                answer_policy_names.append(invalid_policy_names[idx])
                answer_policy_nums.append(idx)

            # print('Invalid action indices:', answer_idxs)

            # Save the combined trajectories
            save_file = Path(__file__).parent / "qa_data" / f"combined_trajectory_{i+1}_{t+1}" / "trajectory.json"
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(combined_trajectory, f, indent=2)

            # Save the answer for the combined trajectory
            combined_answer = {
                "invalid_action_idxs": answer_idxs,
                "invalid_policy_names": answer_policy_names,
                "invalid_policy_nums": answer_policy_nums
            }

            save_file = Path(__file__).parent / "qa_data" / f"combined_trajectory_{i+1}_{t+1}" / "answer.json"
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w") as f:
                json.dump(combined_answer, f, indent=2) 

if __name__ == "__main__":

    # instance = SNowInstance()
    # # Call sys_choice table for 1 record 
    # params = {
    #     "sysparm_limit": 50, 
    #     "sysparm_query": "name=kb_knowledge^element=workflow_state", 
    #     "sysparm_fields": "name,label,value,element",
    # }
    # test_any_call(instance, table="sys_choice", params=params) 

    # For each constraint, generate a masked perturbation of the trajectory
    generate_masked_trajectories()

    # Randomly combine the full trajectories of all constraints
    generate_combined_trajectories()