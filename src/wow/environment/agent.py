#!/usr/bin/env python3
"""
World Model Agent - State/Action Sequence Generation

Generates sequences of StateDiff (state) and tool calls (actions) for ServiceNow tasks.
One-pass generation where LLM outputs complete sequence, then tools are executed sequentially.
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Literal, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import openai
from servicenow_mcp.server import ServiceNowMCP
from servicenow_mcp.utils.config import ServerConfig
from pathlib import Path
import random
import re
from datetime import datetime
import pytz
from .states import get_sys_audit

class operation(Enum):
    get = "get"
    put = "put"
    delete = "delete"
    post = "post"

class AdditionalInformation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    num_audits: int = Field(..., description="Number of audits")
    num_modified_entries: int = Field(..., description="Number of updated entries")
    num_deleted_entries: int = Field(..., description="Number of deleted entries")
    num_created_entries: int = Field(..., description="Number of created entries")
    operation_type: List[operation] = Field(..., description="Effect of operation, can only choose from ['get', 'put', 'delete', 'post']")
    tables_modified: List[str] = Field(..., description="List of tables that were modified")


class SysAuditRecord(BaseModel):
    fieldname: str = Field(..., description="Name of the field that was changed")
    newvalue: str = Field(..., description="New value after the change")
    tablename: str = Field(..., description="Name of the table where the change occurred")
    oldvalue: str = Field(..., description="Old value before the change")


class StateDiff(BaseModel):
    sysauditrecord: List[SysAuditRecord]
    additional_information: AdditionalInformation


class ActionCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool call")


class StateActionPair(BaseModel):
    predicted_state: StateDiff = Field(..., description="Predicted state changes from this action")
    action: ActionCall = Field(..., description="Tool call to execute")


class StateActionSequence(BaseModel):
    task_description: str = Field(..., description="Description of the task being performed")
    sequence: List[StateActionPair] = Field(..., description="Sequence of state/action pairs")
    final_state_summary: str = Field(..., description="Summary of expected final state")


class WorldModelAgent:
    """Agent that generates state/action sequences for ServiceNow tasks"""

    def __init__(self, model: str = "openai/gpt-4o"):
        self.client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model = model
        self.mcp_server = None
        self.tools = []

    async def initialize_mcp_server(self, tool_package: str = "full"):
        """Initialize ServiceNow MCP server and extract tools"""
        os.environ["MCP_TOOL_PACKAGE"] = tool_package

        config = {
            "instance_url": os.getenv("SNOW_INSTANCE_URL"),
            "auth": {
                "type": "basic",
                "basic": {
                    "username": os.getenv("SNOW_INSTANCE_UNAME"),
                    "password": os.getenv("SNOW_INSTANCE_PWD")
                }
            }
        }

        print(f"🔑 ServiceNow Config: {config['instance_url']}, user: {config['auth']['basic']['username']}")
        server_config = ServerConfig(**config)
        self.mcp_server = ServiceNowMCP(server_config)

        # Get tools from MCP server
        tools_response = await self.mcp_server._list_tools_impl()
        self.tools = tools_response

        print(f"✅ Initialized {len(self.tools)} ServiceNow MCP tools")

    def get_tool_request_types(self, tool_name: str) -> List[str]:
        """
        Get the HTTP request types for a given tool name.

        Args:
            tool_name: Name of the tool to lookup

        Returns:
            List[str]: HTTP request types (e.g., ['get', 'put'])
        """
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "prompts", "tool_request_mapping.json")
            
            with open(json_path, "r") as f:
                mapping = json.load(f)

            return mapping.get(tool_name, ["unknown"])

        except FileNotFoundError:
            print("⚠️ tool_request_mapping.json not found. Run generate_tool_mapping.py first.")
            return ["unknown"]
        except Exception as e:
            print(f"⚠️ Error reading tool mapping: {e}")
            return ["unknown"]

    def _build_state_prediction_prompt(self, action: Dict, previous_states: Optional[List[StateDiff]] = None, task: str = None) -> str:
        """Build prompt for state prediction given task and action"""
        with open(Path(__file__).parent / "prompts" / "state_prediction_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        with open(Path(__file__).parent / "prompts" / "all_table_schemas.json", "r") as f:
            table_schemas = json.load(f)

        with open(Path(__file__).parent / "prompts" / "all_unique_tables.json", "r") as f:
            tables_to_include = json.load(f)

        with open(Path(__file__).parent / "prompts" / "mcp_tool_specifications.json", "r") as f:
            tool_specifications = json.load(f)

        with open(Path(__file__).parent / "prompts" / "tool_request_mapping.json", "r") as f:
            tool_request_mapping = json.load(f)
        
        filtered_schemas = {table: schema for table, schema in table_schemas.items() 
                          if table in tables_to_include}
        
        included_tools = [tool for tool,tool_type  in tool_request_mapping.items() if 'get' not in tool_type or len(tool_type) > 1]
        included_tools_specs = [tool for tool in tool_specifications if tool['name'] in included_tools]

        prompt_template = prompt_template.format(mcp_tools=included_tools_specs, table_schema=json.dumps(filtered_schemas, indent=2))
        
        state_prediction_prompt = prompt_template + f"""

        ## Previous States from earlier to latest
        {previous_states}

        ## Given Action
        Action: {action}

        Predict the resulting state change as a JSON array. 

        Output ONLY the JSON array without code blocks or comments."""
        
        return state_prediction_prompt

    def _build_custom_state_prediction_prompt(self, action: Dict, task: str, custom_schema_path: str, previous_states: Optional[List[StateDiff]] = None) -> str:
        """Build prompt for state prediction using task-specific custom schemas"""
        with open(Path(__file__).parent / "prompts" / "state_prediction_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        # Load custom schema for the specific task
        task_schema_file = Path(custom_schema_path) / f"{task.lower()}.json"
        
        if not task_schema_file.exists():
            raise FileNotFoundError(f"Custom schema file not found: {task_schema_file}")
        
        with open(task_schema_file, "r") as f:
            task_schema_data = json.load(f)
        
        # Extract tool_calls from the custom schema
        tool_calls = task_schema_data.get("tool_calls", [])
        
        if not tool_calls:
            raise ValueError(f"No tool_calls found in custom schema: {task_schema_file}")
        
        # Convert tool_calls to MCP tool specifications format
        included_tools_specs = []
        for tool_call in tool_calls:
            tool_spec = {
                "name": tool_call.get("name", ""),
                "description": f"Tool call from {task} task",
                "parameters": {}
            }
            
            # Parse input parameters if available
            input_str = tool_call.get("input", "{}")
            try:
                if isinstance(input_str, str):
                    tool_spec["parameters"] = json.loads(input_str)
                else:
                    tool_spec["parameters"] = input_str
            except json.JSONDecodeError:
                tool_spec["parameters"] = {}
            
            included_tools_specs.append(tool_spec)
        
        # For table schemas, we'll use a minimal set since we're focusing on tool_calls
        # You could extend this to extract table schemas from the custom schema if needed
        minimal_table_schemas = {
            "incident": {
                "short_description": {"element": "short_description", "mandatory": True, "internal_type": "string"},
                "description": {"element": "description", "mandatory": False, "internal_type": "string"},
                "state": {"element": "state", "mandatory": True, "internal_type": "string"},
                "priority": {"element": "priority", "mandatory": True, "internal_type": "string"}
            }
        }
        
        prompt_template = prompt_template.format(
            mcp_tools=json.dumps(included_tools_specs, indent=2), 
            table_schema=json.dumps(minimal_table_schemas, indent=2)
        )
        
        state_prediction_prompt = prompt_template + f"""

        ## Previous States from earlier to latest
        {previous_states}

        ## Given Action
        Action: {action}

        ## Task-Specific Context
        This prediction is for the '{task}' task using custom schema with {len(tool_calls)} tool calls.

        Predict the resulting state change as a JSON array. 

        Output ONLY the JSON array without code blocks or comments."""
        
        return state_prediction_prompt

    def _build_action_prediction_prompt(self, state_diffs: List[Dict], task: str = None) -> str:
        """Build prompt for multi-step action prediction given task and k state diffs"""
        with open(Path(__file__).parent / "prompts" / "action_prediction_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        with open(Path(__file__).parent / "prompts" / "all_table_schemas.json", "r") as f:
            table_schemas = json.load(f)

        with open(Path(__file__).parent / "prompts" / "all_unique_tables.json", "r") as f:
            tables_to_include = json.load(f)

        with open(Path(__file__).parent / "prompts" / "mcp_tool_specifications.json", "r") as f:
            tool_specifications = json.load(f)

        with open(Path(__file__).parent / "prompts" / "tool_request_mapping.json", "r") as f:
            tool_request_mapping = json.load(f)
        
        filtered_schemas = {table: schema for table, schema in table_schemas.items() 
                          if table in tables_to_include}
        
        included_tools = [tool for tool,tool_type  in tool_request_mapping.items() if 'get' not in tool_type or len(tool_type) > 1]
        included_tools_specs = [tool for tool in tool_specifications if tool['name'] in included_tools]

        prompt_template = prompt_template.format(mcp_tools=included_tools_specs, table_schema=json.dumps(filtered_schemas, indent=2))
        
        # Format state diffs for the prompt
        state_diffs_str = ""
        for i, state_diff in enumerate(state_diffs):
            state_diffs_str += f"Step {i}: {json.dumps(state_diff, indent=2)}\n"
        
        action_prediction_prompt = prompt_template + f"""
        Given these {len(state_diffs)} sequential state changes that should occur:
        {state_diffs_str}

        Predict the {len(state_diffs)} actions that would lead to these state changes as a JSON array:
        [Action1, Action2, ..., Action{len(state_diffs)}]

        Each Action should follow this schema:
        {{
        "tool_name": "name_of_tool",
        "parameters": {{"param1": "value1", "param2": "value2"}}
        }}

        Output ONLY the JSON array without code blocks or comments."""
        
        return action_prediction_prompt

    def _build_custom_action_prediction_prompt(self, state_diffs: List[Dict], task: str, custom_schema_path: str) -> str:
        """Build prompt for action prediction using task-specific custom schemas"""
        with open(Path(__file__).parent / "prompts" / "action_prediction_prompt.txt", "r") as f:
            prompt_template = f.read()
        
        # Load custom schema for the specific task
        task_schema_file = Path(custom_schema_path) / f"{task.lower()}.json"
        
        if not task_schema_file.exists():
            raise FileNotFoundError(f"Custom schema file not found: {task_schema_file}")
        
        with open(task_schema_file, "r") as f:
            task_schema_data = json.load(f)
        
        # Extract tool_calls from the custom schema
        tool_calls = task_schema_data.get("tool_calls", [])
        
        if not tool_calls:
            raise ValueError(f"No tool_calls found in custom schema: {task_schema_file}")
        
        # Convert tool_calls to MCP tool specifications format
        included_tools_specs = []
        for tool_call in tool_calls:
            tool_spec = {
                "name": tool_call.get("name", ""),
                "description": f"Tool call from {task} task",
                "parameters": {}
            }
            
            # Parse input parameters if available
            input_str = tool_call.get("input", "{}")
            try:
                if isinstance(input_str, str):
                    tool_spec["parameters"] = json.loads(input_str)
                else:
                    tool_spec["parameters"] = input_str
            except json.JSONDecodeError:
                tool_spec["parameters"] = {}
            
            included_tools_specs.append(tool_spec)
        
        # For table schemas, we'll use a minimal set since we're focusing on tool_calls
        # You could extend this to extract table schemas from the custom schema if needed
        minimal_table_schemas = {
            "incident": {
                "short_description": {"element": "short_description", "mandatory": True, "internal_type": "string"},
                "description": {"element": "description", "mandatory": False, "internal_type": "string"},
                "state": {"element": "state", "mandatory": True, "internal_type": "string"},
                "priority": {"element": "priority", "mandatory": True, "internal_type": "string"}
            }
        }
        
        prompt_template = prompt_template.format(
            mcp_tools=json.dumps(included_tools_specs, indent=2), 
            table_schema=json.dumps(minimal_table_schemas, indent=2)
        )
        
        # Format state diffs for the prompt
        state_diffs_str = ""
        for i, state_diff in enumerate(state_diffs):
            state_diffs_str += f"Step {i}: {json.dumps(state_diff, indent=2)}\n"
        
        action_prediction_prompt = prompt_template + f"""
        Given these {len(state_diffs)} sequential state changes that should occur:
        {state_diffs_str}

        ## Task-Specific Context
        This prediction is for the '{task}' task using custom schema with {len(tool_calls)} tool calls.

        Predict the {len(state_diffs)} actions that would lead to these state changes as a JSON array:
        [Action1, Action2, ..., Action{len(state_diffs)}]

        Each Action should follow this schema:
        {{
        "tool_name": "name_of_tool",
        "parameters": {{"param1": "value1", "param2": "value2"}}
        }}

        Output ONLY the JSON array without code blocks or comments."""
        
        return action_prediction_prompt

    def _build_constraint_violation_prompt(self, sequence: List[Dict], policies: List[str], 
    mode: Literal["action_only", "state_action", "state_only"], perfect_schema: bool = True) -> str:
        """
        Build prompt for constraint violation prediction given sequence of actions. The sequece may or may not include the state changes after each action.

        Args:
            sequence: List[Dict] - The sequence of actions and their corresponding state changes.
            policies: List[str] - The list of policies that are being tracked.
            mode: Literal["action_only", "state_action"] - The mode to use for the constraint violation prediction. If "action_only", only the action will be used to predict the constraint violation. If "state_action", the state and action will be used to predict the constraint violation.
            perfect_schema: bool - Whether to use the perfect schema for the constraint violation prediction.
        """
        # Read prompt template 
        with open(Path(__file__).parent / "prompts" / "constraint_violation_prompt.txt", "r") as f:
            prompt_template = f.read()

        # Read table schemas 
        with open(Path(__file__).parent / "prompts" / "all_table_schemas.json", "r") as f:
            table_schemas = json.load(f)

        # Get all the tables that were touched in the trajectory. 
        trajectory_tables = []
        for action in sequence:
            trajectory_tables.extend(action["ground_truth_state"]["additional_information"]["tables_modified"])
        trajectory_tables = list(set(trajectory_tables))

        if perfect_schema:
            tables_to_include = trajectory_tables
        else:
            # Add some random number of tables to the trajectory_tables list 
            num_random_tables = 20
            random_tables = random.sample(list(set(table_schemas.keys()) - set(trajectory_tables)), num_random_tables)
            tables_to_include = trajectory_tables + random_tables
        
        # Filter table schemas to only include tables that are in the tables_to_include list # TODO: Table list must be updated
        filtered_schemas = {table: schema for table, schema in table_schemas.items() 
                          if table in tables_to_include} 

        # Read tool specifications for descriptions and parameters for each tool 
        with open(Path(__file__).parent / "prompts" / "tool_specifications.json", "r") as f:
            tool_specifications = json.load(f) 

        tools_desc = []
        # Only include tool descriptions for tools that are in the actions in the sequence first
        trajectory_tool_names = [action["action"]["tool_name"] for action in sequence] 
        for tool in tool_specifications["tool_specifications"]:
            if tool["name"] in trajectory_tool_names:
                tools_desc.append(tool)
        
        if not perfect_schema:
            # Add some random number of tools to the tools_desc list 
            num_random_tools = 10
            num_tools_added = 0
            random.shuffle(tool_specifications["tool_specifications"])
            for tool in tool_specifications["tool_specifications"]:
                if tool["name"] not in trajectory_tool_names:
                    tools_desc.append(tool)
                    num_tools_added += 1
                    if num_tools_added >= num_random_tools:
                        break

        # Add index to each action in the sequence. Add via zip. 
        # sequence_with_index = [{"action_index": i, **action} for i, action in enumerate(sequence)] 
        sequence_with_index = [{"action_index": i, **action} for i, action in enumerate(sequence)] 

        # If mode = "action_only", only retain the action and action_index fields
        if mode == "action_only":
            sequence_with_index = [{"action_index": action["action_index"], "action": action["action"], "tool_response": action["tool_response"]} for action in sequence_with_index]
        elif mode == "state_only": 
            sequence_with_index = [{"action_index": action["action_index"], "ground_truth_state": action["ground_truth_state"]} for action in sequence_with_index]

        # Add index to each policy 
        policies_with_index = [{"policy_index": i, "policy": policy} for i, policy in enumerate(policies)]
        
        return prompt_template.format(
            sequence=json.dumps(sequence_with_index, indent=2),
            table_schema=json.dumps(filtered_schemas, indent=2),
            mcp_tools=json.dumps(tools_desc, indent=2), 
            policies=json.dumps(policies_with_index, indent=2)
        )

    async def predict_states(self, actions: List[Dict], task: str = None) -> List[StateDiff]:
        """
        Predict state changes for k sequential actions. 
        Uses LLM sequentially to predict state changes for each action. Uses the predicted state from previous action to predict
        the state for the next action.
        
        """ 
        predicted_states = []
        for action in actions:
            messages = [
                {"role": "user", "content": self._build_state_prediction_prompt(action, predicted_states, task)}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )

            result_text = response.choices[0].message.content 
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            json_str = json_match.group() if json_match else result_text
            predicted_states.append(StateDiff(**json.loads(json_str)))

        return predicted_states

    async def predict_states_custom(self, actions: List[Dict], task: str, custom_schema_path: str) -> List[StateDiff]:
        """
        Predict state changes for k sequential actions using custom task-specific schemas.
        Uses LLM sequentially to predict state changes for each action with custom schema context.
        
        Args:
            actions: List of action dictionaries
            task: Task name (used to find corresponding schema file)
            custom_schema_path: Path to directory containing custom schema JSON files
            
        Returns:
            List of predicted StateDiff objects
        """ 
        predicted_states = []
        for action in actions:
            messages = [
                {"role": "user", "content": self._build_custom_state_prediction_prompt(action, task, custom_schema_path, predicted_states)}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )

            result_text = response.choices[0].message.content 
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            json_str = json_match.group() if json_match else result_text
            predicted_states.append(StateDiff(**json.loads(json_str)))

        return predicted_states

    async def predict_actions(self, state_diffs: List[Dict], task: str = None) -> List[ActionCall]:
        """Predict k actions that would lead to k given state diffs"""
        messages = [
            {"role": "user", "content": self._build_action_prediction_prompt(state_diffs, task)}
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        json_str = json_match.group() if json_match else result_text
        
        parsed_data = json.loads(json_str)
        return [ActionCall(**action) for action in parsed_data]

    async def predict_actions_custom(self, state_diffs: List[Dict], task: str, custom_schema_path: str) -> List[ActionCall]:
        """Predict k actions that would lead to k given state diffs using custom task-specific schemas"""
        messages = [
            {"role": "user", "content": self._build_custom_action_prediction_prompt(state_diffs, task, custom_schema_path)}
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        json_str = json_match.group() if json_match else result_text
        
        parsed_data = json.loads(json_str)
        return [ActionCall(**action) for action in parsed_data]

    async def predict_state(self, action: List[Dict], task: str = None) -> StateDiff:
        """Predict state changes for a single action"""
        states = await self.predict_states(task, [action])
        return states[0]

    async def predict_action(self, state_diff: List[Dict], task: str = None) -> ActionCall:
        """Predict single action that would lead to given state diff"""
        actions = await self.predict_actions(task, state_diff)
        return actions[0]

    async def predict_constraint_violation(self, sequence: List[Dict], policies: List[str], 
    mode: Literal["action_only", "state_action", "state_only"], perfect_schema: bool = True) -> Dict:
        """
        Predict constraint violation for a given sequence of actions and policies
        Args: 
            sequence: List[Dict] - The sequence of actions and their corresponding state changes.
            policies: List[str] - The list of policies that are being tracked.
            mode: Literal["action_only", "state_action"] - The mode to use for the constraint violation prediction. If "action_only", only the action will be used to predict the constraint violation. If "state_action", the state and action will be used to predict the constraint violation.
            perfect_schema: bool - Whether to use the perfect schema for the constraint violation prediction.
        
        """
        messages = [
            {"role": "user", "content": self._build_constraint_violation_prompt(sequence, policies, mode, perfect_schema)}
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body = { # Allows for tracking of precise token usage and costs for each model call
                "usage" : { 
                    "include": True
                }
            }
        )
        
        result_text = response.choices[0].message.content 
        usage = response.usage
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        json_str = json_match.group() if json_match else result_text
        
        try: 
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError:
            return None, usage
        
        return parsed_data, usage

    def _generate_ground_truth_state(self, audits: List[Dict], tool_response: str, tool_name: str) -> StateDiff:
        """Generate ground truth StateDiff from actual audits and tool response"""
        import json

        # Convert audits to SysAuditRecord format
        sysauditrecords = []
        for audit in audits:
            sysauditrecords.append(SysAuditRecord(
                fieldname=audit.get('fieldname', ''),
                newvalue=audit.get('newvalue', ''),
                tablename=audit.get('tablename', ''),
                oldvalue=audit.get('oldvalue', '')
            ))

        # Determine operation type using tool_request_mapping.json
        request_types = self.get_tool_request_types(tool_name)
        operation_types = []
        for req_type in request_types:
            if req_type == 'post':
                operation_types.append(operation.post)
            elif req_type == 'put':
                operation_types.append(operation.put)
            elif req_type == 'delete':
                operation_types.append(operation.delete)
            elif req_type == 'get':
                operation_types.append(operation.get)

        # Count different types of changes
        num_created = len([a for a in audits if a.get('oldvalue', '') == '' and a.get('newvalue', '') != ''])
        num_modified = len([a for a in audits if a.get('oldvalue', '') != '' and a.get('newvalue', '') != ''])
        num_deleted = len([a for a in audits if a.get('newvalue', '') == 'DELETED' and a.get('oldvalue', '') != ''])

        # Get unique tables modified
        tables_modified = list(set([audit.get('tablename', '') for audit in audits if audit.get('tablename')]))

        additional_info = AdditionalInformation(
            num_audits=len(audits),
            num_modified_entries=num_modified,
            num_deleted_entries=num_deleted,
            num_created_entries=num_created,
            operation_type=operation_types,
            tables_modified=tables_modified
        )

        return StateDiff(
            sysauditrecord=sysauditrecords,
            additional_information=additional_info
        )

    async def run_mcp_action(self,tool_name: str, tool_params: Dict[str, Any]) -> Tuple[StateDiff, str]: 
        """
        Run an MCP action and return the state diff and tool response
        Args:
            tool_name: Name of the tool to run
            tool_params: Parameters for the tool

        Returns:
            Tuple[StateDiff, str]: Tuple containing the ground truth state diff and the tool response
        """

        start_time = datetime.now(pytz.timezone('GMT')).strftime("%Y-%m-%d %H:%M:%S")
        await asyncio.sleep(2)  
        resp = await self.mcp_server._call_tool_impl(tool_name, tool_params)
        await asyncio.sleep(10)  
        end_time = datetime.now(pytz.timezone('GMT')).strftime("%Y-%m-%d %H:%M:%S") 

        audits = get_sys_audit(start_time, end_time)
        ground_truth_state = self._generate_ground_truth_state(audits, resp[0].text, tool_name)

        return ground_truth_state, resp[0].text