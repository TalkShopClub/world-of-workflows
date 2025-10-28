import os
import asyncio
from pathlib import Path
import logging
from io import StringIO

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse


class LLMResult(BaseModel):
    llm_result: str = Field(..., description="The LLM result. The result is allowed to have tags to specify answers if needed. This is not mandatory and the LLM is allowed to return any plain text to reply to the query. ")
    mcp_reasoning: List[str] = Field(..., description="The reasoning behind calling each MCP tool. Each entry in the list should be a string that describes the reasoning behind calling the corresponding MCP tool.")
    mcp_functions_called: List[str] = Field(..., description="The list of MCP function names called by the LLM")
    mcp_function_params: List[str] = Field(..., description="The parameters passed to each MCP function. Each entry in the list should be a dictionary of the parameters passed to the corresponding MCP function.")


class IntermediateOutput(BaseModel):
    """Model for storing intermediate tool calls and responses"""
    step_number: int = Field(..., description="The step number in the agent execution")
    tool_name: str = Field(..., description="Name of the tool/function called")
    tool_input: Dict[str, Any] = Field(..., description="Input parameters passed to the tool")
    tool_output: Any = Field(..., description="Output returned by the tool")
    timestamp: str = Field(..., description="Timestamp of the tool call")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the execution")


async def solve_llm_with_tracing(
    task_query: str,
    llm,
    trace_name: Optional[str] = None,
    save_intermediate_outputs: bool = True,
    langfuse_session_id: Optional[str] = None
) -> tuple[Any, List[IntermediateOutput]]:
    """
    Execute LLM task with Langfuse tracing and capture intermediate outputs.

    Args:
        task_query: The task query to send to the LLM
        llm: The language model instance
        trace_name: Optional name for the Langfuse trace
        save_intermediate_outputs: Whether to capture and return intermediate outputs
        langfuse_session_id: Optional session ID for grouping related traces

    Returns:
        Tuple of (final_result, intermediate_outputs_list)
    """
    # Look for mcp_config.json in the mcp_local folder
    project_root = Path(__file__).parent.parent.parent
    config_file_path = project_root / "mcp_local" / "mcp_config.json"

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"❌ MCP config file not found at: {config_file_path}")

    # Initialize Langfuse
    langfuse_client = Langfuse()

    # Create a span with custom name (newer Langfuse API uses spans instead of traces)
    try:
        trace = langfuse_client.start_span(
            name=trace_name or f"mcp_agent_{task_query[:50]}",
            metadata={
                "session_id": langfuse_session_id,
                "tags": ["mcp-agent", "schema-retrieval"]
            }
        )
    except Exception as e:
        print(f"⚠️ Langfuse trace creation failed: {e}. Continuing without tracing.")
        trace = None

    # List to store intermediate outputs
    intermediate_outputs = []

    try:
        # Create MCPClient from configuration file
        client = MCPClient.from_config_file(str(config_file_path))

        # Create agent with Langfuse callback
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=15,
        )

        print(f"🤖 Sending task to LLM with tracing enabled...")

        result = await agent.run(task_query, output_schema=LLMResult)

        # Finalize the trace
        if trace:
            trace.update(
                output=result.model_dump() if hasattr(result, 'model_dump') else str(result),
                metadata={
                    "total_steps": len(intermediate_outputs),
                    "task_query": task_query
                }
            )
            trace.end()

        # Flush Langfuse to ensure all data is sent
        langfuse_client.flush()

        await asyncio.sleep(2)
    
        # Retrieve traces
        traces = langfuse_client.api.trace.list(limit=10)

        if not traces.data:
            print("\nNo traces found in Langfuse")
            return result, []
        
        latest_trace = traces.data[0]
        trace_id = latest_trace.id
            
        # Get all TOOL type observations for this trace
        tool_observations = langfuse_client.api.observations.get_many(
            trace_id=trace_id,
            type="TOOL",
            limit=100
        )
        
        print(f"\nFound {len(tool_observations.data)} tool calls")
        
        if not tool_observations.data:
            print("\nNo tool calls found in this trace.")
            return result, []
        
        # Extract and display tool calls
        tool_calls = []
        for obs in tool_observations.data:
            # Convert datetime objects to ISO format strings for JSON serialization
            start_time, end_time = None, None
            if hasattr(obs, 'start_time') and obs.start_time:
                start_time = obs.start_time.isoformat() if hasattr(obs.start_time, 'isoformat') else str(obs.start_time)
            if hasattr(obs, 'end_time') and obs.end_time:
                end_time = obs.end_time.isoformat() if hasattr(obs.end_time, 'isoformat') else str(obs.end_time)

            tool_info = {
                'observation_id': obs.id if hasattr(obs, 'id') else None,
                'name': obs.name if hasattr(obs, 'name') else 'Unknown',
                'type': obs.type if hasattr(obs, 'type') else 'TOOL',
                'input': obs.input if hasattr(obs, 'input') else None,
                'output': obs.output if hasattr(obs, 'output') else None,
                'start_time': start_time,
                'end_time': end_time,
            }
            tool_calls.append(tool_info)

            if tool_info['input']:
                try:
                    input_str = json.dumps(tool_info['input'], indent=6)
                    print(f"   {input_str}")
                except:
                    print(f"   {tool_info['input']}")
            else:
                print("   No input captured")

            if tool_info['output']:
                try:
                    output_str = json.dumps(tool_info['output'], indent=6)
                    if len(output_str) > 2000:
                        print(f"   {output_str[:2000]}...")
                        print(f"   [Output truncated - {len(output_str)} total characters]")
                    else:
                        print(f"   {output_str}")
                except:
                    output_str = str(tool_info['output'])
                    if len(output_str) > 2000:
                        print(f"   {output_str[:2000]}...")
                        print(f"   [Output truncated - {len(output_str)} total characters]")
                    else:
                        print(f"   {output_str}")
            else:
                print("   No output captured")

            print(f"\n   {'-'*90}")

        print(f"\n✅ Task completed with {len(tool_calls)} tool calls traced from Langfuse")

        # Return result and tool_calls
        return result, tool_calls


    except Exception as e:
        print(f"❌ LLM execution failed: {e}")
        import traceback
        traceback.print_exc()

        # Log error to Langfuse
        if trace:
            trace.update(
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
            trace.end()
        langfuse_client.flush()

        return None, intermediate_outputs


async def solve_llm(task_query: str, llm) -> str:
    """
    Original function maintained for backward compatibility.
    For new implementations, use solve_llm_with_tracing() instead.
    """
    os.environ["MCP_TOOL_PACKAGE"] = 'get_request_tools'
    # Look for mcp_config.json in the mcp_local folder
    project_root = Path(__file__).parent.parent.parent
    config_file_path = project_root / "mcp_local" / "mcp_config.json"

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"❌ MCP config file not found at: {config_file_path}")

    try:
        # Create MCPClient from configuration file
        client = MCPClient.from_config_file(str(config_file_path))
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=15,
        )

        print(f"🤖 Sending task to LLM...")

        result = await agent.run(task_query, output_schema=LLMResult)

    except Exception as e:
        print(f"❌ LLM execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


    return result

def save_tool_calls_to_file(
    tool_calls: List[Dict[str, Any]],
    output_file: str = "tool_calls.json"
) -> None:
    """
    Save tool calls retrieved from Langfuse to a JSON file.

    Args:
        tool_calls: List of tool call dictionaries from Langfuse
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        json.dump(tool_calls, f, indent=2)

    print(f"💾 Saved {len(tool_calls)} tool calls to {output_file}")


async def agentic_schema_pipeline(
    trajectory: List[Dict],
    evaluation_mode: str = "both",  # "state", "constraint", "both"
    policies: Optional[List[str]] = None,
    model: str = "gpt-4o",
    trace_name: Optional[str] = None,
    langfuse_session_id: Optional[str] = None
) -> Dict:
    """
    Agentic schema pipeline that dynamically fetches schemas and uses them for evaluation.

    Args:
        trajectory: List of action dictionaries with format:
                   [{"action": {"tool_name": "...", "parameters": {...}},
                     "tool_response": ...,
                     "ground_truth_state": ...}, ...]
        evaluation_mode: "state", "constraint", or "both"
        policies: List of policy strings (required for constraint mode)
        model: OpenAI model to use
        trace_name: Custom trace name for Langfuse
        langfuse_session_id: Session ID for grouping traces

    Returns:
        Dictionary with evaluation results and fetched schemas
    """
    from datetime import datetime


    schema_query = _build_schema_fetching_query(trajectory)

    # Step 2: Fetch schemas using MCP agent with OpenRouter

    # OpenRouter requires "openai/" prefix for OpenAI models
    openrouter_model = f"openai/{model}" if not model.startswith("openai/") else model
    llm = ChatOpenAI(
        model=openrouter_model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Agentic Schema Pipeline"
            }
        }
    )
    result, tool_calls = await solve_llm_with_tracing(
        task_query=schema_query,
        llm=llm,
        trace_name=trace_name or f"schema_fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_intermediate_outputs=True,
        langfuse_session_id=langfuse_session_id
    )

    print(f"✅ Retrieved {len(tool_calls)} tool calls from Langfuse")


    schema_context = _extract_schema_context(tool_calls)
    print(f"✅ Extracted schema context ({len(schema_context)} characters)")

    # Step 4: Run evaluations based on mode
    results = {
        "trajectory_length": len(trajectory),
        "tool_calls_made": len(tool_calls),
        "schema_context_size": len(schema_context),
        "evaluation_mode": evaluation_mode
    }

    if evaluation_mode in ["state", "both"]:
        print("\n" + "="*80)
        print("STEP 4A: State Prediction Evaluation")
        print("="*80)
        state_results = await _run_state_prediction(trajectory, schema_context, model)
        results["state_prediction"] = state_results
        print(f"✅ State prediction completed for {len(state_results['predictions'])} actions")

    if evaluation_mode in ["constraint", "both"]:
        print("\n" + "="*80)
        print(f"STEP 4{'B' if evaluation_mode == 'both' else ''}: Constraint Understanding Evaluation")
        print("="*80)

        if not policies:
            print("⚠️ No policies provided, skipping constraint evaluation")
        else:
            constraint_results = await _run_constraint_understanding(
                trajectory, schema_context, policies, model
            )
            results["constraint_understanding"] = constraint_results
            print(f"✅ Constraint understanding completed")

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED")
    print("="*80)

    return results


def _build_schema_fetching_query(trajectory: List[Dict]) -> str:
    """Build schema fetching query from trajectory"""
    # Read schema fetching prompt template
    with open(Path(__file__).parent / "prompts" / "schema_fetching_prompt.txt", "r") as f:
        prompt_template = f.read()

    # Read all unique tables
    with open(Path(__file__).parent / "prompts" / "all_unique_tables.json", "r") as f:
        all_tables = json.load(f)

    # Extract actions from trajectory
    actions = []
    for item in trajectory:
        if "action" in item:
            actions.append(item["action"])

    # Format the prompt
    query = prompt_template.format(
        all_tables=json.dumps(all_tables, indent=2),
        actions=json.dumps(actions, indent=2)
    )

    return query


def _extract_schema_context(tool_calls: List[Dict[str, Any]]) -> str:
    """Extract and format schema context from tool responses"""
    schema_context = "## Retrieved Schema Information\n\n"
    schema_context += "The following information was retrieved by calling MCP tools:\n\n"

    for idx, tool_call in enumerate(tool_calls, 1):
        tool_name = tool_call.get("name", "unknown")
        tool_input = tool_call.get("input", {})
        tool_output = tool_call.get("output", {})

        schema_context += f"### Tool Call {idx}: {tool_name}\n"
        schema_context += f"**Input:** {json.dumps(tool_input, indent=2)}\n\n"
        schema_context += f"**Output:**\n```json\n{json.dumps(tool_output, indent=2)}\n```\n\n"

    return schema_context


async def _run_state_prediction(
    trajectory: List[Dict],
    schema_context: str,
    model: str
) -> Dict:
    """Run state prediction evaluation with fetched schema context"""
    from langchain_openai import ChatOpenAI
    import openai

    # Initialize OpenAI client for state prediction
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    # Read state prediction prompt template (without schema)
    with open(Path(__file__).parent / "prompts" / "state_prediction_prompt.txt", "r") as f:
        prompt_template = f.read()

    # Read MCP tools
    with open(Path(__file__).parent / "prompts" / "mcp_tool_specifications.json", "r") as f:
        tool_specifications = json.load(f)

    # Remove the {table_schema} placeholder and add our fetched schema context
    # The prompt template has {table_schema} and {mcp_tools} placeholders
    prompt_base = prompt_template.replace("{table_schema}", schema_context)
    prompt_base = prompt_base.replace("{mcp_tools}", json.dumps(tool_specifications, indent=2))

    predictions = []
    previous_states = []

    for idx, item in enumerate(trajectory):
        action = item["action"]

        # Build prompt for this action
        state_prediction_prompt = prompt_base + f"""

        ## Previous States from earlier to latest
        {json.dumps([s.dict() if hasattr(s, 'dict') else s for s in previous_states], indent=2)}

        ## Given Action
        Action: {json.dumps(action, indent=2)}

        Predict the resulting state change as a JSON object following the StateDiff schema.

        Output ONLY the JSON object without code blocks or comments."""

        messages = [{"role": "user", "content": state_prediction_prompt}]

        response = await client.chat.completions.create(
            model=f"openai/{model}",
            messages=messages,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content

        # Parse the JSON response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        json_str = json_match.group() if json_match else result_text

        try:
            predicted_state = json.loads(json_str)
            previous_states.append(predicted_state)
            predictions.append({
                "action_index": idx,
                "action": action,
                "predicted_state": predicted_state
            })
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse state prediction for action {idx}: {e}")
            predictions.append({
                "action_index": idx,
                "action": action,
                "error": str(e),
                "raw_response": result_text[:500]
            })

    return {
        "predictions": predictions,
        "total_actions": len(trajectory)
    }


async def _run_constraint_understanding(
    trajectory: List[Dict],
    schema_context: str,
    policies: List[str],
    model: str
) -> Dict:
    """Run constraint understanding evaluation with fetched schema context"""
    import openai

    # Initialize OpenAI client
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    # Read constraint violation prompt template (without schema)
    with open(Path(__file__).parent / "prompts" / "constraint_violation_prompt.txt", "r") as f:
        prompt_template = f.read()

    # Read MCP tools
    with open(Path(__file__).parent / "prompts" / "tool_specifications.json", "r") as f:
        tool_specs_data = json.load(f)
        tool_specifications = tool_specs_data.get("tool_specifications", [])

    # Format trajectory with indices
    sequence_with_index = [{"action_index": i, **action} for i, action in enumerate(trajectory)]

    # Format policies with indices
    policies_with_index = [{"policy_index": i, "policy": policy} for i, policy in enumerate(policies)]

    # Replace placeholders with fetched schema context and format
    constraint_prompt = prompt_template.replace("{table_schema}", schema_context)
    constraint_prompt = constraint_prompt.replace("{mcp_tools}", json.dumps(tool_specifications, indent=2))
    constraint_prompt = constraint_prompt.replace("{policies}", json.dumps(policies_with_index, indent=2))
    constraint_prompt = constraint_prompt.replace("{sequence}", json.dumps(sequence_with_index, indent=2))

    messages = [{"role": "user", "content": constraint_prompt}]

    response = await client.chat.completions.create(
        model=f"openai/{model}",
        messages=messages,
        temperature=0.0,
    )

    result_text = response.choices[0].message.content

    # Parse the JSON response
    import re
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    json_str = json_match.group() if json_match else result_text

    try:
        constraint_result = json.loads(json_str)
        return {
            "violated_policy_idxs": constraint_result.get("violated_policy_idxs", []),
            "invalid_action_idxs": constraint_result.get("invalid_action_idxs", []),
            "reason": constraint_result.get("reason", ""),
            "total_policies": len(policies),
            "total_actions": len(trajectory)
        }
    except json.JSONDecodeError as e:
        return {
            "error": str(e),
            "raw_response": result_text[:500]
        }


def main():
    """
    Example usage of the agentic schema pipeline.
    To use Langfuse, set the following environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)
    """

    # Example trajectory (format from generate_constraint_violation_trajectories.py)
    example_trajectory = [
        {
            "action": {
                "tool_name": "create_incident",
                "parameters": {
                    "short_description": "Test incident",
                    "priority": "1"
                }
            },
            "tool_response": {"sys_id": "test123"},
            "ground_truth_state": {
                "sysauditrecord": [],
                "additional_information": {
                    "num_audits": 0,
                    "num_modified_entries": 0,
                    "num_deleted_entries": 0,
                    "num_created_entries": 1,
                    "operation_type": ["post"],
                    "tables_modified": ["incident"]
                }
            }
        },
        {
            "action": {
                "tool_name": "update_incident",
                "parameters": {
                    "incident_id": "test123",
                    "assigned_to": "user456"
                }
            },
            "tool_response": {"success": True},
            "ground_truth_state": {
                "sysauditrecord": [],
                "additional_information": {
                    "num_audits": 1,
                    "num_modified_entries": 1,
                    "num_deleted_entries": 0,
                    "num_created_entries": 0,
                    "operation_type": ["put"],
                    "tables_modified": ["incident"]
                }
            }
        }
    ]

    example_policies = [
        "Incidents with priority 1 cannot be assigned to users who already have more than 3 priority 1 incidents",
        "Editing the assigned user for priority 1 incident ticket is not allowed unless there is no user assigned"
    ]

    # Run pipeline
    results = asyncio.run(
        agentic_schema_pipeline(
            trajectory=example_trajectory,
            evaluation_mode="both",
            policies=example_policies,
            model="gpt-4o",
            trace_name="example_pipeline_run",
            langfuse_session_id="example_session"
        )
    )

    print("\n" + "="*80)
    print("PIPELINE RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()  


