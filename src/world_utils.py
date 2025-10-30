#!/usr/bin/env python3
"""
World Model Utilities

Collection of utilities for generating schemas, tool mappings, and specifications
for the World of Workflows project.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from langchain_openai import ChatOpenAI

# Import for trajectory schema generation
from .agentic_schema_pipeline import (
    _build_schema_fetching_query,
    solve_llm_with_tracing,
    _extract_schema_context
)

# Import for table schema generation
from .states import get_all_tables
from .wow.instance import SNowInstance
from .wow.api.utils import table_api_call


# ============================================================================
# TRAJECTORY SCHEMA GENERATOR
# ============================================================================

class TrajectorySchemaGenerator:
    """Generate contextual, trajectory-specific schemas using an LLM agent"""

    def __init__(self):
        self.base_dir = Path(__file__).parent

    async def generate_schema_for_trajectory(
        self,
        trajectory_file: Path,
        output_dir: Path,
        model: str = "openai/gpt-4o"
    ) -> dict:
        """
        Generate schema for a single trajectory file.

        Args:
            trajectory_file: Path to trajectory JSON file
            output_dir: Directory to save schema output
            model: OpenAI model to use for schema fetching

        Returns:
            Dictionary with schema generation results
        """
        print(f"\n{'='*80}")
        print(f"Processing: {trajectory_file.name}")
        print(f"{'='*80}")

        try:
            # Load trajectory
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)

            print(f"Loaded {len(trajectory)} actions")

            # Build schema fetching query
            schema_query = _build_schema_fetching_query(trajectory)
            print(f"Built schema query ({len(schema_query)} characters)")

            # Fetch schemas using MCP agent with OpenRouter
            llm = ChatOpenAI(
                model=model,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://github.com/your-repo",
                        "X-Title": "Schema Generation"
                    }
                }
            )
            result, tool_calls = await solve_llm_with_tracing(
                task_query=schema_query,
                llm=llm,
                trace_name=f"schema_gen_{trajectory_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                save_intermediate_outputs=True,
                langfuse_session_id=f"schema_generation_{datetime.now().strftime('%Y%m%d')}"
            )

            if result is None or tool_calls is None:
                raise Exception("Schema fetching failed - check logs for details")

            print(f"Retrieved {len(tool_calls)} tool calls")

            # Extract schema context
            schema_context = _extract_schema_context(tool_calls)
            print(f"Extracted schema context ({len(schema_context)} characters)")

            # Prepare output data
            schema_data = {
                "trajectory_file": trajectory_file.name,
                "generated_at": datetime.now().isoformat(),
                "num_actions": len(trajectory),
                "num_tool_calls": len(tool_calls),
                "schema_context_size": len(schema_context),
                "tool_calls": tool_calls,
                "schema_context": schema_context
            }

            # Save to schemas directory
            output_file = output_dir / trajectory_file.name
            with open(output_file, "w") as f:
                json.dump(schema_data, f, indent=2, default=str)

            print(f"✅ Saved schema to: {output_file}")

            return {
                "file": trajectory_file.name,
                "status": "success",
                "num_tool_calls": len(tool_calls),
                "schema_size": len(schema_context)
            }

        except Exception as e:
            print(f"❌ Error processing {trajectory_file.name}: {e}")
            import traceback
            traceback.print_exc()

            return {
                "file": trajectory_file.name,
                "status": "error",
                "error": str(e)
            }

    async def generate_all_schemas(
        self,
        trajectories_dir: Path = None,
        schemas_dir: Path = None,
        model: str = "gpt-4o",
        limit: int = None,
        selected_file: str = None,
    ):
        """
        Generate schemas for all trajectory files.

        Args:
            trajectories_dir: Directory containing trajectory files
            schemas_dir: Directory to save schemas
            model: OpenAI model to use
            limit: Optional limit on number of files to process
            selected_file: Optional specific file to process
        """
        # Set default paths
        trajectories_dir = trajectories_dir or self.base_dir / "trajectories"
        schemas_dir = schemas_dir or self.base_dir / "schemas"

        # Create model-specific subfolder
        model_schemas_dir = schemas_dir / model
        model_schemas_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Schemas will be saved to: {model_schemas_dir}")

        # Get all trajectory files
        trajectory_files = sorted(trajectories_dir.glob("*.json"))

        if limit:
            trajectory_files = trajectory_files[:limit]
            print(f"⚠️ Limited to {limit} files for testing")

        if selected_file:
            trajectory_files = [i for i in trajectory_files if selected_file in i.name]
            print(f"⚠️ Selected file: {selected_file}")

        print(f"\n📊 Found {len(trajectory_files)} trajectory files to process")

        # Process each trajectory
        results = []
        for i, trajectory_file in enumerate(trajectory_files, 1):
            print(f"\n[{i}/{len(trajectory_files)}]")
            result = await self.generate_schema_for_trajectory(
                trajectory_file=trajectory_file,
                output_dir=model_schemas_dir,
                model=model
            )
            results.append(result)

            # Small delay to avoid rate limits
            if i < len(trajectory_files):
                print("⏳ Waiting 2 seconds before next file...")
                await asyncio.sleep(2)

        # Print summary
        print(f"\n{'='*80}")
        print("GENERATION SUMMARY")
        print(f"{'='*80}")

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        print(f"✅ Successful: {len(successful)}/{len(results)}")
        print(f"❌ Failed: {len(failed)}/{len(results)}")

        if successful:
            print(f"\nSuccessful files:")
            for r in successful:
                print(f"  - {r['file']}: {r['num_tool_calls']} tool calls, {r['schema_size']} chars")

        if failed:
            print(f"\n❌ Failed files:")
            for r in failed:
                print(f"  - {r['file']}: {r['error']}")

        # Save summary
        summary_file = model_schemas_dir / "_generation_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "model": model,
                "generated_at": datetime.now().isoformat(),
                "total_files": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "results": results
            }, f, indent=2, default=str)

        print(f"\n💾 Summary saved to: {summary_file}")
        return results


# ============================================================================
# TABLE SCHEMA GENERATOR
# ============================================================================

class TableSchemaGenerator:
    """Generate comprehensive ServiceNow database schemas"""

    def __init__(self):
        self.base_dir = Path(__file__).parent

    def fetch_javascript_default_value(self, js_default_value: str) -> List[str]:
        """
        Parse the javascript default value function and extract all function/class names used.
        Returns a list of external dependencies (classes/functions) that need to be looked up.
        """
        js_default_value = js_default_value.replace("javascript:", "").strip()

        # Remove comments to avoid picking up identifiers from comments
        js_default_value = re.sub(r'//.*?$', '', js_default_value, flags=re.MULTILINE)
        js_default_value = re.sub(r'/\*.*?\*/', '', js_default_value, flags=re.DOTALL)

        external_dependencies = set()

        # 1. Extract script names from gs.include() calls
        include_pattern = r'gs\.include\s*\(\s*["\']([^"\']+)["\']\s*\)'
        includes = re.findall(include_pattern, js_default_value)
        external_dependencies.update(includes)

        # 2. Find Class.method patterns (external class usage)
        class_method_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*'
        class_methods = re.findall(class_method_pattern, js_default_value)

        # 3. Find variables/classes being instantiated or used
        instantiation_pattern = r'\bnew\s+([A-Z][a-zA-Z0-9_]*)'
        instantiations = re.findall(instantiation_pattern, js_default_value)

        standalone_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\b'
        all_capitals = re.findall(standalone_pattern, js_default_value)

        # Filter out those that are being defined in this script
        defined_pattern1 = r'\b(var|let|const|function)\s+([A-Z][a-zA-Z0-9_]*)'
        defined_pattern2 = r'\b([A-Z][a-zA-Z0-9_]*)\s*=\s*function'
        defined_pattern3 = r'\b([A-Z][a-zA-Z0-9_]*)\.prototype\s*='
        defined_pattern4 = r'\b([A-Z][a-zA-Z0-9_]*)\s*='

        defined_names = set()

        matches1 = re.findall(defined_pattern1, js_default_value)
        for match in matches1:
            defined_names.add(match[1])

        matches2 = re.findall(defined_pattern2, js_default_value)
        defined_names.update(matches2)

        matches3 = re.findall(defined_pattern3, js_default_value)
        defined_names.update(matches3)

        matches4 = re.findall(defined_pattern4, js_default_value)
        defined_names.update(matches4)

        # Define built-ins to filter out
        js_builtins = {'Object', 'Array', 'String', 'Number', 'Boolean', 'Date', 'RegExp', 'Error', 'JSON', 'Math', 'Class'}

        def is_valid_external_dependency(name):
            return (name not in defined_names and
                    name not in js_builtins and
                    not name.isupper())

        # Add class names from various sources
        for class_name in class_methods:
            if is_valid_external_dependency(class_name):
                external_dependencies.add(class_name)

        for class_name in instantiations:
            if is_valid_external_dependency(class_name):
                external_dependencies.add(class_name)

        for capital in all_capitals:
            if is_valid_external_dependency(capital):
                external_dependencies.add(capital)

        # Remove common ServiceNow built-ins
        servicenow_builtins = {'GlideRecord', 'GlideElement', 'GlideDateTime', 'GlideUser', 'GlideSysAttachment'}
        external_dependencies = external_dependencies - servicenow_builtins

        return list(external_dependencies)

    def generate_table_schemas(self, output_file: str = None):
        """
        Generate the table schemas for all tables in ServiceNow.
        """
        instance = SNowInstance()
        tables = get_all_tables(return_sys_tables=False)
        params = {
            "sysparm_limit": 8000,
            "sysparm_fields": "reference, element, mandatory, internal_type, default_value",
        }

        if output_file is None:
            output_file = self.base_dir / "prompts" / "all_table_schemas.json"
        else:
            output_file = Path(output_file)

        all_table_schemas = {}
        num_errors = 0

        for table in tqdm(tables, desc="Getting table schemas"):
            try:
                all_table_schemas[table] = []
                params["sysparm_query"] = f"name={table}"
                resp = table_api_call(instance, table="sys_dictionary", params=params)

                # Check all the choices for the columns of the table
                col_choices_resp = table_api_call(instance, table="sys_choice", params={
                    "sysparm_query": f"name={table}",
                    "sysparm_fields": "value, element",
                    "sysparm_limit": 50,
                })

                col_choices_dict = {}
                if col_choices_resp['result']:
                    for col_choice in col_choices_resp['result']:
                        if col_choice['element'] not in col_choices_dict:
                            col_choices_dict[col_choice['element']] = []
                        col_choices_dict[col_choice['element']].append(col_choice['value'])

                for column_record in resp['result']:
                    column_record['internal_type'] = column_record['internal_type'].get('value')

                    # Check if column data type is time-field
                    if column_record['internal_type'] in ['glide_date_time', 'glide_duration', 'glide_time']:
                        continue

                    # Check if there are fixed choices for the column
                    if col_choices_resp['result']:
                        col_choices = col_choices_dict.get(column_record['element'], [])
                        if col_choices:
                            column_record['choices'] = json.dumps(col_choices)

                    # Check if column's default value has javascript code
                    if 'javascript' in column_record['default_value']:
                        print(f"Processing javascript code for column {column_record['element']}: {column_record['default_value']}")
                        total_external_dependencies = set()
                        javascript_codes = column_record['default_value']
                        total_javascript_codes = javascript_codes

                        while True:
                            external_dependencies = self.fetch_javascript_default_value(javascript_codes)
                            external_dependencies = set(external_dependencies) - total_external_dependencies
                            if not external_dependencies:
                                break
                            total_external_dependencies.update(external_dependencies)

                            js_params = {"sysparm_query": f"nameIN{','.join(external_dependencies)}"}
                            javascript_codes = table_api_call(instance, table="sys_script_include", params=js_params)['result']
                            javascript_codes = "\n".join([js['script'] for js in javascript_codes])
                            total_javascript_codes += "\n" + javascript_codes

                        print(f"Total javascript code for column {column_record['element']} in table {table}: {total_javascript_codes}")
                        column_record['all_javascript_context'] = total_javascript_codes

                    all_table_schemas[table].append(column_record)

            except Exception as e:
                print(f"Error getting table schema for table {table}: {e}")
                num_errors += 1
                continue

        print(f"Number of errors: {num_errors}")

        with open(output_file, "w") as f:
            json.dump(all_table_schemas, f, indent=2)

        print(f"✅ Saved table schemas to: {output_file}")
        return all_table_schemas


# ============================================================================
# TOOL MAPPING GENERATOR
# ============================================================================

class ToolMappingGenerator:
    """Generator for tool request type mappings"""

    def __init__(self):
        self.mcp_server = None
        self.tools = []
        self.base_dir = Path(__file__).parent

    async def initialize_mcp_server(self, tool_package: str = "full"):
        """Initialize ServiceNow MCP server and extract tools"""
        # Import here to avoid circular dependencies
        sys.path.insert(0, str(self.base_dir.parent.parent / "servicenow-mcp" / "src"))

        from servicenow_mcp.server import ServiceNowMCP
        from servicenow_mcp.utils.config import ServerConfig

        os.environ["MCP_TOOL_PACKAGE"] = tool_package

        config = {
            "instance_url": "https://example.service-now.com",
            "auth": {
                "type": "basic",
                "basic": {
                    "username": "dummy",
                    "password": "dummy"
                }
            }
        }

        server_config = ServerConfig(**config)
        self.mcp_server = ServiceNowMCP(server_config)

        # Get tools from MCP server
        tools_response = await self.mcp_server._list_tools_impl()
        self.tools = tools_response

        print(f"✅ Loaded {len(self.tools)} ServiceNow MCP tools")

    def parse_tool_implementations(self) -> Dict[str, List[str]]:
        """
        Parse actual tool implementations to find HTTP request types.

        Returns:
            Dict[str, List[str]]: Mapping of tool_name -> [actual_request_types]
        """
        tool_request_mapping = {}
        servicenow_mcp_path = self.base_dir.parent.parent / "servicenow-mcp" / "src" / "servicenow_mcp" / "tools"

        # Map tool names to their implementation functions
        tool_to_function = {tool.name: tool.name for tool in self.tools}

        # HTTP request patterns to search for
        request_patterns = {
            'get': [r'requests\.get\s*\(', r'\.get\s*\(.*headers'],
            'post': [r'requests\.post\s*\(', r'\.post\s*\(.*headers'],
            'put': [r'requests\.put\s*\(', r'\.put\s*\(.*headers', r'requests\.patch\s*\(', r'\.patch\s*\(.*headers'],
            'delete': [r'requests\.delete\s*\(', r'\.delete\s*\(.*headers']
        }

        # Process each tool file
        for tool_file in servicenow_mcp_path.glob("*.py"):
            if tool_file.name == "__init__.py":
                continue

            try:
                content = tool_file.read_text()

                # Find all function definitions
                function_matches = re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)

                for func_match in function_matches:
                    func_name = func_match.group(1)
                    func_start = func_match.start()

                    # Find the end of this function
                    next_func = re.search(r'\ndef\s+\w+\s*\(', content[func_start + 1:])
                    if next_func:
                        func_end = func_start + next_func.start() + 1
                    else:
                        func_end = len(content)

                    func_body = content[func_start:func_end]

                    # Check if this function corresponds to a tool
                    if func_name in tool_to_function:
                        request_types = set()

                        # Search for HTTP request patterns in function body
                        for req_type, patterns in request_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, func_body):
                                    request_types.add(req_type)

                        if request_types:
                            tool_request_mapping[func_name] = sorted(list(request_types))

            except Exception as e:
                print(f"Error parsing {tool_file}: {e}")
                continue

        return tool_request_mapping

    def create_pattern_fallback_mapping(self) -> Dict[str, List[str]]:
        """
        Create pattern-based mapping for tools not found in code analysis.
        """
        tool_request_mapping = {}

        # Common patterns to identify request types
        get_patterns = ['get_', 'list_', 'search_', 'find_', 'retrieve_', 'fetch_', 'read_']
        post_patterns = ['create_', 'add_', 'insert_', 'new_']
        put_patterns = ['update_', 'modify_', 'edit_', 'change_', 'set_']
        delete_patterns = ['delete_', 'remove_', 'drop_']

        for tool in self.tools:
            tool_name = tool.name
            tool_desc = tool.description.lower()
            tool_name_lower = tool_name.lower()

            request_types = set()

            # Check for explicit patterns in tool name
            if any(pattern in tool_name_lower for pattern in get_patterns):
                request_types.add("get")
            if any(pattern in tool_name_lower for pattern in post_patterns):
                request_types.add("post")
            if any(pattern in tool_name_lower for pattern in put_patterns):
                request_types.add("put")
            if any(pattern in tool_name_lower for pattern in delete_patterns):
                request_types.add("delete")

            # Check description for additional clues
            if any(keyword in tool_desc for keyword in ['retrieve', 'get', 'list', 'search', 'find', 'fetch', 'read']):
                request_types.add("get")
            if any(keyword in tool_desc for keyword in ['create', 'add', 'insert', 'new']):
                request_types.add("post")
            if any(keyword in tool_desc for keyword in ['update', 'modify', 'edit', 'change', 'set', 'assign', 'attach']):
                request_types.add("put")
            if any(keyword in tool_desc for keyword in ['delete', 'remove', 'drop']):
                request_types.add("delete")

            # If no request types identified, make educated guess
            if not request_types:
                params = tool.inputSchema.get("properties", {})
                has_id_params = any("id" in param_name.lower() for param_name in params.keys())

                if has_id_params:
                    request_types.add("get")
                else:
                    request_types.add("post")

            tool_request_mapping[tool_name] = sorted(list(request_types))

        return tool_request_mapping

    def generate_complete_mapping(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Generate complete mapping using code analysis with pattern-based fallback.
        """
        print("🔍 Parsing actual tool implementations...")
        actual_mapping = self.parse_tool_implementations()

        print("🔍 Using pattern-based fallback for unmapped tools...")
        pattern_mapping = self.create_pattern_fallback_mapping()

        # Combine actual parsing with pattern-based fallback
        mapping = {}
        for tool in self.tools:
            tool_name = tool.name
            if tool_name in actual_mapping:
                mapping[tool_name] = actual_mapping[tool_name]
            elif tool_name in pattern_mapping:
                mapping[tool_name] = pattern_mapping[tool_name]
            else:
                mapping[tool_name] = ["unknown"]

        return mapping, actual_mapping

    def analyze_and_save_mapping(self, output_file: str = "tool_request_mapping.json"):
        """
        Generate, analyze, and save the complete tool mapping.
        """
        mapping, actual_mapping = self.generate_complete_mapping()

        # Group by request type
        by_type = {"get": [], "post": [], "put": [], "delete": []}
        multi_type_tools = []

        for tool_name, request_types in mapping.items():
            if len(request_types) > 1:
                multi_type_tools.append((tool_name, request_types))

            for req_type in request_types:
                if req_type in by_type:
                    by_type[req_type].append(tool_name)

        print(f"\n📊 Tool Request Type Analysis ({len(mapping)} total tools):")
        print("=" * 60)
        print(f"✅ {len(actual_mapping)} tools parsed from actual code")
        print(f"🔤 {len(mapping) - len(actual_mapping)} tools from pattern matching")

        # Show tools with multiple request types
        if multi_type_tools:
            print(f"\n🔀 MULTI-TYPE TOOLS ({len(multi_type_tools)} tools):")
            for tool_name, request_types in sorted(multi_type_tools):
                types_str = ", ".join(request_types)
                print(f"   - {tool_name}: [{types_str}]")

        # Show single-type tools by category
        single_type_tools = {k: [t for t in v if len(mapping[t]) == 1] for k, v in by_type.items()}

        for req_type, tools in single_type_tools.items():
            if tools:
                print(f"\n🔹 {req_type.upper()} ONLY ({len(tools)} tools):")
                for tool in sorted(tools):
                    print(f"   - {tool}")

        # Summary statistics
        total_single = sum(len(tools) for tools in single_type_tools.values())
        print(f"\n📈 SUMMARY:")
        print(f"   Single-type tools: {total_single}")
        print(f"   Multi-type tools: {len(multi_type_tools)}")
        print(f"   Total tools: {len(mapping)}")

        # Save to JSON file
        output_path = self.base_dir / "prompts" / output_file
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)

        print(f"\n💾 Saved complete mapping to {output_path}")

        return mapping


# ============================================================================
# TOOL SPEC GENERATOR
# ============================================================================

class ToolSpecGenerator:
    """Generates tool specifications from MCP server"""

    def __init__(self, model: str = "gpt-4o"):
        self.agent = None
        self.model = model

    async def initialize(self, tool_package: str = "full"):
        """Initialize the MCP server"""
        from .world_model_agent import WorldModelAgent
        self.agent = WorldModelAgent(model=self.model)
        await self.agent.initialize_mcp_server(tool_package)

    def extract_tool_specs(self) -> List[Dict[str, Any]]:
        """Extract tool specifications from the MCP server"""
        tool_specs = []

        for tool in self.agent.tools:
            spec = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "metadata": getattr(tool, 'metadata', {})
            }

            # Add additional analysis
            spec["analysis"] = self._analyze_tool_spec(tool)

            tool_specs.append(spec)

        return tool_specs

    def _analyze_tool_spec(self, tool) -> Dict[str, Any]:
        """Analyze a tool specification to extract useful metadata"""
        analysis = {
            "parameter_count": 0,
            "required_parameters": [],
            "optional_parameters": [],
            "parameter_types": {},
            "has_description": bool(tool.description),
            "complexity_score": 0
        }

        # Analyze input schema
        if tool.inputSchema and isinstance(tool.inputSchema, dict):
            properties = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])

            analysis["parameter_count"] = len(properties)
            analysis["required_parameters"] = required
            analysis["optional_parameters"] = [p for p in properties.keys() if p not in required]

            # Extract parameter types
            for param_name, param_spec in properties.items():
                if isinstance(param_spec, dict):
                    param_type = param_spec.get("type", "unknown")
                    analysis["parameter_types"][param_name] = param_type

            # Calculate complexity score
            complexity = len(required) * 2 + len(analysis["optional_parameters"])
            analysis["complexity_score"] = complexity

        return analysis

    def save_tool_specs(self, tool_specs: List[Dict[str, Any]], output_file: str):
        """Save tool specifications to JSON file"""
        output_data = {
            "tool_specifications": tool_specs,
            "metadata": {
                "total_tools": len(tool_specs),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"✅ Saved tool specifications to: {output_file}")
