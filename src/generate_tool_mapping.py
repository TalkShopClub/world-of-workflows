#!/usr/bin/env python3
"""
Tool Request Type Mapping Generator

Analyzes ServiceNow MCP tool implementations to determine actual HTTP request types.
Saves the mapping to tool_request_mapping.json for use by other scripts.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "servicenow-mcp" / "src"))

try:
    from servicenow_mcp.server import ServiceNowMCP
    from servicenow_mcp.utils.config import ServerConfig
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure servicenow-mcp is available")
    sys.exit(1)


class ToolMappingGenerator:
    """Generator for tool request type mappings"""

    def __init__(self):
        self.mcp_server = None
        self.tools = []

    async def initialize_mcp_server(self, tool_package: str = "full"):
        """Initialize ServiceNow MCP server and extract tools"""
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
        servicenow_mcp_path = Path(__file__).parent.parent.parent.parent / "servicenow-mcp" / "src" / "servicenow_mcp" / "tools"

        # Map tool names to their implementation functions
        tool_to_function = {}
        for tool in self.tools:
            tool_to_function[tool.name] = tool.name

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

                    # Find the end of this function (next function or end of file)
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

            # Collect all possible request types for this tool
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

            # Convert to sorted list for consistency
            tool_request_mapping[tool_name] = sorted(list(request_types))

        return tool_request_mapping

    def generate_complete_mapping(self) -> Dict[str, List[str]]:
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

        # Group by request type (tools can appear in multiple categories)
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

        # Show tools with multiple request types first
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
        with open(os.path.join(Path(__file__).parent, "prompts", output_file), "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)

        print(f"\n💾 Saved complete mapping to {output_file}")

        return mapping


async def main():
    """Generate and save tool request type mapping"""

    print("🚀 ServiceNow MCP Tool Request Type Mapping Generator")
    print("=" * 60)

    generator = ToolMappingGenerator()
    await generator.initialize_mcp_server("full")

    mapping = generator.analyze_and_save_mapping()

    print(f"\n✅ Generated mapping for {len(mapping)} tools")
    print("📄 Mapping saved to tool_request_mapping.json")
    print("🔧 Use this file with WorldModelAgent.get_tool_request_types()")


if __name__ == "__main__":
    asyncio.run(main())