import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List


def _parse_tool_implementations(tools, servicenow_mcp_path: Path) -> Dict[str, List[str]]:
    """Parse actual tool implementations to find HTTP request types."""
    tool_request_mapping = {}
    tool_to_function = {tool.name: tool.name for tool in tools}

    request_patterns = {
        'get': [r'requests\.get\s*\(', r'\.get\s*\(.*headers'],
        'post': [r'requests\.post\s*\(', r'\.post\s*\(.*headers'],
        'put': [r'requests\.put\s*\(', r'\.put\s*\(.*headers', r'requests\.patch\s*\(', r'\.patch\s*\(.*headers'],
        'delete': [r'requests\.delete\s*\(', r'\.delete\s*\(.*headers']
    }

    for tool_file in servicenow_mcp_path.glob("*.py"):
        if tool_file.name == "__init__.py":
            continue

        try:
            content = tool_file.read_text()
            function_matches = re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)

            for func_match in function_matches:
                func_name = func_match.group(1)
                func_start = func_match.start()

                next_func = re.search(r'\ndef\s+\w+\s*\(', content[func_start + 1:])
                if next_func:
                    func_end = func_start + next_func.start() + 1
                else:
                    func_end = len(content)

                func_body = content[func_start:func_end]

                if func_name in tool_to_function:
                    request_types = set()

                    for req_type, patterns in request_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, func_body):
                                request_types.add(req_type)

                    if request_types:
                        tool_request_mapping[func_name] = sorted(list(request_types))

        except Exception:
            continue

    return tool_request_mapping


def _create_pattern_fallback_mapping(tools) -> Dict[str, List[str]]:
    """Create pattern-based mapping for tools not found in code analysis."""
    tool_request_mapping = {}

    get_patterns = ['get_', 'list_', 'search_', 'find_', 'retrieve_', 'fetch_', 'read_']
    post_patterns = ['create_', 'add_', 'insert_', 'new_']
    put_patterns = ['update_', 'modify_', 'edit_', 'change_', 'set_']
    delete_patterns = ['delete_', 'remove_', 'drop_']

    for tool in tools:
        tool_name = tool.name
        tool_desc = tool.description.lower()
        tool_name_lower = tool_name.lower()

        request_types = set()

        if any(pattern in tool_name_lower for pattern in get_patterns):
            request_types.add("get")
        if any(pattern in tool_name_lower for pattern in post_patterns):
            request_types.add("post")
        if any(pattern in tool_name_lower for pattern in put_patterns):
            request_types.add("put")
        if any(pattern in tool_name_lower for pattern in delete_patterns):
            request_types.add("delete")

        if any(keyword in tool_desc for keyword in ['retrieve', 'get', 'list', 'search', 'find', 'fetch', 'read']):
            request_types.add("get")
        if any(keyword in tool_desc for keyword in ['create', 'add', 'insert', 'new']):
            request_types.add("post")
        if any(keyword in tool_desc for keyword in ['update', 'modify', 'edit', 'change', 'set', 'assign', 'attach']):
            request_types.add("put")
        if any(keyword in tool_desc for keyword in ['delete', 'remove', 'drop']):
            request_types.add("delete")

        if not request_types:
            params = tool.inputSchema.get("properties", {})
            has_id_params = any("id" in param_name.lower() for param_name in params.keys())

            if has_id_params:
                request_types.add("get")
            else:
                request_types.add("post")

        tool_request_mapping[tool_name] = sorted(list(request_types))

    return tool_request_mapping


async def generate_tool_request_mapping(tool_package: str = "full", output_file: str = None):
    """
    Generate tool request type mapping from MCP server tools.

    Args:
        tool_package: Package type for MCP tools
        output_file: Output file path. If None, saves to prompts/tool_request_mapping.json
    """
    base_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(base_dir.parent.parent / "servicenow-mcp" / "src"))

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
    mcp_server = ServiceNowMCP(server_config)
    tools = await mcp_server._list_tools_impl()

    servicenow_mcp_path = base_dir.parent.parent / "servicenow-mcp" / "src" / "servicenow_mcp" / "tools"
    actual_mapping = _parse_tool_implementations(tools, servicenow_mcp_path)
    pattern_mapping = _create_pattern_fallback_mapping(tools)

    mapping = {}
    for tool in tools:
        tool_name = tool.name
        if tool_name in actual_mapping:
            mapping[tool_name] = actual_mapping[tool_name]
        elif tool_name in pattern_mapping:
            mapping[tool_name] = pattern_mapping[tool_name]
        else:
            mapping[tool_name] = ["unknown"]

    if output_file is None:
        output_path = base_dir / "prompts" / "tool_request_mapping.json"
    else:
        output_path = Path(output_file)

    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    return mapping
