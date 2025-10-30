import json
from typing import Dict, List, Any

async def generate_tool_specs(model: str = "gpt-4o", tool_package: str = "full", output_file: str = None):
    """
    Generate tool specifications from MCP server.

    Args:
        model: Model name for WorldModelAgent
        tool_package: Package type for MCP tools
        output_file: Output file path. If None, saves to prompts/tool_specs.json
    """
    from ..agent import WorldModelAgent
    from pathlib import Path

    agent = WorldModelAgent(model=model)
    await agent.initialize_mcp_server(tool_package)

    tool_specs = []
    for tool in agent.tools:
        spec = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
            "metadata": getattr(tool, 'metadata', {})
        }
        tool_specs.append(spec)

    output_data = {
        "tool_specifications": tool_specs,
        "metadata": {
            "total_tools": len(tool_specs),
        }
    }

    if output_file is None:
        base_dir = Path(__file__).parent.parent.parent
        output_path = base_dir / "prompts" / "tool_specs.json"
    else:
        output_path = Path(output_file)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    return tool_specs