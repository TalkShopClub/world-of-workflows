#!/usr/bin/env python3
"""
Simple Tool Specification Extractor

Extracts basic MCP tool specifications (name, description, inputSchema)
and saves them to a clean JSON file without analysis metadata.
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from datetime import datetime

# Import for direct execution or as module
try:
    from .world_model_agent import WorldModelAgent
except ImportError:
    from world_model_agent import WorldModelAgent

async def extract_simple_tool_specs():
    """Extract simple tool specifications"""

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        return

    if not os.getenv("SNOW_INSTANCE_URL"):
        print("❌ SNOW_INSTANCE_URL environment variable is required")
        return

    print("🔧 Extracting Simple Tool Specifications")
    print("=" * 50)

    # Initialize the agent
    agent = WorldModelAgent(os.getenv("OPENAI_API_KEY"))
    await agent.initialize_mcp_server("full")

    # Extract simple tool specs
    simple_specs = []
    for tool in agent.tools:
        spec = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema
        }
        simple_specs.append(spec)

    # Save to JSON file
    output_file = "mcp_tool_specifications.json"
    with open(output_file, 'w') as f:
        json.dump(simple_specs, f, indent=2, default=str)

    print(f"✅ Extracted {len(simple_specs)} tool specifications")
    print(f"💾 Saved to: {output_file}")

    # Print a few examples
    print(f"\n📋 Example tools:")
    for i, spec in enumerate(simple_specs[:3]):
        print(f"  {i+1}. {spec['name']}: {spec['description']}")

if __name__ == "__main__":
    asyncio.run(extract_simple_tool_specs())