#!/usr/bin/env python3
"""
Tool Specification Generator

Extracts all MCP tool specifications from the ServiceNow MCP server
and saves them to a JSON file for reference and analysis.
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

class ToolSpecGenerator:
    """Generates tool specifications from MCP server"""

    def __init__(self, openai_api_key: str):
        self.agent = WorldModelAgent(openai_api_key)

    async def initialize(self, tool_package: str = "full"):
        """Initialize the MCP server"""
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

    def generate_summary_stats(self, tool_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics about the tools"""
        stats = {
            "total_tools": len(tool_specs),
            "tools_with_parameters": 0,
            "tools_without_parameters": 0,
            "average_parameters_per_tool": 0,
            "most_complex_tools": [],
            "parameter_type_distribution": {},
            "common_parameter_names": {},
            "generation_timestamp": datetime.now().isoformat()
        }

        total_params = 0
        complexity_scores = []

        for spec in tool_specs:
            analysis = spec["analysis"]
            param_count = analysis["parameter_count"]

            total_params += param_count
            complexity_scores.append((spec["name"], analysis["complexity_score"]))

            if param_count > 0:
                stats["tools_with_parameters"] += 1
            else:
                stats["tools_without_parameters"] += 1

            # Count parameter types
            for param_type in analysis["parameter_types"].values():
                stats["parameter_type_distribution"][param_type] = stats["parameter_type_distribution"].get(param_type, 0) + 1

            # Count parameter names
            for param_name in analysis["parameter_types"].keys():
                stats["common_parameter_names"][param_name] = stats["common_parameter_names"].get(param_name, 0) + 1

        # Calculate averages
        if len(tool_specs) > 0:
            stats["average_parameters_per_tool"] = total_params / len(tool_specs)

        # Sort and get most complex tools
        complexity_scores.sort(key=lambda x: x[1], reverse=True)
        stats["most_complex_tools"] = complexity_scores[:10]

        # Sort parameter names by frequency
        stats["common_parameter_names"] = dict(sorted(
            stats["common_parameter_names"].items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return stats

    def save_tool_specs(self, tool_specs: List[Dict[str, Any]], output_file: str, include_stats: bool = True):
        """Save tool specifications to JSON file"""
        output_data = {
            "tool_specifications": tool_specs,
            "metadata": {
                "total_tools": len(tool_specs),
                "generated_at": datetime.now().isoformat(),
                "generator_version": "1.0"
            }
        }

        if include_stats:
            output_data["summary_statistics"] = self.generate_summary_stats(tool_specs)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"💾 Tool specifications saved to {output_file}")

    def print_summary(self, tool_specs: List[Dict[str, Any]]):
        """Print a summary of the extracted tool specifications"""
        stats = self.generate_summary_stats(tool_specs)

        print("\n" + "=" * 60)
        print("📊 TOOL SPECIFICATIONS SUMMARY")
        print("=" * 60)
        print(f"Total tools: {stats['total_tools']}")
        print(f"Tools with parameters: {stats['tools_with_parameters']}")
        print(f"Tools without parameters: {stats['tools_without_parameters']}")
        print(f"Average parameters per tool: {stats['average_parameters_per_tool']:.1f}")

        print(f"\n🔧 Most Complex Tools:")
        for tool_name, complexity in stats['most_complex_tools'][:5]:
            print(f"  • {tool_name}: {complexity} complexity points")

        print(f"\n📋 Common Parameter Types:")
        for param_type, count in list(stats['parameter_type_distribution'].items())[:5]:
            print(f"  • {param_type}: {count} occurrences")

        print(f"\n🏷️  Common Parameter Names:")
        for param_name, count in list(stats['common_parameter_names'].items())[:5]:
            print(f"  • {param_name}: {count} tools")


async def main():
    """Main function to generate tool specifications"""

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        return

    if not os.getenv("SNOW_INSTANCE_URL"):
        print("❌ SNOW_INSTANCE_URL environment variable is required")
        return

    print("🔧 Initializing Tool Specification Generator")
    print("=" * 50)

    # Initialize the generator
    generator = ToolSpecGenerator(os.getenv("OPENAI_API_KEY"))

    try:
        # Initialize MCP server
        print("🔗 Connecting to ServiceNow MCP server...")
        await generator.initialize()

        # Extract tool specifications
        print("📊 Extracting tool specifications...")
        tool_specs = generator.extract_tool_specs()

        # Save to file
        output_file = f"tool_specifications.json"
        generator.save_tool_specs(tool_specs, output_file)

        # Print summary
        generator.print_summary(tool_specs)

        print(f"\n✅ Tool specification generation completed!")
        print(f"📄 Specifications saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error generating tool specifications: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())