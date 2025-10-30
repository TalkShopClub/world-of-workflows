#!/usr/bin/env python3
"""
World Model Utilities CLI

Command-line interface for running one-time generation utilities:
- Trajectory schemas (LLM-based, per-trajectory)
- Table schemas (ServiceNow database-wide)
- Tool request mappings (HTTP request type analysis)
- Tool specifications (MCP tool specs)

Usage:
    python generate_utilities.py trajectory-schemas --model gpt-4o --limit 5
    python generate_utilities.py table-schemas
    python generate_utilities.py tool-mapping
    python generate_utilities.py tool-specs
"""

import asyncio
import argparse
from pathlib import Path

from .world_utils import (
    TrajectorySchemaGenerator,
    TableSchemaGenerator,
    ToolMappingGenerator,
    ToolSpecGenerator
)


async def generate_trajectory_schemas(args):
    """Generate trajectory-specific schemas using LLM"""
    print("🚀 Trajectory Schema Generator")
    print("=" * 60)

    generator = TrajectorySchemaGenerator()
    await generator.generate_all_schemas(
        trajectories_dir=Path(args.trajectories_dir) if args.trajectories_dir else None,
        schemas_dir=Path(args.schemas_dir) if args.schemas_dir else None,
        model=args.model,
        limit=args.limit,
        selected_file=args.selected_file
    )


async def generate_table_schemas(args):
    """Generate ServiceNow database table schemas"""
    print("🚀 Table Schema Generator")
    print("=" * 60)

    generator = TableSchemaGenerator()
    generator.generate_table_schemas(
        output_file=args.output if args.output else None
    )


async def generate_tool_mapping(args):
    """Generate tool request type mappings"""
    print("🚀 Tool Request Mapping Generator")
    print("=" * 60)

    generator = ToolMappingGenerator()
    await generator.initialize_mcp_server(args.tool_package)
    generator.analyze_and_save_mapping(args.output if args.output else "tool_request_mapping.json")


async def generate_tool_specs(args):
    """Generate tool specifications"""
    print("🚀 Tool Specification Generator")
    print("=" * 60)

    generator = ToolSpecGenerator(model=args.model)
    await generator.initialize(args.tool_package)

    tool_specs = generator.extract_tool_specs()

    output_file = args.output if args.output else "tool_specifications.json"
    generator.save_tool_specs(tool_specs, output_file)

    print(f"\n✅ Generated {len(tool_specs)} tool specifications")
    print(f"📄 Saved to: {output_file}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="World Model Utilities - Generation Tools CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate trajectory schemas for all trajectories
  python generate_utilities.py trajectory-schemas --model gpt-4o

  # Generate trajectory schemas with limit
  python generate_utilities.py trajectory-schemas --model gpt-4o --limit 5

  # Generate trajectory schemas for specific file
  python generate_utilities.py trajectory-schemas --model gpt-4o --selected-file "task_123.json"

  # Generate table schemas
  python generate_utilities.py table-schemas

  # Generate tool request mappings
  python generate_utilities.py tool-mapping

  # Generate tool specifications
  python generate_utilities.py tool-specs --model gpt-4o
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Generation command to run')

    # Trajectory schemas subcommand
    trajectory_parser = subparsers.add_parser(
        'trajectory-schemas',
        help='Generate LLM-based trajectory-specific schemas'
    )
    trajectory_parser.add_argument(
        '--trajectories-dir',
        type=str,
        help='Directory containing trajectory files (default: ./trajectories/)'
    )
    trajectory_parser.add_argument(
        '--schemas-dir',
        type=str,
        help='Directory to save schemas (default: ./schemas/)'
    )
    trajectory_parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-4o',
        help='Model to use for schema generation (default: openai/gpt-4o)'
    )
    trajectory_parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process (for testing)'
    )
    trajectory_parser.add_argument(
        '--selected-file',
        type=str,
        help='Process only files matching this pattern'
    )

    # Table schemas subcommand
    table_parser = subparsers.add_parser(
        'table-schemas',
        help='Generate ServiceNow database table schemas'
    )
    table_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: prompts/all_table_schemas.json)'
    )

    # Tool mapping subcommand
    mapping_parser = subparsers.add_parser(
        'tool-mapping',
        help='Generate HTTP request type mappings for MCP tools'
    )
    mapping_parser.add_argument(
        '--tool-package',
        type=str,
        default='full',
        help='Tool package to load (default: full)'
    )
    mapping_parser.add_argument(
        '--output',
        type=str,
        default='tool_request_mapping.json',
        help='Output file name (default: tool_request_mapping.json)'
    )

    # Tool specs subcommand
    specs_parser = subparsers.add_parser(
        'tool-specs',
        help='Generate MCP tool specifications'
    )
    specs_parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use (default: gpt-4o)'
    )
    specs_parser.add_argument(
        '--tool-package',
        type=str,
        default='full',
        help='Tool package to load (default: full)'
    )
    specs_parser.add_argument(
        '--output',
        type=str,
        default='tool_specifications.json',
        help='Output file name (default: tool_specifications.json)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate generator
    if args.command == 'trajectory-schemas':
        await generate_trajectory_schemas(args)
    elif args.command == 'table-schemas':
        await generate_table_schemas(args)
    elif args.command == 'tool-mapping':
        await generate_tool_mapping(args)
    elif args.command == 'tool-specs':
        await generate_tool_specs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())