#!/usr/bin/env python3
"""
CLI for world-of-workflows environment utilities.

Provides command-line interfaces for:
- Generating table schemas
- Generating tool mappings
- Generating tool specifications
- Generating trajectory-specific schemas
"""

import asyncio
import argparse
from pathlib import Path


def create_parser():
    """Create the argument parser for the CLI"""
    parser = argparse.ArgumentParser(
        prog="wow-env",
        description="World of Workflows environment utilities CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Table schemas command
    table_parser = subparsers.add_parser(
        "generate-table-schemas",
        help="Generate ServiceNow table schemas"
    )
    table_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: prompts/all_table_schemas.json)"
    )

    # Tool mapping command
    mapping_parser = subparsers.add_parser(
        "generate-tool-mapping",
        help="Generate tool request type mapping"
    )
    mapping_parser.add_argument(
        "--tool-package",
        type=str,
        default="full",
        help="MCP tool package type (default: full)"
    )
    mapping_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: prompts/tool_request_mapping.json)"
    )

    # Tool specs command
    specs_parser = subparsers.add_parser(
        "generate-tool-specs",
        help="Generate tool specifications"
    )
    specs_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for WorldModelAgent (default: gpt-4o)"
    )
    specs_parser.add_argument(
        "--tool-package",
        type=str,
        default="full",
        help="MCP tool package type (default: full)"
    )
    specs_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: prompts/tool_specs.json)"
    )

    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "generate-table-schemas":
        from .schema_generation.tables import generate_table_schemas
        generate_table_schemas(output_file=args.output)

    elif args.command == "generate-tool-mapping":
        from .tools.mapping import generate_tool_request_mapping
        await generate_tool_request_mapping(
            tool_package=args.tool_package,
            output_file=args.output
        )

    elif args.command == "generate-tool-specs":
        from .tools.specs import generate_tool_specs
        await generate_tool_specs(
            model=args.model,
            tool_package=args.tool_package,
            output_file=args.output
        )


if __name__ == "__main__":
    asyncio.run(main())
