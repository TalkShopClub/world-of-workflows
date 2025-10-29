#!/usr/bin/env python3
"""
Generate and cache schemas for all trajectories in the trajectories/ directory.
Saves schema context to schemas/ folder with matching filenames.
"""

import asyncio
import json
from pathlib import Path
import os
from datetime import datetime

# Import for direct execution or as module
try:
    from .agentic_schema_pipeline import (
        _build_schema_fetching_query,
        solve_llm_with_tracing,
        _extract_schema_context
    )
except ImportError:
    from agentic_schema_pipeline import (
        _build_schema_fetching_query,
        solve_llm_with_tracing,
        _extract_schema_context
    )
from langchain_openai import ChatOpenAI


async def generate_schema_for_trajectory(
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
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional
                    "X-Title": "Schema Generation"  # Optional
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
    trajectories_dir: Path = None,
    schemas_dir: Path = None,
    model: str = "gpt-4o",
    limit: int = None,
    selected_file: str = None,
):
    """
    Generate schemas for all trajectory files.

    Args:
        trajectories_dir: Directory containing trajectory files (default: ./trajectories/)
        schemas_dir: Directory to save schemas (default: ./schemas/)
        model: OpenAI model to use
        limit: Optional limit on number of files to process (for testing)
    """
    # Set default paths
    script_dir = Path(__file__).parent
    trajectories_dir = trajectories_dir or script_dir / "trajectories"
    schemas_dir = schemas_dir or script_dir / "schemas"

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
    # breakpoint()
    # Process each trajectory
    results = []
    for i, trajectory_file in enumerate(trajectory_files, 1):
        print(f"\n[{i}/{len(trajectory_files)}]")
        result = await generate_schema_for_trajectory(
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


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate schemas for trajectory files")
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        help="Directory containing trajectory files (default: ./trajectories/)"
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        help="Directory to save schemas (default: ./schemas/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--selected-file",
        type=str,
        help="Selected file to process"
    )
    args = parser.parse_args()

    await generate_all_schemas(
        trajectories_dir=args.trajectories_dir,
        schemas_dir=args.schemas_dir,
        model=args.model,
        limit=args.limit,
        selected_file=args.selected_file
    )


if __name__ == "__main__":
    asyncio.run(main())
