#!/usr/bin/env python
"""
Evidence Suite - Command Line Interface
Batch processing and management tool.
"""
import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def configure_logging(verbose: bool = False):
    """Configure logging output."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        level=level,
    )


async def process_files(
    files: List[str],
    case_id: str,
    output_dir: Optional[str] = None,
    parallel: int = 4
):
    """Process multiple evidence files."""
    from pipeline import EvidencePipeline
    from core.models import EvidenceType

    pipeline = EvidencePipeline()
    await pipeline.initialize()

    results = []
    total = len(files)

    for i, file_path in enumerate(files, 1):
        try:
            logger.info(f"[{i}/{total}] Processing: {file_path}")

            # Read file
            with open(file_path, "rb") as f:
                content = f.read()

            # Determine evidence type
            ext = Path(file_path).suffix.lower()
            if ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                etype = EvidenceType.IMAGE
            elif ext in [".wav", ".mp3", ".ogg", ".flac"]:
                etype = EvidenceType.AUDIO
            elif ext in [".eml", ".msg"]:
                etype = EvidenceType.EMAIL
            else:
                etype = EvidenceType.TEXT

            # Process
            packet = pipeline.process(content, evidence_type=etype, case_id=case_id)

            results.append({
                "file": file_path,
                "status": "success",
                "evidence_id": str(packet.evidence_id),
                "classification": packet.fusion_results.get("classification") if packet.fusion_results else None,
                "score": packet.fusion_results.get("fused_score") if packet.fusion_results else None,
            })

            logger.info(f"  -> Classification: {results[-1]['classification']}, Score: {results[-1]['score']:.3f}")

        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            results.append({
                "file": file_path,
                "status": "error",
                "error": str(e),
            })

    await pipeline.shutdown()

    # Save results
    if output_dir:
        import json
        output_path = Path(output_dir) / f"batch_results_{case_id}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    return results


async def analyze_text(text: str, case_id: str = "cli"):
    """Analyze text content."""
    from pipeline import EvidencePipeline
    from core.models import EvidenceType

    pipeline = EvidencePipeline()
    await pipeline.initialize()

    try:
        packet = pipeline.process(
            text.encode(),
            evidence_type=EvidenceType.TEXT,
            case_id=case_id
        )

        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)

        if packet.behavioral_indicators:
            bi = packet.behavioral_indicators
            print(f"\nSentiment: {bi.sentiment_compound:.3f}")
            print(f"DARVO Score: {bi.darvo_score:.3f}")
            print(f"Gaslighting Score: {bi.gaslighting_score:.3f}")
            print(f"Manipulation Score: {bi.manipulation_score:.3f}")
            print(f"Primary Behavior: {bi.primary_behavior_class}")

        if packet.fusion_results:
            fr = packet.fusion_results
            print(f"\nFused Score: {fr.get('fused_score', 0):.3f}")
            print(f"Classification: {fr.get('classification', 'unknown')}")

        print("=" * 60)

    finally:
        await pipeline.shutdown()


async def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


async def init_database():
    """Initialize database tables."""
    from core.database.session import init_db_async
    await init_db_async()
    logger.info("Database initialized successfully")


async def run_benchmark(iterations: int = 50):
    """Run performance benchmark."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_pipeline.py", "--iterations", str(iterations)],
        cwd=Path(__file__).parent
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Evidence Suite CLI - Forensic Behavioral Intelligence Platform"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process evidence files")
    process_parser.add_argument("files", nargs="+", help="Files to process")
    process_parser.add_argument("--case-id", required=True, help="Case identifier")
    process_parser.add_argument("--output", "-o", help="Output directory for results")
    process_parser.add_argument("--parallel", type=int, default=4, help="Parallel workers")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text")
    analyze_parser.add_argument("--text", "-t", help="Text to analyze")
    analyze_parser.add_argument("--file", "-f", help="File containing text to analyze")
    analyze_parser.add_argument("--case-id", default="cli", help="Case identifier")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Database command
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_parser.add_argument("action", choices=["init", "migrate", "reset"], help="Database action")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    configure_logging(args.verbose)

    if args.command == "process":
        asyncio.run(process_files(
            args.files,
            args.case_id,
            args.output,
            args.parallel
        ))

    elif args.command == "analyze":
        if args.file:
            with open(args.file, "r") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            print("Error: Provide --text or --file")
            sys.exit(1)

        asyncio.run(analyze_text(text, args.case_id))

    elif args.command == "server":
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "db":
        if args.action == "init":
            asyncio.run(init_database())
        elif args.action == "migrate":
            os.system("alembic upgrade head")
        elif args.action == "reset":
            logger.warning("Database reset not implemented - requires confirmation")

    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args.iterations))


if __name__ == "__main__":
    main()
