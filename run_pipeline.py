"""
Evidence Suite - Pipeline Runner
Run the first agent pipeline with sample data.
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.models import EvidencePacket, EvidenceType
from pipeline import EvidencePipeline, PipelineResult


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
    level="INFO"
)


# Sample test cases with behavioral patterns
SAMPLE_TEXTS = {
    "normal": """
    Hi, I wanted to follow up on our meeting yesterday.
    I think we made good progress on the project timeline.
    Let me know if you have any questions about the next steps.
    Looking forward to our next discussion.
    """,

    "darvo_pattern": """
    I can't believe you're accusing me of this! You always blame me for everything.
    It's YOUR fault this happened in the first place. You made me react that way.
    I'm the real victim here. After everything I've done for you, this is how you treat me?
    You're the one who needs to apologize. Look what you did to our relationship!
    """,

    "gaslighting_pattern": """
    That never happened. You're imagining things again.
    You're being way too sensitive about this. Nobody else thinks there's a problem.
    I never said that - you're remembering it wrong as usual.
    You're crazy if you think that's what occurred. Trust me, I know what really happened.
    Maybe you should see someone about your memory issues.
    """,

    "manipulation_pattern": """
    If you really loved me, you wouldn't question this.
    After all I've done for you, you owe me this much.
    No one else would put up with you like I do.
    You're lucky I'm still here. Don't tell anyone about our conversation.
    This is your fault, and you need to fix it.
    """,

    "mixed_patterns": """
    I can't believe you're bringing this up again. You always do this.
    That conversation never happened the way you remember it.
    You're being paranoid and too sensitive. I'm the one being hurt here!
    If you cared about me at all, you'd stop asking these questions.
    You made me act that way - look what you did to us!
    No one would believe you anyway. You're imagining things.
    """
}


def format_result(result: PipelineResult) -> str:
    """Format pipeline result for display."""
    packet = result.packet
    lines = [
        "=" * 60,
        f"Evidence ID: {packet.id}",
        f"Case ID: {packet.case_id or 'N/A'}",
        f"Status: {'SUCCESS' if result.success else 'FAILED'}",
        f"Total Time: {result.total_time_ms:.2f}ms",
        "-" * 60,
        "STAGE TIMES:",
    ]

    for stage, time_ms in result.stage_times.items():
        lines.append(f"  {stage}: {time_ms:.2f}ms")

    lines.append("-" * 60)
    lines.append("BEHAVIORAL INDICATORS:")

    if packet.behavioral_indicators:
        bi = packet.behavioral_indicators
        lines.extend([
            f"  Sentiment (compound): {bi.sentiment_compound:.3f}",
            f"  DARVO Score: {bi.darvo_score:.3f}",
            f"  Gaslighting Score: {bi.gaslighting_score:.3f}",
            f"  Manipulation Score: {bi.manipulation_score:.3f}",
            f"  Deception Indicators: {bi.deception_indicators:.3f}",
            f"  Primary Behavior: {bi.primary_behavior_class}",
        ])

        if bi.behavior_probabilities:
            lines.append("  Behavior Probabilities:")
            for behavior, prob in sorted(
                bi.behavior_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"    {behavior}: {prob:.3f}")
    else:
        lines.append("  No behavioral indicators available")

    lines.append("-" * 60)
    lines.append("FUSION RESULTS:")
    lines.extend([
        f"  Fused Score: {packet.fused_score:.3f}" if packet.fused_score else "  Fused Score: N/A",
        f"  Classification: {packet.fused_classification or 'N/A'}",
    ])

    if packet.fusion_metadata:
        consistency = packet.fusion_metadata.get("consistency", {})
        lines.append(f"  Consistency Score: {consistency.get('score', 'N/A')}")

        anomalies = packet.fusion_metadata.get("anomalies", [])
        if anomalies:
            lines.append(f"  Anomalies: {len(anomalies)} detected")

    lines.append("-" * 60)
    lines.append("CHAIN OF CUSTODY:")
    lines.append(f"  Entries: {len(packet.chain_of_custody.entries)}")
    lines.append(f"  Chain Valid: {packet.chain_of_custody.verify_chain()}")
    lines.append(f"  Chain Hash: {packet.chain_of_custody.chain_hash[:16]}...")

    if result.errors:
        lines.append("-" * 60)
        lines.append("ERRORS:")
        for error in result.errors:
            lines.append(f"  - {error}")

    lines.append("=" * 60)

    return "\n".join(lines)


async def run_single_test(
    pipeline: EvidencePipeline,
    name: str,
    text: str
) -> PipelineResult:
    """Run a single test case."""
    logger.info(f"Processing: {name}")

    packet = EvidencePacket(
        raw_content=text.encode('utf-8'),
        evidence_type=EvidenceType.TEXT,
        case_id=f"test_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    result = await pipeline.process(packet, skip_ocr=True)

    print(format_result(result))
    print()

    return result


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("   EVIDENCE SUITE: SAVANT GENESIS EDITION")
    print("   First Agent Pipeline Test")
    print("=" * 60 + "\n")

    # Initialize pipeline
    pipeline = EvidencePipeline()

    try:
        logger.info("Initializing pipeline...")
        await pipeline.initialize()
        logger.info("Pipeline ready!")
        print()

        # Run test cases
        results = {}
        for name, text in SAMPLE_TEXTS.items():
            result = await run_single_test(pipeline, name, text)
            results[name] = {
                "success": result.success,
                "fused_score": result.packet.fused_score,
                "classification": result.packet.fused_classification,
                "darvo": result.packet.behavioral_indicators.darvo_score if result.packet.behavioral_indicators else None,
                "gaslighting": result.packet.behavioral_indicators.gaslighting_score if result.packet.behavioral_indicators else None,
                "time_ms": result.total_time_ms
            }

        # Summary
        print("\n" + "=" * 60)
        print("   SUMMARY")
        print("=" * 60)

        for name, data in results.items():
            status = "PASS" if data["success"] else "FAIL"
            print(f"\n{name}:")
            print(f"  Status: {status}")
            print(f"  Score: {data['fused_score']:.3f}" if data['fused_score'] else "  Score: N/A")
            print(f"  Class: {data['classification']}")
            print(f"  DARVO: {data['darvo']:.3f}" if data['darvo'] else "  DARVO: N/A")
            print(f"  Gaslighting: {data['gaslighting']:.3f}" if data['gaslighting'] else "  Gaslighting: N/A")
            print(f"  Time: {data['time_ms']:.2f}ms")

        # Pipeline metrics
        print("\n" + "-" * 60)
        print("PIPELINE METRICS:")
        metrics = pipeline.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("   PIPELINE TEST COMPLETE")
        print("=" * 60 + "\n")

    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
