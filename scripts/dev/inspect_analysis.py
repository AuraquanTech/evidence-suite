# scripts/dev/inspect_analysis.py
"""Introspect pipeline output to understand the analysis object shape."""

import asyncio
import json
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/dev/inspect_analysis.py <path-to-file>")
        print("       python scripts/dev/inspect_analysis.py --sample")
        return 2

    if sys.argv[1] == "--sample":
        # Run with sample text
        return asyncio.run(run_sample())

    file_path = Path(sys.argv[1]).expanduser().resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 2

    return asyncio.run(run_file(file_path))


async def run_sample() -> int:
    """Run with sample text to show analysis shape."""
    from core.models import EvidencePacket, EvidenceType
    from pipeline import EvidencePipeline

    sample_text = """
    MRI REPORT
    Patient: Jane Doe
    Date of Loss: 2025-01-10
    Provider: Dr. Smith, MD
    Diagnosis (ICD-10): S13.4 Cervical sprain
    Treatment Start: 2025-01-12
    Treatment End: 2025-02-28
    Finding: disc bulge C5-C6
    """

    pipeline = EvidencePipeline()
    try:
        await pipeline.initialize()

        packet = EvidencePacket(
            raw_content=sample_text.encode("utf-8"),
            evidence_type=EvidenceType.TEXT,
            case_id="introspection_test",
        )

        result = await pipeline.process(packet, skip_ocr=True)
        dump_result(result)
        return 0 if result.success else 1

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await pipeline.shutdown()


async def run_file(file_path: Path) -> int:
    """Run pipeline on a file."""
    from core.models import EvidencePacket, EvidenceType
    from pipeline import EvidencePipeline

    # Determine evidence type
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        evidence_type = EvidenceType.DOCUMENT
    elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        evidence_type = EvidenceType.IMAGE
    elif suffix in [".txt", ".md"]:
        evidence_type = EvidenceType.TEXT
    else:
        evidence_type = EvidenceType.DOCUMENT

    content = file_path.read_bytes()

    pipeline = EvidencePipeline()
    try:
        await pipeline.initialize()

        packet = EvidencePacket(
            raw_content=content,
            evidence_type=evidence_type,
            case_id=f"introspection_{file_path.stem}",
            source_path=str(file_path),
        )

        result = await pipeline.process(packet)
        dump_result(result)
        return 0 if result.success else 1

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await pipeline.shutdown()


def dump_result(result) -> None:
    """Dump the result structure for introspection."""
    print("\n" + "=" * 60)
    print("PIPELINE RESULT STRUCTURE")
    print("=" * 60)

    # PipelineResult fields
    print("\n## PipelineResult")
    print(
        json.dumps(
            {
                "success": result.success,
                "stages_completed": [str(s) for s in result.stages_completed],
                "total_time_ms": result.total_time_ms,
                "stage_times": result.stage_times,
                "errors": result.errors,
            },
            indent=2,
        )
    )

    # EvidencePacket fields
    packet = result.packet
    print("\n## EvidencePacket")
    packet_info = {
        "id": packet.id,
        "case_id": packet.case_id,
        "evidence_type": str(packet.evidence_type),
        "stage": str(packet.stage),
        "extracted_text_len": len(packet.extracted_text) if packet.extracted_text else 0,
        "ocr_confidence": packet.ocr_confidence,
        "fused_score": packet.fused_score,
        "fused_classification": packet.fused_classification,
        "content_hash": packet.content_hash[:16] + "...",
    }
    print(json.dumps(packet_info, indent=2))

    # BehavioralIndicators
    if packet.behavioral_indicators:
        print("\n## BehavioralIndicators")
        bi = packet.behavioral_indicators
        print(
            json.dumps(
                {
                    "sentiment_compound": bi.sentiment_compound,
                    "sentiment_positive": bi.sentiment_positive,
                    "sentiment_negative": bi.sentiment_negative,
                    "darvo_score": bi.darvo_score,
                    "gaslighting_score": bi.gaslighting_score,
                    "manipulation_score": bi.manipulation_score,
                    "deception_indicators": bi.deception_indicators,
                    "hedging_frequency": bi.hedging_frequency,
                    "certainty_markers": bi.certainty_markers,
                    "emotional_intensity": bi.emotional_intensity,
                    "primary_behavior_class": bi.primary_behavior_class,
                    "behavior_probabilities": bi.behavior_probabilities,
                },
                indent=2,
            )
        )

    # AnalysisResults
    if packet.analysis_results:
        print(f"\n## AnalysisResults ({len(packet.analysis_results)} entries)")
        for i, ar in enumerate(packet.analysis_results):
            print(f"\n  [{i}] agent_type={ar.agent_type}, confidence={ar.confidence:.3f}")
            print(f"      findings keys: {list(ar.findings.keys())}")

    # FusionMetadata
    if packet.fusion_metadata:
        print("\n## FusionMetadata")
        print(json.dumps(packet.fusion_metadata, indent=2, default=str))

    # ChainOfCustody
    print("\n## ChainOfCustody")
    print(f"   entries: {len(packet.chain_of_custody.entries)}")
    print(f"   valid: {packet.chain_of_custody.verify_chain()}")
    print(f"   chain_hash: {packet.chain_of_custody.chain_hash[:16]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
