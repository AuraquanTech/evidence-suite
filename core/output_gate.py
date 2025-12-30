# core/output_gate.py
"""Non-bypassable output gate for client-facing content.

This module implements a default-deny gate that prevents unverified
analysis from reaching clients. There is intentionally NO bypass flag.

Policy:
- ONLY allow when routing == 'auto_ok'
- ALWAYS block on hard-stop flags (impossible timelines, missing data)
- Internal tools can access full data; client-facing outputs cannot
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from core.models import EvidencePacket
    from pipeline import PipelineResult


class RoutingDecision(str, Enum):
    """Routing decisions for analysis output."""

    AUTO_OK = "auto_ok"  # Clean, can go to client
    REVIEW = "review"  # Needs human review
    NEEDS_DOCS = "needs_docs"  # Missing documentation
    BLOCKED = "blocked"  # Hard stop, cannot proceed


@dataclass(frozen=True)
class OutputBlockReason:
    """Reason why output was blocked."""

    code: str
    detail: str
    severity: str = "high"  # high, medium, low


class ClientOutputBlocked(RuntimeError):
    """Exception raised when client-facing output is blocked."""

    def __init__(self, reasons: list[OutputBlockReason]):
        msg = "Client-facing output blocked:\n" + "\n".join(
            f"  [{r.severity.upper()}] {r.code}: {r.detail}" for r in reasons
        )
        super().__init__(msg)
        self.reasons = reasons


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Safely get attribute from object, trying multiple names."""
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return default


def _iter_flags(obj: Any) -> Iterable[Any]:
    """Iterate over flags if present."""
    flags = _get(obj, "flags", "hard_stop_flags", default=[])
    return flags if flags else []


def compute_routing(result: PipelineResult) -> RoutingDecision:
    """Compute routing decision for a pipeline result.

    Returns:
        RoutingDecision indicating where this case should go
    """
    if not result.success:
        return RoutingDecision.BLOCKED

    packet = result.packet
    reasons = _collect_block_reasons(packet)

    if any(r.severity == "critical" for r in reasons):
        return RoutingDecision.BLOCKED

    if any(r.severity == "high" for r in reasons):
        return RoutingDecision.REVIEW

    if any(r.code.startswith("MISSING_") for r in reasons):
        return RoutingDecision.NEEDS_DOCS

    if any(r.severity == "medium" for r in reasons):
        return RoutingDecision.REVIEW

    return RoutingDecision.AUTO_OK


def _collect_block_reasons(packet: EvidencePacket) -> list[OutputBlockReason]:
    """Collect all reasons that might block or flag this packet."""
    reasons: list[OutputBlockReason] = []

    # Check behavioral indicators for concerning patterns
    bi = packet.behavioral_indicators
    if bi:
        if bi.deception_indicators > 0.7:
            reasons.append(
                OutputBlockReason(
                    code="HIGH_DECEPTION_SCORE",
                    detail=f"Deception indicators at {bi.deception_indicators:.2f}",
                    severity="high",
                )
            )

        if bi.manipulation_score > 0.7:
            reasons.append(
                OutputBlockReason(
                    code="HIGH_MANIPULATION_SCORE",
                    detail=f"Manipulation score at {bi.manipulation_score:.2f}",
                    severity="medium",
                )
            )

    # Check fusion results
    if packet.fused_score is not None and packet.fused_score < 0.3:
        reasons.append(
            OutputBlockReason(
                code="LOW_FUSION_SCORE",
                detail=f"Fused confidence score only {packet.fused_score:.2f}",
                severity="high",
            )
        )

    # Check for missing critical data
    if not packet.extracted_text or len(packet.extracted_text.strip()) < 50:
        reasons.append(
            OutputBlockReason(
                code="MISSING_CONTENT",
                detail="Insufficient extracted text content",
                severity="high",
            )
        )

    # Check analysis results for errors
    for ar in packet.analysis_results:
        if ar.errors:
            reasons.append(
                OutputBlockReason(
                    code="ANALYSIS_ERROR",
                    detail=f"{ar.agent_type}: {ar.errors[0]}",
                    severity="high",
                )
            )
        if ar.confidence < 0.5:
            reasons.append(
                OutputBlockReason(
                    code="LOW_AGENT_CONFIDENCE",
                    detail=f"{ar.agent_type} confidence only {ar.confidence:.2f}",
                    severity="medium",
                )
            )

    # Check fusion metadata for anomalies
    fusion_meta = packet.fusion_metadata or {}
    anomalies = fusion_meta.get("anomalies", [])
    for anomaly in anomalies:
        if isinstance(anomaly, dict):
            reasons.append(
                OutputBlockReason(
                    code=anomaly.get("code", "ANOMALY_DETECTED"),
                    detail=anomaly.get("detail", str(anomaly)),
                    severity=anomaly.get("severity", "medium"),
                )
            )
        else:
            reasons.append(
                OutputBlockReason(
                    code="ANOMALY_DETECTED",
                    detail=str(anomaly),
                    severity="medium",
                )
            )

    # Check for impossible timeline flags
    consistency = fusion_meta.get("consistency", {})
    if consistency.get("timeline_valid") is False:
        reasons.append(
            OutputBlockReason(
                code="IMPOSSIBLE_TIMELINE",
                detail="Treatment dates precede incident date",
                severity="critical",
            )
        )

    return reasons


def ensure_client_output_allowed(result: PipelineResult) -> None:
    """Default-deny gate for anything client-facing.

    This function raises ClientOutputBlocked if the result cannot
    be safely sent to clients. There is NO bypass flag by design.

    Args:
        result: The pipeline result to check

    Raises:
        ClientOutputBlocked: If output cannot go to client
    """
    routing = compute_routing(result)

    if routing == RoutingDecision.AUTO_OK:
        return  # Safe to proceed

    # Collect all reasons for the block
    reasons = _collect_block_reasons(result.packet)

    if routing == RoutingDecision.BLOCKED:
        reasons.insert(
            0,
            OutputBlockReason(
                code="ROUTING_BLOCKED",
                detail="Case has critical issues that prevent any output",
                severity="critical",
            ),
        )
    elif routing == RoutingDecision.REVIEW:
        reasons.insert(
            0,
            OutputBlockReason(
                code="ROUTING_REVIEW",
                detail="Case requires human review before client output",
                severity="high",
            ),
        )
    elif routing == RoutingDecision.NEEDS_DOCS:
        reasons.insert(
            0,
            OutputBlockReason(
                code="ROUTING_NEEDS_DOCS",
                detail="Case is missing required documentation",
                severity="high",
            ),
        )

    raise ClientOutputBlocked(reasons)


def get_internal_summary(result: PipelineResult) -> dict[str, Any]:
    """Get full analysis summary for INTERNAL use only.

    This bypasses the client gate and returns everything.
    Use for internal dashboards, review queues, debugging.
    """
    packet = result.packet

    return {
        "id": packet.id,
        "case_id": packet.case_id,
        "success": result.success,
        "routing": compute_routing(result).value,
        "block_reasons": [
            {"code": r.code, "detail": r.detail, "severity": r.severity}
            for r in _collect_block_reasons(packet)
        ],
        "fused_score": packet.fused_score,
        "fused_classification": packet.fused_classification,
        "behavioral": {
            "darvo": packet.behavioral_indicators.darvo_score,
            "gaslighting": packet.behavioral_indicators.gaslighting_score,
            "manipulation": packet.behavioral_indicators.manipulation_score,
            "deception": packet.behavioral_indicators.deception_indicators,
        }
        if packet.behavioral_indicators
        else None,
        "stages_completed": [str(s) for s in result.stages_completed],
        "total_time_ms": result.total_time_ms,
        "errors": result.errors,
    }


def get_client_summary(result: PipelineResult) -> dict[str, Any]:
    """Get analysis summary for CLIENT-FACING use.

    This enforces the output gate. Raises ClientOutputBlocked if
    the result cannot be safely sent to clients.
    """
    ensure_client_output_allowed(result)

    packet = result.packet

    return {
        "id": packet.id,
        "case_id": packet.case_id,
        "status": "complete",
        "classification": packet.fused_classification,
        "confidence": packet.fused_score,
        "summary": _generate_safe_summary(packet),
    }


def _generate_safe_summary(packet: EvidencePacket) -> str:
    """Generate a safe summary for client output."""
    if packet.fused_classification:
        return f"Analysis complete. Classification: {packet.fused_classification}"
    return "Analysis complete. See full report for details."
