"""Evidence Suite - Pipeline Orchestrator
Coordinates the flow of evidence through the agent hierarchy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from agents.behavioral_agent import BehavioralAgent
from agents.fusion_agent import FusionAgent
from agents.ocr_agent import OCRAgent
from core.config import Config, default_config
from core.models import EvidencePacket, EvidenceType, ProcessingStage


class PipelineStage(str, Enum):
    """Stages in the processing pipeline."""

    INTAKE = "intake"
    OCR = "ocr"
    BEHAVIORAL = "behavioral"
    FUSION = "fusion"
    OUTPUT = "output"


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    packet: EvidencePacket
    success: bool
    stages_completed: list[PipelineStage]
    total_time_ms: float
    stage_times: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class EvidencePipeline:
    """Main pipeline orchestrator for Evidence Suite.

    Manages the flow: OCR -> Behavioral -> Fusion

    Features:
    - Sequential and parallel execution modes
    - Error handling and recovery
    - Performance monitoring
    - Extensible stage hooks
    """

    def __init__(self, config: Config | None = None, use_ray: bool = False):
        self.config = config or default_config
        self.use_ray = use_ray
        self._initialized = False

        # Agents
        self._ocr_agent: OCRAgent | None = None
        self._behavioral_agent: BehavioralAgent | None = None
        self._fusion_agent: FusionAgent | None = None

        # Hooks for extensibility
        self._pre_hooks: dict[PipelineStage, list[Callable]] = {
            stage: [] for stage in PipelineStage
        }
        self._post_hooks: dict[PipelineStage, list[Callable]] = {
            stage: [] for stage in PipelineStage
        }

        # Metrics
        self._metrics = {"packets_processed": 0, "successful": 0, "failed": 0, "total_time_ms": 0.0}

        logger.info("EvidencePipeline created")

    async def initialize(self) -> None:
        """Initialize all agents in the pipeline."""
        if self._initialized:
            return

        logger.info("Initializing Evidence Pipeline...")

        # Create agents
        self._ocr_agent = OCRAgent(config=self.config.ocr)
        self._behavioral_agent = BehavioralAgent(config=self.config.behavioral)
        self._fusion_agent = FusionAgent(config=self.config.fusion)

        # Initialize in parallel
        await asyncio.gather(
            self._ocr_agent.initialize(),
            self._behavioral_agent.initialize(),
            self._fusion_agent.initialize(),
        )

        self._initialized = True
        logger.info("Evidence Pipeline initialized successfully")

    async def process(self, packet: EvidencePacket, skip_ocr: bool = False) -> PipelineResult:
        """Process an evidence packet through the full pipeline.

        Args:
            packet: The evidence to process
            skip_ocr: Skip OCR if text is already available

        Returns:
            PipelineResult with the processed packet and metadata
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        stages_completed = []
        stage_times = {}
        errors = []

        current_packet = packet

        try:
            # Stage 1: Intake (validation)
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.INTAKE, current_packet, self._intake_stage
            )
            stage_times["intake"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.INTAKE)

            # Stage 2: OCR (if needed)
            if not skip_ocr and self._needs_ocr(current_packet):
                stage_start = time.perf_counter()
                current_packet = await self._run_stage(
                    PipelineStage.OCR, current_packet, self._ocr_stage
                )
                stage_times["ocr"] = (time.perf_counter() - stage_start) * 1000
                stages_completed.append(PipelineStage.OCR)

            # Stage 3: Behavioral Analysis
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.BEHAVIORAL, current_packet, self._behavioral_stage
            )
            stage_times["behavioral"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.BEHAVIORAL)

            # Stage 4: Fusion
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.FUSION, current_packet, self._fusion_stage
            )
            stage_times["fusion"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.FUSION)

            # Stage 5: Output
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.OUTPUT, current_packet, self._output_stage
            )
            stage_times["output"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.OUTPUT)

            success = True
            self._metrics["successful"] += 1

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            errors.append(str(e))
            success = False
            self._metrics["failed"] += 1

        total_time = (time.perf_counter() - start_time) * 1000
        self._metrics["packets_processed"] += 1
        self._metrics["total_time_ms"] += total_time

        logger.info(
            f"Pipeline {'completed' if success else 'failed'} for {packet.id} in {total_time:.2f}ms"
        )

        return PipelineResult(
            packet=current_packet,
            success=success,
            stages_completed=stages_completed,
            total_time_ms=total_time,
            stage_times=stage_times,
            errors=errors,
        )

    async def process_batch(
        self, packets: list[EvidencePacket], parallel: bool = True
    ) -> list[PipelineResult]:
        """Process multiple packets.

        Args:
            packets: List of packets to process
            parallel: If True, process in parallel; else sequential
        """
        if not self._initialized:
            await self.initialize()

        if parallel:
            results = await asyncio.gather(
                *[self.process(p) for p in packets], return_exceptions=True
            )
            # Handle exceptions
            processed = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed.append(
                        PipelineResult(
                            packet=packets[i],
                            success=False,
                            stages_completed=[],
                            total_time_ms=0,
                            errors=[str(result)],
                        )
                    )
                else:
                    processed.append(result)
            return processed
        return [await self.process(p) for p in packets]

    async def _run_stage(
        self, stage: PipelineStage, packet: EvidencePacket, stage_func: Callable
    ) -> EvidencePacket:
        """Run a pipeline stage with hooks."""
        # Pre-hooks
        for hook in self._pre_hooks[stage]:
            packet = await hook(packet)

        # Run stage
        packet = await stage_func(packet)

        # Post-hooks
        for hook in self._post_hooks[stage]:
            packet = await hook(packet)

        return packet

    async def _intake_stage(self, packet: EvidencePacket) -> EvidencePacket:
        """Validate and prepare incoming evidence."""
        if not packet.raw_content and not packet.extracted_text:
            raise ValueError("Evidence packet has no content")

        logger.debug(f"Intake: {packet.id}, type={packet.evidence_type}")
        return packet

    def _needs_ocr(self, packet: EvidencePacket) -> bool:
        """Determine if OCR is needed."""
        # Skip if already has extracted text
        if packet.extracted_text and len(packet.extracted_text) > 0:
            return False

        # Skip if pure text type with raw content that's decodable
        if packet.evidence_type == EvidenceType.TEXT:
            if packet.raw_content:
                try:
                    packet.raw_content.decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    pass

        # Need OCR for images, documents, etc.
        return packet.evidence_type in [
            EvidenceType.IMAGE,
            EvidenceType.DOCUMENT,
            EvidenceType.MIXED,
        ]

    async def _ocr_stage(self, packet: EvidencePacket) -> EvidencePacket:
        """Run OCR processing."""
        return await self._ocr_agent.process(packet)

    async def _behavioral_stage(self, packet: EvidencePacket) -> EvidencePacket:
        """Run behavioral analysis."""
        return await self._behavioral_agent.process(packet)

    async def _fusion_stage(self, packet: EvidencePacket) -> EvidencePacket:
        """Run result fusion."""
        return await self._fusion_agent.process(packet)

    async def _output_stage(self, packet: EvidencePacket) -> EvidencePacket:
        """Finalize and validate output."""
        # Verify chain of custody
        if not packet.chain_of_custody.verify_chain():
            logger.warning(f"Chain of custody verification failed for {packet.id}")

        return packet.with_updates(stage=ProcessingStage.VALIDATED)

    def add_hook(self, stage: PipelineStage, hook: Callable, pre: bool = True) -> None:
        """Add a pre or post hook to a stage."""
        if pre:
            self._pre_hooks[stage].append(hook)
        else:
            self._post_hooks[stage].append(hook)

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        metrics = self._metrics.copy()
        if metrics["packets_processed"] > 0:
            metrics["avg_time_ms"] = metrics["total_time_ms"] / metrics["packets_processed"]
            metrics["success_rate"] = metrics["successful"] / metrics["packets_processed"]
        return metrics

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        logger.info("Shutting down Evidence Pipeline...")

        if self._ocr_agent:
            await self._ocr_agent.shutdown()
        if self._behavioral_agent:
            await self._behavioral_agent.shutdown()
        if self._fusion_agent:
            await self._fusion_agent.shutdown()

        self._initialized = False
        logger.info("Evidence Pipeline shutdown complete")


# Convenience function for quick processing
async def process_evidence(
    content: bytes, evidence_type: EvidenceType = EvidenceType.TEXT, case_id: str | None = None
) -> PipelineResult:
    """Quick function to process evidence through the full pipeline.

    Args:
        content: Raw evidence content
        evidence_type: Type of evidence
        case_id: Optional case identifier

    Returns:
        PipelineResult with processed evidence
    """
    packet = EvidencePacket(raw_content=content, evidence_type=evidence_type, case_id=case_id)

    pipeline = EvidencePipeline()
    try:
        result = await pipeline.process(packet)
        return result
    finally:
        await pipeline.shutdown()
