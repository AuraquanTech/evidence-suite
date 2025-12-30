"""Evidence Suite - Pipeline Orchestrator
Coordinates the flow of evidence through the agent hierarchy with type-specific routing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from agents.audio_agent import AudioAgent, AudioConfig
from agents.behavioral_agent import BehavioralAgent
from agents.document_agent import DocumentAgent, DocumentConfig
from agents.email_agent import EmailAgent, EmailConfig
from agents.fusion_agent import FusionAgent
from agents.image_agent import ImageAgent, ImageConfig
from agents.ocr_agent import OCRAgent
from agents.video_agent import VideoAgent, VideoConfig
from core.config import Config, default_config
from core.models import EvidencePacket, EvidenceType, ProcessingStage


class PipelineStage(str, Enum):
    """Stages in the processing pipeline."""

    INTAKE = "intake"
    PREPROCESSING = "preprocessing"  # Type-specific preprocessing
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
    agents_used: list[str] = field(default_factory=list)


class EvidencePipeline:
    """Main pipeline orchestrator for Evidence Suite.

    Manages type-specific routing:
    - TEXT: Direct -> Behavioral -> Fusion
    - IMAGE: ImageAgent -> OCR (optional) -> Behavioral -> Fusion
    - AUDIO: AudioAgent -> Behavioral -> Fusion
    - VIDEO: VideoAgent -> Behavioral -> Fusion
    - EMAIL: EmailAgent -> Behavioral -> Fusion
    - DOCUMENT: OCR -> Behavioral -> Fusion
    - MIXED: OCR -> Behavioral -> Fusion

    Features:
    - Evidence-type-aware routing
    - Sequential and parallel execution modes
    - Error handling and recovery
    - Performance monitoring
    - Extensible stage hooks
    """

    def __init__(self, config: Config | None = None, use_ray: bool = False):
        self.config = config or default_config
        self.use_ray = use_ray
        self._initialized = False

        # Core agents (always loaded)
        self._ocr_agent: OCRAgent | None = None
        self._behavioral_agent: BehavioralAgent | None = None
        self._fusion_agent: FusionAgent | None = None

        # Type-specific agents (lazy loaded)
        self._audio_agent: AudioAgent | None = None
        self._email_agent: EmailAgent | None = None
        self._video_agent: VideoAgent | None = None
        self._image_agent: ImageAgent | None = None
        self._document_agent: DocumentAgent | None = None

        # Track which agents are initialized
        self._agent_status: dict[str, bool] = {}

        # Hooks for extensibility
        self._pre_hooks: dict[PipelineStage, list[Callable]] = {
            stage: [] for stage in PipelineStage
        }
        self._post_hooks: dict[PipelineStage, list[Callable]] = {
            stage: [] for stage in PipelineStage
        }

        # Metrics
        self._metrics = {
            "packets_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time_ms": 0.0,
            "by_type": {},
        }

        logger.info("EvidencePipeline created with type-specific routing")

    async def initialize(self) -> None:
        """Initialize core agents in the pipeline."""
        if self._initialized:
            return

        logger.info("Initializing Evidence Pipeline...")

        # Create and initialize core agents
        self._ocr_agent = OCRAgent(config=self.config.ocr)
        self._behavioral_agent = BehavioralAgent(config=self.config.behavioral)
        self._fusion_agent = FusionAgent(config=self.config.fusion)

        # Initialize core agents in parallel
        await asyncio.gather(
            self._ocr_agent.initialize(),
            self._behavioral_agent.initialize(),
            self._fusion_agent.initialize(),
        )

        self._agent_status["ocr"] = True
        self._agent_status["behavioral"] = True
        self._agent_status["fusion"] = True

        self._initialized = True
        logger.info("Evidence Pipeline core agents initialized")

    async def _ensure_agent_initialized(self, agent_type: str) -> None:
        """Lazy initialization of type-specific agents."""
        if self._agent_status.get(agent_type):
            return

        if agent_type == "audio":
            if self._audio_agent is None:
                audio_config = AudioConfig(
                    whisper_model=getattr(self.config.audio, "whisper_model", "base"),
                    enable_diarization=getattr(self.config.audio, "enable_diarization", True),
                    device=getattr(self.config.audio, "device", "auto"),
                )
                self._audio_agent = AudioAgent(config=audio_config)
            await self._audio_agent.initialize()
            self._agent_status["audio"] = True
            logger.info("AudioAgent initialized on-demand")

        elif agent_type == "email":
            if self._email_agent is None:
                email_config = EmailConfig(
                    extract_attachments=getattr(self.config.email, "extract_attachments", True),
                    parse_headers=getattr(self.config.email, "parse_headers", True),
                    detect_spoofing=getattr(self.config.email, "detect_spoofing", True),
                )
                self._email_agent = EmailAgent(config=email_config)
            await self._email_agent.initialize()
            self._agent_status["email"] = True
            logger.info("EmailAgent initialized on-demand")

        elif agent_type == "video":
            if self._video_agent is None:
                video_config = VideoConfig(
                    extract_audio=getattr(self.config.video, "extract_audio", True),
                    extract_frames=getattr(self.config.video, "extract_frames", True),
                    whisper_model=getattr(self.config.video, "whisper_model", "base"),
                    scene_detection=getattr(self.config.video, "scene_detection", True),
                )
                self._video_agent = VideoAgent(config=video_config)
            await self._video_agent.initialize()
            self._agent_status["video"] = True
            logger.info("VideoAgent initialized on-demand")

        elif agent_type == "image":
            if self._image_agent is None:
                image_config = ImageConfig(
                    extract_exif=getattr(self.config.image, "extract_exif", True),
                    detect_manipulation=getattr(self.config.image, "detect_manipulation", True),
                    extract_text=getattr(self.config.image, "extract_text", True),
                )
                self._image_agent = ImageAgent(config=image_config)
            await self._image_agent.initialize()
            self._agent_status["image"] = True
            logger.info("ImageAgent initialized on-demand")

        elif agent_type == "document":
            if self._document_agent is None:
                document_config = DocumentConfig(
                    extract_metadata=getattr(self.config, "document", {}).get(
                        "extract_metadata", True
                    )
                    if hasattr(self.config, "document")
                    else True,
                    detect_hidden_content=getattr(self.config, "document", {}).get(
                        "detect_hidden_content", True
                    )
                    if hasattr(self.config, "document")
                    else True,
                    detect_tampering=getattr(self.config, "document", {}).get(
                        "detect_tampering", True
                    )
                    if hasattr(self.config, "document")
                    else True,
                    scan_macros=getattr(self.config, "document", {}).get("scan_macros", True)
                    if hasattr(self.config, "document")
                    else True,
                )
                self._document_agent = DocumentAgent(config=document_config)
            await self._document_agent.initialize()
            self._agent_status["document"] = True
            logger.info("DocumentAgent initialized on-demand")

    def _get_routing_plan(self, evidence_type: EvidenceType) -> list[str]:
        """Determine which agents should process this evidence type."""
        routing = {
            EvidenceType.TEXT: ["behavioral", "fusion"],
            EvidenceType.IMAGE: ["image", "ocr", "behavioral", "fusion"],
            EvidenceType.AUDIO: ["audio", "behavioral", "fusion"],
            EvidenceType.VIDEO: ["video", "behavioral", "fusion"],
            EvidenceType.EMAIL: ["email", "behavioral", "fusion"],
            EvidenceType.DOCUMENT: ["document", "ocr", "behavioral", "fusion"],
            EvidenceType.MIXED: ["document", "ocr", "behavioral", "fusion"],
        }
        return routing.get(evidence_type, ["behavioral", "fusion"])

    async def process(
        self, packet: EvidencePacket, skip_preprocessing: bool = False
    ) -> PipelineResult:
        """Process an evidence packet through the type-specific pipeline.

        Args:
            packet: The evidence to process
            skip_preprocessing: Skip type-specific preprocessing

        Returns:
            PipelineResult with the processed packet and metadata
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        stages_completed = []
        stage_times = {}
        errors = []
        agents_used = []

        current_packet = packet
        evidence_type = packet.evidence_type

        try:
            # Stage 1: Intake (validation)
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.INTAKE, current_packet, self._intake_stage
            )
            stage_times["intake"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.INTAKE)

            # Get routing plan for this evidence type
            routing_plan = self._get_routing_plan(evidence_type)
            logger.debug(f"Routing plan for {evidence_type}: {routing_plan}")

            # Stage 2: Type-specific preprocessing
            if not skip_preprocessing:
                stage_start = time.perf_counter()

                if "audio" in routing_plan:
                    await self._ensure_agent_initialized("audio")
                    current_packet = await self._audio_agent.process(current_packet)
                    agents_used.append("audio")

                elif "email" in routing_plan:
                    await self._ensure_agent_initialized("email")
                    current_packet = await self._email_agent.process(current_packet)
                    agents_used.append("email")

                elif "video" in routing_plan:
                    await self._ensure_agent_initialized("video")
                    current_packet = await self._video_agent.process(current_packet)
                    agents_used.append("video")

                elif "image" in routing_plan:
                    await self._ensure_agent_initialized("image")
                    current_packet = await self._image_agent.process(current_packet)
                    agents_used.append("image")

                elif "document" in routing_plan:
                    await self._ensure_agent_initialized("document")
                    current_packet = await self._document_agent.process(current_packet)
                    agents_used.append("document")

                stage_times["preprocessing"] = (time.perf_counter() - stage_start) * 1000
                stages_completed.append(PipelineStage.PREPROCESSING)

            # Stage 3: OCR (if needed for this type)
            if "ocr" in routing_plan and self._needs_ocr(current_packet):
                stage_start = time.perf_counter()
                current_packet = await self._run_stage(
                    PipelineStage.OCR, current_packet, self._ocr_stage
                )
                stage_times["ocr"] = (time.perf_counter() - stage_start) * 1000
                stages_completed.append(PipelineStage.OCR)
                agents_used.append("ocr")

            # Stage 4: Behavioral Analysis
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.BEHAVIORAL, current_packet, self._behavioral_stage
            )
            stage_times["behavioral"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.BEHAVIORAL)
            agents_used.append("behavioral")

            # Stage 5: Fusion
            stage_start = time.perf_counter()
            current_packet = await self._run_stage(
                PipelineStage.FUSION, current_packet, self._fusion_stage
            )
            stage_times["fusion"] = (time.perf_counter() - stage_start) * 1000
            stages_completed.append(PipelineStage.FUSION)
            agents_used.append("fusion")

            # Stage 6: Output
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

        # Track by type
        type_key = evidence_type.value
        if type_key not in self._metrics["by_type"]:
            self._metrics["by_type"][type_key] = {"count": 0, "time_ms": 0}
        self._metrics["by_type"][type_key]["count"] += 1
        self._metrics["by_type"][type_key]["time_ms"] += total_time

        logger.info(
            f"Pipeline {'completed' if success else 'failed'} for {packet.id} "
            f"({evidence_type.value}) in {total_time:.2f}ms, agents: {agents_used}"
        )

        return PipelineResult(
            packet=current_packet,
            success=success,
            stages_completed=stages_completed,
            total_time_ms=total_time,
            stage_times=stage_times,
            errors=errors,
            agents_used=agents_used,
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

        # Pre-initialize required agents for all packet types
        types_needed = set(p.evidence_type for p in packets)
        for evidence_type in types_needed:
            routing = self._get_routing_plan(evidence_type)
            for agent_type in ["audio", "email", "video", "image", "document"]:
                if agent_type in routing:
                    await self._ensure_agent_initialized(agent_type)

        if parallel:
            results = await asyncio.gather(
                *[self.process(p) for p in packets], return_exceptions=True
            )
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
        for hook in self._pre_hooks[stage]:
            packet = await hook(packet)

        packet = await stage_func(packet)

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
        if packet.extracted_text and len(packet.extracted_text) > 0:
            return False

        if packet.evidence_type == EvidenceType.TEXT:
            if packet.raw_content:
                try:
                    packet.raw_content.decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    pass

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

    def get_agent_status(self) -> dict[str, bool]:
        """Get initialization status of all agents."""
        return self._agent_status.copy()

    async def shutdown(self) -> None:
        """Shutdown all agents."""
        logger.info("Shutting down Evidence Pipeline...")

        shutdown_tasks = []

        if self._ocr_agent:
            shutdown_tasks.append(self._ocr_agent.shutdown())
        if self._behavioral_agent:
            shutdown_tasks.append(self._behavioral_agent.shutdown())
        if self._fusion_agent:
            shutdown_tasks.append(self._fusion_agent.shutdown())
        if self._audio_agent:
            shutdown_tasks.append(self._audio_agent.shutdown())
        if self._email_agent:
            shutdown_tasks.append(self._email_agent.shutdown())
        if self._video_agent:
            shutdown_tasks.append(self._video_agent.shutdown())
        if self._image_agent:
            shutdown_tasks.append(self._image_agent.shutdown())
        if self._document_agent:
            shutdown_tasks.append(self._document_agent.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)

        self._initialized = False
        self._agent_status.clear()
        logger.info("Evidence Pipeline shutdown complete")


# Convenience function for quick processing
async def process_evidence(
    content: bytes,
    evidence_type: EvidenceType = EvidenceType.TEXT,
    case_id: str | None = None,
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
