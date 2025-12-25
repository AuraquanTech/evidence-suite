"""
Evidence Suite - Base Agent
Abstract base class for all agents using Ray Actor pattern.
"""
from __future__ import annotations
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar
from loguru import logger

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from core.models import EvidencePacket, AnalysisResult, ChainOfCustodyEntry


T = TypeVar('T', bound='BaseAgent')


class BaseAgent(ABC):
    """
    Abstract base class for Evidence Suite agents.

    Implements the core agent contract from the architecture:
    - process(packet) -> packet
    - self_critique(result) -> float

    Can be used as a Ray Actor when distributed processing is needed.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "base",
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.config = config or {}
        self._initialized = False
        self._metrics = {
            "packets_processed": 0,
            "total_processing_time_ms": 0.0,
            "errors": 0,
            "avg_confidence": 0.0
        }
        logger.info(f"Agent created: {self.agent_id} ({self.agent_type})")

    async def initialize(self) -> None:
        """
        Initialize agent resources (models, connections, etc).
        Override in subclasses for custom initialization.
        """
        if self._initialized:
            return
        await self._setup()
        self._initialized = True
        logger.info(f"Agent initialized: {self.agent_id}")

    async def _setup(self) -> None:
        """Override this for custom setup logic."""
        pass

    async def process(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Main processing entry point.
        Wraps _process_impl with timing, error handling, and chain of custody.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()
        input_hash = packet.content_hash

        try:
            # Run the actual processing
            result_packet = await self._process_impl(packet)

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000

            # Update chain of custody
            result_packet.chain_of_custody.add_entry(
                agent_id=self.agent_id,
                action=f"{self.agent_type}_process",
                input_data=input_hash,
                output_data=result_packet.content_hash,
                metadata={
                    "processing_time_ms": processing_time,
                    "stage": result_packet.stage.value
                }
            )

            # Update metrics
            self._update_metrics(processing_time, success=True)

            logger.debug(
                f"{self.agent_id} processed packet {packet.id} "
                f"in {processing_time:.2f}ms"
            )

            return result_packet

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_metrics(processing_time, success=False)
            logger.error(f"{self.agent_id} error processing {packet.id}: {e}")

            # Add error to packet and return
            error_result = AnalysisResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=0.0,
                errors=[str(e)]
            )
            return packet.add_analysis_result(error_result)

    @abstractmethod
    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Implement the actual processing logic in subclasses.
        """
        raise NotImplementedError

    async def self_critique(self, result: AnalysisResult) -> float:
        """
        Self-evaluate the quality of the analysis.
        Returns a score from 0.0 to 1.0.

        Default implementation uses confidence as the critique score.
        Override for more sophisticated self-evaluation.
        """
        if not result.is_successful:
            return 0.0

        # Base critique: confidence weighted by presence of findings
        base_score = result.confidence

        # Bonus for having detailed findings
        if result.findings:
            finding_bonus = min(0.1, len(result.findings) * 0.02)
            base_score = min(1.0, base_score + finding_bonus)

        return base_score

    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update internal metrics."""
        self._metrics["packets_processed"] += 1
        self._metrics["total_processing_time_ms"] += processing_time
        if not success:
            self._metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        metrics = self._metrics.copy()
        if metrics["packets_processed"] > 0:
            metrics["avg_processing_time_ms"] = (
                metrics["total_processing_time_ms"] /
                metrics["packets_processed"]
            )
        return metrics

    async def shutdown(self) -> None:
        """Clean up resources. Override for custom cleanup."""
        logger.info(f"Agent shutting down: {self.agent_id}")
        self._initialized = False


def as_ray_actor(agent_class: type[T], **ray_options) -> type:
    """
    Decorator to convert an agent class to a Ray Actor.

    Usage:
        @as_ray_actor
        class MyAgent(BaseAgent):
            ...

    Or with options:
        @as_ray_actor(num_cpus=2, num_gpus=1)
        class MyAgent(BaseAgent):
            ...
    """
    if not RAY_AVAILABLE:
        logger.warning("Ray not available, returning original class")
        return agent_class

    return ray.remote(**ray_options)(agent_class)


# Convenience function to create Ray actor from agent instance
def make_remote(agent: BaseAgent, **ray_options) -> Any:
    """Create a Ray remote actor from an agent instance."""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not installed")

    @ray.remote(**ray_options)
    class RemoteAgent(agent.__class__):
        pass

    return RemoteAgent.remote(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type,
        config=agent.config
    )
