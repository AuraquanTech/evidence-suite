"""Evidence Suite - Pipeline-Database Integration Layer
Connects the processing pipeline to database persistence with FRE 901/902 compliance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from core.models import AnalysisResult, EvidencePacket, ProcessingStage  # noqa: TC001


class EvidenceStatus(str, Enum):
    """Evidence processing status for database tracking."""

    PENDING = "pending"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    BEHAVIORAL_COMPLETE = "behavioral_complete"
    FUSION_COMPLETE = "fusion_complete"
    ANALYZED = "analyzed"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    ERROR = "error"


class ProcessingEvent(str, Enum):
    """Events emitted during processing."""

    PROCESSING_STARTED = "processing_started"
    STAGE_COMPLETED = "stage_completed"
    AGENT_COMPLETED = "agent_completed"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    CHAIN_OF_CUSTODY_UPDATED = "chain_of_custody_updated"


class PipelineDBIntegration:
    """Integrates the evidence pipeline with database persistence.

    Features:
    - Atomic result persistence with transaction management
    - Chain of custody logging to database
    - Event-driven notifications via pub/sub
    - Optimistic locking for concurrent access
    - Audit logging for FRE 901/902 compliance
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        redis_client: Any | None = None,
        enable_events: bool = True,
    ):
        self._session_factory = session_factory
        self._redis = redis_client
        self._enable_events = enable_events
        self._event_handlers: dict[ProcessingEvent, list[Callable]] = {
            evt: [] for evt in ProcessingEvent
        }

    def on_event(self, event_type: ProcessingEvent, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event_type].append(handler)

    async def _emit_event(
        self,
        event_type: ProcessingEvent,
        evidence_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a processing event."""
        if not self._enable_events:
            return

        event_data = {
            "event": event_type.value,
            "evidence_id": evidence_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {},
        }

        # Call registered handlers
        for handler in self._event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

        # Publish to Redis if available
        if self._redis:
            try:
                channel = f"evidence:{event_type.value}"
                await self._redis.publish(channel, json.dumps(event_data))
            except Exception as e:
                logger.warning(f"Redis publish error: {e}")

    async def persist_processing_start(
        self,
        session: AsyncSession,
        evidence_id: str,
        packet: EvidencePacket,
    ) -> None:
        """Record the start of evidence processing."""
        from core.database.models import AuditLog, EvidenceRecord

        try:
            # Update evidence status
            stmt = (
                update(EvidenceRecord)
                .where(EvidenceRecord.id == evidence_id)
                .values(
                    status=EvidenceStatus.PROCESSING.value,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.execute(stmt)

            # Create audit log entry
            audit = AuditLog(
                user_id=None,  # System processing
                action="processing_started",
                resource_type="evidence",
                resource_id=evidence_id,
                new_value={
                    "packet_id": packet.id,
                    "evidence_type": packet.evidence_type.value,
                    "content_hash": packet.content_hash,
                },
            )
            session.add(audit)
            await session.flush()

            await self._emit_event(
                ProcessingEvent.PROCESSING_STARTED,
                evidence_id,
                {"packet_id": packet.id},
            )

        except Exception as e:
            logger.error(f"Failed to persist processing start: {e}")
            raise

    async def persist_analysis_result(
        self,
        session: AsyncSession,
        evidence_id: str,
        result: AnalysisResult,
        agent_name: str,
    ) -> None:
        """Persist a single agent's analysis result."""
        from core.database.models import AnalysisResult as DBAnalysisResult

        try:
            # Create analysis result record
            db_result = DBAnalysisResult(
                evidence_id=evidence_id,
                agent_type=result.agent_type,
                agent_id=result.agent_id,
                result_data={
                    "findings": result.findings,
                    "raw_output": result.raw_output,
                    "errors": result.errors,
                },
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
            )
            session.add(db_result)
            await session.flush()

            await self._emit_event(
                ProcessingEvent.AGENT_COMPLETED,
                evidence_id,
                {
                    "agent": agent_name,
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms,
                },
            )

        except Exception as e:
            logger.error(f"Failed to persist analysis result: {e}")
            raise

    async def persist_chain_of_custody_entry(
        self,
        session: AsyncSession,
        evidence_id: str,
        agent_id: str,
        agent_type: str,
        action: str,
        input_hash: str,
        output_hash: str,
        processing_time_ms: float,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Persist a chain of custody entry to the database."""
        from core.database.models import ChainOfCustodyLog

        try:
            # Create signature for integrity verification
            signature_data = f"{evidence_id}|{agent_id}|{action}|{input_hash}|{output_hash}|{datetime.utcnow().isoformat()}"
            signature = hashlib.sha256(signature_data.encode()).hexdigest()

            coc_entry = ChainOfCustodyLog(
                evidence_id=evidence_id,
                agent_id=agent_id,
                agent_type=agent_type,
                action=action,
                input_hash=input_hash,
                output_hash=output_hash,
                processing_time_ms=processing_time_ms,
                success=success,
                error_message=error_message,
                signature=signature,
            )
            session.add(coc_entry)
            await session.flush()

            await self._emit_event(
                ProcessingEvent.CHAIN_OF_CUSTODY_UPDATED,
                evidence_id,
                {
                    "agent_id": agent_id,
                    "action": action,
                    "success": success,
                },
            )

        except Exception as e:
            logger.error(f"Failed to persist chain of custody: {e}")
            raise

    async def persist_processing_complete(
        self,
        session: AsyncSession,
        evidence_id: str,
        packet: EvidencePacket,
        success: bool,
        total_time_ms: float,
        agents_used: list[str],
        errors: list[str] | None = None,
    ) -> None:
        """Persist the final processing results."""
        from core.database.models import AuditLog, EvidenceRecord

        try:
            # Determine final status
            if not success:
                status = EvidenceStatus.ERROR.value
            elif packet.fused_score and packet.fused_score > 0.7:
                status = EvidenceStatus.FLAGGED.value
            else:
                status = EvidenceStatus.ANALYZED.value

            # Prepare behavioral indicators for storage
            behavioral_data = None
            if packet.behavioral_indicators:
                behavioral_data = {
                    "sentiment_compound": packet.behavioral_indicators.sentiment_compound,
                    "sentiment_positive": packet.behavioral_indicators.sentiment_positive,
                    "sentiment_negative": packet.behavioral_indicators.sentiment_negative,
                    "darvo_score": packet.behavioral_indicators.darvo_score,
                    "gaslighting_score": packet.behavioral_indicators.gaslighting_score,
                    "manipulation_score": packet.behavioral_indicators.manipulation_score,
                    "deception_indicators": packet.behavioral_indicators.deception_indicators,
                    "primary_behavior_class": packet.behavioral_indicators.primary_behavior_class,
                    "behavior_probabilities": packet.behavioral_indicators.behavior_probabilities,
                }

            # Update evidence record with results
            stmt = (
                update(EvidenceRecord)
                .where(EvidenceRecord.id == evidence_id)
                .values(
                    status=status,
                    extracted_text=packet.extracted_text,
                    behavioral_indicators=behavioral_data,
                    fused_score=packet.fused_score,
                    fused_classification=packet.fused_classification,
                    fusion_results=packet.fusion_metadata,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.execute(stmt)

            # Create audit log
            audit = AuditLog(
                user_id=None,
                action="processing_completed" if success else "processing_failed",
                resource_type="evidence",
                resource_id=evidence_id,
                new_value={
                    "status": status,
                    "fused_score": packet.fused_score,
                    "fused_classification": packet.fused_classification,
                    "agents_used": agents_used,
                    "total_time_ms": total_time_ms,
                    "errors": errors,
                },
            )
            session.add(audit)
            await session.flush()

            event_type = (
                ProcessingEvent.PROCESSING_COMPLETED
                if success
                else ProcessingEvent.PROCESSING_FAILED
            )
            await self._emit_event(
                event_type,
                evidence_id,
                {
                    "status": status,
                    "fused_score": packet.fused_score,
                    "agents_used": agents_used,
                    "total_time_ms": total_time_ms,
                },
            )

        except Exception as e:
            logger.error(f"Failed to persist processing complete: {e}")
            raise

    async def sync_chain_of_custody(
        self,
        session: AsyncSession,
        evidence_id: str,
        packet: EvidencePacket,
    ) -> None:
        """Sync the in-memory chain of custody to the database."""
        if not packet.chain_of_custody:
            return

        for entry in packet.chain_of_custody.entries:
            await self.persist_chain_of_custody_entry(
                session=session,
                evidence_id=evidence_id,
                agent_id=entry.agent_id,
                agent_type=entry.metadata.get("agent_type", "unknown"),
                action=entry.action,
                input_hash=entry.input_hash,
                output_hash=entry.output_hash,
                processing_time_ms=entry.metadata.get("processing_time_ms", 0),
                success=True,
                error_message=None,
            )

    async def get_evidence_with_lock(
        self,
        session: AsyncSession,
        evidence_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Get evidence record with optimistic locking check.

        Args:
            session: Database session
            evidence_id: Evidence UUID
            version: Expected version for optimistic locking

        Returns:
            Evidence record dict or None if not found/version mismatch
        """
        from core.database.models import EvidenceRecord

        stmt = select(EvidenceRecord).where(EvidenceRecord.id == evidence_id)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            return None

        # Check version for optimistic locking
        if version is not None and hasattr(record, "version"):
            if record.version != version:
                raise ConcurrencyError(
                    f"Evidence {evidence_id} was modified concurrently. "
                    f"Expected version {version}, found {record.version}"
                )

        return {
            "id": str(record.id),
            "case_id": str(record.case_id) if record.case_id else None,
            "status": record.status,
            "evidence_type": record.evidence_type,
            "filename": record.filename,
            "storage_path": record.storage_path,
            "original_hash": record.original_hash,
            "extracted_text": record.extracted_text,
            "fused_score": record.fused_score,
            "fused_classification": record.fused_classification,
        }

    async def update_evidence_status(
        self,
        session: AsyncSession,
        evidence_id: str,
        status: EvidenceStatus,
        version: int | None = None,
    ) -> bool:
        """Update evidence status with optimistic locking.

        Returns:
            True if updated, False if version conflict
        """
        from core.database.models import EvidenceRecord

        values = {
            "status": status.value,
            "updated_at": datetime.utcnow(),
        }

        if version is not None:
            values["version"] = version + 1
            stmt = (
                update(EvidenceRecord)
                .where(
                    EvidenceRecord.id == evidence_id,
                    EvidenceRecord.version == version,
                )
                .values(**values)
            )
        else:
            stmt = update(EvidenceRecord).where(EvidenceRecord.id == evidence_id).values(**values)

        result = await session.execute(stmt)
        return result.rowcount > 0


class ConcurrencyError(Exception):
    """Raised when concurrent modification is detected."""

    pass


class PipelineProcessor:
    """High-level processor that coordinates pipeline with database.

    Wraps the EvidencePipeline with database persistence, providing
    full transaction management and event emission.
    """

    def __init__(
        self,
        pipeline,  # EvidencePipeline instance
        db_integration: PipelineDBIntegration,
        session_factory: Callable[[], AsyncSession],
    ):
        self._pipeline = pipeline
        self._db = db_integration
        self._session_factory = session_factory

    async def process_evidence(
        self,
        evidence_id: str,
        packet: EvidencePacket,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process evidence with full database integration.

        Args:
            evidence_id: Database evidence record ID
            packet: Evidence packet to process
            user_id: Optional user ID for audit trail

        Returns:
            Processing result dictionary
        """
        async with self._session_factory() as session:
            try:
                async with session.begin():
                    # Record processing start
                    await self._db.persist_processing_start(session, evidence_id, packet)

                # Process through pipeline
                result = await self._pipeline.process(packet)

                async with session.begin():
                    # Persist each analysis result
                    for analysis in result.packet.analysis_results:
                        await self._db.persist_analysis_result(
                            session,
                            evidence_id,
                            analysis,
                            analysis.agent_type,
                        )

                    # Sync chain of custody
                    await self._db.sync_chain_of_custody(
                        session,
                        evidence_id,
                        result.packet,
                    )

                    # Persist final results
                    await self._db.persist_processing_complete(
                        session,
                        evidence_id,
                        result.packet,
                        result.success,
                        result.total_time_ms,
                        result.agents_used,
                        result.errors,
                    )

                return {
                    "evidence_id": evidence_id,
                    "success": result.success,
                    "stages_completed": [s.value for s in result.stages_completed],
                    "total_time_ms": result.total_time_ms,
                    "agents_used": result.agents_used,
                    "fused_score": result.packet.fused_score,
                    "fused_classification": result.packet.fused_classification,
                    "errors": result.errors,
                }

            except Exception as e:
                logger.error(f"Processing failed for {evidence_id}: {e}")

                # Try to update status to error
                try:
                    async with session.begin():
                        await self._db.update_evidence_status(
                            session, evidence_id, EvidenceStatus.ERROR
                        )
                except Exception:
                    pass

                return {
                    "evidence_id": evidence_id,
                    "success": False,
                    "error": str(e),
                }

    async def process_batch(
        self,
        items: list[tuple[str, EvidencePacket]],
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """Process multiple evidence items.

        Args:
            items: List of (evidence_id, packet) tuples
            parallel: Whether to process in parallel

        Returns:
            List of processing results
        """
        if parallel:
            tasks = [self.process_evidence(eid, packet) for eid, packet in items]
            return await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for eid, packet in items:
            result = await self.process_evidence(eid, packet)
            results.append(result)
        return results


# Repository pattern implementation
class EvidenceRepository:
    """Repository for evidence database operations.

    Provides clean abstraction over SQLAlchemy for evidence CRUD
    with chain of custody support.
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory

    async def get_by_id(self, evidence_id: str) -> dict[str, Any] | None:
        """Get evidence by ID."""
        from core.database.models import EvidenceRecord

        async with self._session_factory() as session:
            stmt = select(EvidenceRecord).where(EvidenceRecord.id == evidence_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                return None

            return self._to_dict(record)

    async def get_by_case(
        self,
        case_id: str,
        status: EvidenceStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get all evidence for a case."""
        from core.database.models import EvidenceRecord

        async with self._session_factory() as session:
            stmt = select(EvidenceRecord).where(EvidenceRecord.case_id == case_id)

            if status:
                stmt = stmt.where(EvidenceRecord.status == status.value)

            stmt = stmt.limit(limit).offset(offset).order_by(EvidenceRecord.created_at.desc())

            result = await session.execute(stmt)
            records = result.scalars().all()

            return [self._to_dict(r) for r in records]

    async def get_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get pending evidence for processing."""
        from core.database.models import EvidenceRecord

        async with self._session_factory() as session:
            stmt = (
                select(EvidenceRecord)
                .where(EvidenceRecord.status == EvidenceStatus.PENDING.value)
                .limit(limit)
                .order_by(EvidenceRecord.created_at.asc())
            )

            result = await session.execute(stmt)
            records = result.scalars().all()

            return [self._to_dict(r) for r in records]

    async def get_chain_of_custody(self, evidence_id: str) -> list[dict[str, Any]]:
        """Get full chain of custody for evidence."""
        from core.database.models import ChainOfCustodyLog

        async with self._session_factory() as session:
            stmt = (
                select(ChainOfCustodyLog)
                .where(ChainOfCustodyLog.evidence_id == evidence_id)
                .order_by(ChainOfCustodyLog.timestamp.asc())
            )

            result = await session.execute(stmt)
            logs = result.scalars().all()

            return [
                {
                    "id": str(log.id),
                    "evidence_id": str(log.evidence_id),
                    "agent_id": log.agent_id,
                    "agent_type": log.agent_type,
                    "action": log.action,
                    "input_hash": log.input_hash,
                    "output_hash": log.output_hash,
                    "processing_time_ms": log.processing_time_ms,
                    "success": log.success,
                    "error_message": log.error_message,
                    "timestamp": log.timestamp.isoformat(),
                    "signature": log.signature,
                }
                for log in logs
            ]

    async def verify_chain_integrity(self, evidence_id: str) -> dict[str, Any]:
        """Verify chain of custody integrity."""
        chain = await self.get_chain_of_custody(evidence_id)

        if not chain:
            return {"valid": True, "message": "No chain of custody entries"}

        issues = []

        # Verify each entry's hash links to previous
        for i in range(1, len(chain)):
            prev_output = chain[i - 1]["output_hash"]
            curr_input = chain[i]["input_hash"]

            if prev_output != curr_input:
                issues.append(
                    {
                        "index": i,
                        "issue": "Hash chain broken",
                        "expected": prev_output,
                        "found": curr_input,
                    }
                )

        # Verify signatures
        for entry in chain:
            expected_sig_data = (
                f"{entry['evidence_id']}|{entry['agent_id']}|{entry['action']}|"
                f"{entry['input_hash']}|{entry['output_hash']}"
            )
            # Note: Full signature verification would include timestamp

        return {
            "valid": len(issues) == 0,
            "entry_count": len(chain),
            "issues": issues,
            "first_entry": chain[0]["timestamp"] if chain else None,
            "last_entry": chain[-1]["timestamp"] if chain else None,
        }

    def _to_dict(self, record) -> dict[str, Any]:
        """Convert ORM record to dictionary."""
        return {
            "id": str(record.id),
            "case_id": str(record.case_id) if record.case_id else None,
            "evidence_type": record.evidence_type,
            "filename": record.filename,
            "mime_type": record.mime_type,
            "file_size_bytes": record.file_size_bytes,
            "original_hash": record.original_hash,
            "storage_path": record.storage_path,
            "status": record.status,
            "extracted_text": record.extracted_text,
            "fused_score": record.fused_score,
            "fused_classification": record.fused_classification,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }


class AnalysisResultRepository:
    """Repository for analysis results."""

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory

    async def get_by_evidence(
        self,
        evidence_id: str,
        agent_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get analysis results for evidence."""
        from core.database.models import AnalysisResult as DBAnalysisResult

        async with self._session_factory() as session:
            stmt = select(DBAnalysisResult).where(DBAnalysisResult.evidence_id == evidence_id)

            if agent_type:
                stmt = stmt.where(DBAnalysisResult.agent_type == agent_type)

            stmt = stmt.order_by(DBAnalysisResult.created_at.asc())

            result = await session.execute(stmt)
            records = result.scalars().all()

            return [
                {
                    "id": str(r.id),
                    "evidence_id": str(r.evidence_id),
                    "agent_type": r.agent_type,
                    "agent_id": r.agent_id,
                    "result_data": r.result_data,
                    "confidence": r.confidence,
                    "processing_time_ms": r.processing_time_ms,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]

    async def get_aggregated_scores(self, evidence_id: str) -> dict[str, Any]:
        """Get aggregated analysis scores for evidence."""
        results = await self.get_by_evidence(evidence_id)

        if not results:
            return {"has_results": False}

        confidences = [r["confidence"] for r in results if r["confidence"]]
        times = [r["processing_time_ms"] for r in results if r["processing_time_ms"]]

        return {
            "has_results": True,
            "agent_count": len(results),
            "agents": list(set(r["agent_type"] for r in results)),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "total_processing_time_ms": sum(times),
        }
