"""Evidence Suite - Test Factories
Factory functions for creating test data consistently.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from core.database import (
    AnalysisJob,
    AnalysisResult,
    Case,
    CaseStatus,
    ChainOfCustodyLog,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceTypeDB,
    User,
)


class UserFactory:
    """Factory for creating User instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        email: str | None = None,
        password_hash: str = "$2b$12$test.hash.for.testing",  # noqa: S107
        name: str | None = None,
        role: str = "analyst",
        is_active: bool = True,
        **kwargs,
    ) -> User:
        """Create a User instance."""
        cls._counter += 1
        return User(
            id=kwargs.get("id", uuid4()),
            email=email or f"user{cls._counter}@example.com",
            password_hash=password_hash,
            name=name or f"Test User {cls._counter}",
            role=role,
            is_active=is_active,
            created_at=kwargs.get("created_at", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "created_at"]},
        )

    @classmethod
    def create_admin(cls, **kwargs) -> User:
        """Create an admin user."""
        return cls.create(role="admin", **kwargs)

    @classmethod
    def create_analyst(cls, **kwargs) -> User:
        """Create an analyst user."""
        return cls.create(role="analyst", **kwargs)

    @classmethod
    def create_viewer(cls, **kwargs) -> User:
        """Create a viewer user."""
        return cls.create(role="viewer", **kwargs)


class CaseFactory:
    """Factory for creating Case instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        case_number: str | None = None,
        title: str | None = None,
        description: str | None = None,
        status: CaseStatus = CaseStatus.ACTIVE,
        client_name: str | None = None,
        attorney_name: str | None = None,
        jurisdiction: str | None = None,
        **kwargs,
    ) -> Case:
        """Create a Case instance."""
        cls._counter += 1
        return Case(
            id=kwargs.get("id", uuid4()),
            case_number=case_number or f"CASE-{cls._counter:04d}",
            title=title or f"Test Case {cls._counter}",
            description=description or f"Test case description {cls._counter}",
            status=status,
            client_name=client_name or f"Client {cls._counter}",
            attorney_name=attorney_name,
            jurisdiction=jurisdiction,
            created_at=kwargs.get("created_at", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "created_at"]},
        )


class EvidenceFactory:
    """Factory for creating EvidenceRecord instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        case_id: UUID | None = None,
        evidence_type: EvidenceTypeDB = EvidenceTypeDB.DOCUMENT,
        original_filename: str | None = None,
        mime_type: str = "application/pdf",
        file_size_bytes: int = 1024,
        original_hash: str | None = None,
        storage_path: str | None = None,
        status: EvidenceStatus = EvidenceStatus.PENDING,
        **kwargs,
    ) -> EvidenceRecord:
        """Create an EvidenceRecord instance."""
        cls._counter += 1
        evidence_id = kwargs.get("id", uuid4())
        return EvidenceRecord(
            id=evidence_id,
            case_id=case_id or uuid4(),
            evidence_type=evidence_type,
            original_filename=original_filename or f"evidence_{cls._counter}.pdf",
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            original_hash=original_hash or hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
            storage_path=storage_path
            or f"./evidence_store/{evidence_id}/{original_filename or 'file.pdf'}",
            status=status,
            created_at=kwargs.get("created_at", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "created_at"]},
        )

    @classmethod
    def create_document(cls, **kwargs) -> EvidenceRecord:
        """Create a document evidence record."""
        return cls.create(
            evidence_type=EvidenceTypeDB.DOCUMENT,
            mime_type="application/pdf",
            original_filename=kwargs.pop("original_filename", None) or "document.pdf",
            **kwargs,
        )

    @classmethod
    def create_image(cls, **kwargs) -> EvidenceRecord:
        """Create an image evidence record."""
        return cls.create(
            evidence_type=EvidenceTypeDB.IMAGE,
            mime_type="image/jpeg",
            original_filename=kwargs.pop("original_filename", None) or "image.jpg",
            **kwargs,
        )

    @classmethod
    def create_audio(cls, **kwargs) -> EvidenceRecord:
        """Create an audio evidence record."""
        return cls.create(
            evidence_type=EvidenceTypeDB.AUDIO,
            mime_type="audio/mpeg",
            original_filename=kwargs.pop("original_filename", None) or "audio.mp3",
            **kwargs,
        )

    @classmethod
    def create_video(cls, **kwargs) -> EvidenceRecord:
        """Create a video evidence record."""
        return cls.create(
            evidence_type=EvidenceTypeDB.VIDEO,
            mime_type="video/mp4",
            original_filename=kwargs.pop("original_filename", None) or "video.mp4",
            **kwargs,
        )


class ChainOfCustodyFactory:
    """Factory for creating ChainOfCustodyLog instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        evidence_id: UUID | None = None,
        agent_id: str = "test-agent",
        agent_type: str = "test",
        action: str = "test_action",
        input_hash: str | None = None,
        output_hash: str | None = None,
        processing_time_ms: int | None = None,
        success: bool = True,
        error_message: str | None = None,
        **kwargs,
    ) -> ChainOfCustodyLog:
        """Create a ChainOfCustodyLog instance."""
        cls._counter += 1
        return ChainOfCustodyLog(
            id=kwargs.get("id", cls._counter),
            evidence_id=evidence_id or uuid4(),
            agent_id=agent_id,
            agent_type=agent_type,
            action=action,
            input_hash=input_hash or hashlib.sha256(b"input").hexdigest(),
            output_hash=output_hash or hashlib.sha256(b"output").hexdigest(),
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message,
            timestamp=kwargs.get("timestamp", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "timestamp"]},
        )


class AnalysisJobFactory:
    """Factory for creating AnalysisJob instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        evidence_id: UUID | None = None,
        status: str = "pending",
        current_stage: str | None = None,
        progress_percent: int = 0,
        **kwargs,
    ) -> AnalysisJob:
        """Create an AnalysisJob instance."""
        cls._counter += 1
        return AnalysisJob(
            id=kwargs.get("id", uuid4()),
            evidence_id=evidence_id or uuid4(),
            status=status,
            current_stage=current_stage,
            progress_percent=progress_percent,
            created_at=kwargs.get("created_at", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "created_at"]},
        )


class AnalysisResultFactory:
    """Factory for creating AnalysisResult instances."""

    _counter = 0

    @classmethod
    def create(
        cls,
        evidence_id: UUID | None = None,
        agent_type: str = "behavioral",
        agent_id: str = "test-behavioral-agent",
        result_data: dict[str, Any] | None = None,
        confidence: float = 0.85,
        processing_time_ms: int = 1000,
        **kwargs,
    ) -> AnalysisResult:
        """Create an AnalysisResult instance."""
        cls._counter += 1
        return AnalysisResult(
            id=kwargs.get("id", uuid4()),
            evidence_id=evidence_id or uuid4(),
            agent_type=agent_type,
            agent_id=agent_id,
            result_data=result_data or {"test": "data"},
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            created_at=kwargs.get("created_at", datetime.utcnow()),
            **{k: v for k, v in kwargs.items() if k not in ["id", "created_at"]},
        )


# Convenience functions
def create_test_case_with_evidence(
    evidence_count: int = 3,
    analyzed: int = 0,
) -> tuple[Case, list[EvidenceRecord]]:
    """Create a case with associated evidence records.

    Args:
        evidence_count: Number of evidence records to create
        analyzed: Number of evidence records to mark as analyzed

    Returns:
        (case, evidence_list)
    """
    case = CaseFactory.create()
    evidence = []

    for i in range(evidence_count):
        status = EvidenceStatus.ANALYZED if i < analyzed else EvidenceStatus.PENDING
        e = EvidenceFactory.create(case_id=case.id, status=status)
        evidence.append(e)

    return case, evidence


def create_complete_analysis_chain(
    case_id: UUID | None = None,
) -> dict[str, Any]:
    """Create a complete analysis chain for testing.

    Returns dict with:
    - case: Case instance
    - evidence: EvidenceRecord instance
    - custody_logs: List of ChainOfCustodyLog instances
    - analysis_results: List of AnalysisResult instances
    - job: AnalysisJob instance
    """
    case = CaseFactory.create(id=case_id) if case_id else CaseFactory.create()
    evidence = EvidenceFactory.create(case_id=case.id, status=EvidenceStatus.ANALYZED)

    # Create chain of custody entries
    custody_logs = [
        ChainOfCustodyFactory.create(
            evidence_id=evidence.id,
            agent_id="api-upload",
            agent_type="http",
            action="evidence_uploaded",
            input_hash=evidence.original_hash,
            output_hash=evidence.original_hash,
        ),
        ChainOfCustodyFactory.create(
            evidence_id=evidence.id,
            agent_id="ocr-agent-v1",
            agent_type="ocr",
            action="text_extraction",
            input_hash=evidence.original_hash,
            processing_time_ms=500,
        ),
        ChainOfCustodyFactory.create(
            evidence_id=evidence.id,
            agent_id="behavioral-agent-v1",
            agent_type="behavioral",
            action="behavioral_analysis",
            processing_time_ms=1200,
        ),
    ]

    # Create analysis results
    analysis_results = [
        AnalysisResultFactory.create(
            evidence_id=evidence.id,
            agent_type="ocr",
            agent_id="ocr-agent-v1",
            result_data={"text": "Sample extracted text", "confidence": 0.95},
            confidence=0.95,
            processing_time_ms=500,
        ),
        AnalysisResultFactory.create(
            evidence_id=evidence.id,
            agent_type="behavioral",
            agent_id="behavioral-agent-v1",
            result_data={
                "indicators": ["gaslighting", "darvo"],
                "severity": "high",
            },
            confidence=0.87,
            processing_time_ms=1200,
        ),
    ]

    job = AnalysisJobFactory.create(
        evidence_id=evidence.id,
        status="completed",
        current_stage="complete",
        progress_percent=100,
    )

    return {
        "case": case,
        "evidence": evidence,
        "custody_logs": custody_logs,
        "analysis_results": analysis_results,
        "job": job,
    }
