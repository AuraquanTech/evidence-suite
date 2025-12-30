"""Evidence Suite - Core Data Models
Implements FRE 707-compliant evidence handling with chain of custody tracking.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class EvidenceType(str, Enum):
    """Types of evidence that can be processed."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    EMAIL = "email"
    MIXED = "mixed"


class ProcessingStage(str, Enum):
    """Pipeline processing stages."""

    RAW = "raw"
    OCR_PROCESSED = "ocr_processed"
    BEHAVIORAL_ANALYZED = "behavioral_analyzed"
    FUSED = "fused"
    VALIDATED = "validated"


class ChainOfCustodyEntry(BaseModel):
    """Single entry in the chain of custody log."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    action: str
    input_hash: str
    output_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_verification_string(self) -> str:
        """Create string for hash verification."""
        return f"{self.timestamp.isoformat()}|{self.agent_id}|{self.action}|{self.input_hash}|{self.output_hash}"


class ChainOfCustody(BaseModel):
    """FRE 707-compliant chain of custody tracker.
    Maintains SHA-256 hashes at each processing step.
    """

    evidence_id: str
    entries: list[ChainOfCustodyEntry] = Field(default_factory=list)

    def add_entry(
        self,
        agent_id: str,
        action: str,
        input_data: Any,
        output_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ChainOfCustodyEntry:
        """Add a new entry to the chain of custody."""
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data)

        entry = ChainOfCustodyEntry(
            agent_id=agent_id,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain."""
        if len(self.entries) < 2:
            return True

        for i in range(1, len(self.entries)):
            # Each input should match the previous output
            if self.entries[i].input_hash != self.entries[i - 1].output_hash:
                return False
        return True

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if isinstance(data, bytes):
            content = data
        elif isinstance(data, str):
            content = data.encode("utf-8")
        else:
            content = str(data).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @computed_field
    @property
    def chain_hash(self) -> str:
        """Compute hash of entire chain for verification."""
        chain_str = "|".join(e.to_verification_string() for e in self.entries)
        return hashlib.sha256(chain_str.encode()).hexdigest()


class AnalysisResult(BaseModel):
    """Result from a single agent's analysis."""

    agent_id: str
    agent_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    findings: dict[str, Any] = Field(default_factory=dict)
    raw_output: Any | None = None
    processing_time_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return len(self.errors) == 0


class BehavioralIndicators(BaseModel):
    """Behavioral analysis indicators for forensic assessment."""

    # Sentiment scores
    sentiment_compound: float = 0.0
    sentiment_positive: float = 0.0
    sentiment_negative: float = 0.0
    sentiment_neutral: float = 0.0

    # Behavioral pattern flags
    darvo_score: float = 0.0  # Deny, Attack, Reverse Victim/Offender
    gaslighting_score: float = 0.0
    manipulation_score: float = 0.0
    deception_indicators: float = 0.0

    # Linguistic markers
    hedging_frequency: float = 0.0
    certainty_markers: float = 0.0
    emotional_intensity: float = 0.0

    # Classification
    primary_behavior_class: str | None = None
    behavior_probabilities: dict[str, float] = Field(default_factory=dict)


class EvidencePacket(BaseModel):
    """Core data structure for evidence flowing through the pipeline.
    Immutable-style updates via copy-on-write pattern.
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Source data
    evidence_type: EvidenceType = EvidenceType.TEXT
    raw_content: bytes | None = None
    source_path: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    # Processing state
    stage: ProcessingStage = ProcessingStage.RAW

    # Extracted content
    extracted_text: str | None = None
    ocr_confidence: float | None = None

    # Analysis results
    behavioral_indicators: BehavioralIndicators | None = None
    analysis_results: list[AnalysisResult] = Field(default_factory=list)

    # Fusion output
    fused_score: float | None = None
    fused_classification: str | None = None
    fusion_metadata: dict[str, Any] = Field(default_factory=dict)

    # Chain of custody
    chain_of_custody: ChainOfCustody = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.chain_of_custody is None:
            self.chain_of_custody = ChainOfCustody(evidence_id=self.id)

    def with_updates(self, **kwargs) -> EvidencePacket:
        """Create a new packet with updated fields (immutable pattern)."""
        data = self.model_dump()
        data.update(kwargs)
        return EvidencePacket(**data)

    def add_analysis_result(self, result: AnalysisResult) -> EvidencePacket:
        """Add an analysis result and return updated packet."""
        new_results = self.analysis_results.copy()
        new_results.append(result)
        return self.with_updates(analysis_results=new_results)

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the raw content."""
        if self.raw_content:
            return hashlib.sha256(self.raw_content).hexdigest()
        if self.extracted_text:
            return hashlib.sha256(self.extracted_text.encode()).hexdigest()
        return hashlib.sha256(b"empty").hexdigest()

    def get_text_content(self) -> str:
        """Get the best available text content."""
        if self.extracted_text:
            return self.extracted_text
        if self.raw_content:
            try:
                return self.raw_content.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        return ""

    class Config:
        arbitrary_types_allowed = True
