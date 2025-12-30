"""Evidence Suite - Entity Extraction Agent
Named Entity Recognition and relationship extraction for forensic analysis.
Based on 2024-2025 best practices with spaCy and Transformers.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from agents.base import BaseAgent
from core.models import AnalysisResult, EvidencePacket, ProcessingStage


@dataclass
class Entity:
    """Represents an extracted entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
        }


class EntityConfig:
    """Entity Agent configuration."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        use_transformers: bool = True,
        extract_pii: bool = True,
        extract_relationships: bool = True,
        enable_coreference: bool = True,
        custom_patterns: bool = True,
        device: str = "auto",
        batch_size: int = 32,
        confidence_threshold: float = 0.5,
        max_text_length: int = 1000000,
    ):
        self.spacy_model = spacy_model
        self.use_transformers = use_transformers
        self.extract_pii = extract_pii
        self.extract_relationships = extract_relationships
        self.enable_coreference = enable_coreference
        self.custom_patterns = custom_patterns
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_text_length = max_text_length


class EntityAgent(BaseAgent):
    """Agent for named entity recognition and extraction.

    Features:
    - Standard NER (persons, organizations, locations, dates)
    - Custom forensic entities (case numbers, legal citations)
    - PII detection (phone, email, SSN, credit cards)
    - Monetary amount extraction
    - Relationship extraction (communication, financial flows)
    - Coreference resolution
    - Entity deduplication and linking
    """

    # Regex patterns for custom entities
    PATTERNS = {
        "PHONE": [
            r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}",
            r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",
        ],
        "EMAIL": [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        ],
        "SSN": [
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        ],
        "CREDIT_CARD": [
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9][0-9])[0-9]{12})\b",
        ],
        "MONEY": [
            r"\$[\d,]+(?:\.\d{2})?",
            r"(?:USD|EUR|GBP|CAD)\s*[\d,]+(?:\.\d{2})?",
            r"[\d,]+(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?)",
        ],
        "CASE_NUMBER": [
            r"(?:Case|Docket|No\.?|#)\s*:?\s*\d{1,2}[-:]\d{2,4}[-:][A-Z]{2,3}[-:]?\d{1,6}",
            r"\d{1,2}:\d{2}-[a-z]{2}-\d{5}",
            r"[A-Z]{2,3}-\d{4,}-\d{4,}",
        ],
        "LEGAL_CITATION": [
            r"\d+\s+U\.?S\.?C\.?\s+§?\s*\d+",
            r"\d+\s+C\.?F\.?R\.?\s+§?\s*\d+(?:\.\d+)?",
            r"[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+,?\s*\d+\s+[A-Z]\.\s*\d+d?\s*\d*",
        ],
        "DATE": [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{4}-\d{2}-\d{2}",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
        ],
        "TIME": [
            r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?",
        ],
        "IP_ADDRESS": [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        ],
        "URL": [
            r"https?://[^\s<>\"]+",
            r"www\.[^\s<>\"]+",
        ],
    }

    # Relationship patterns (subject, verb patterns, object)
    RELATIONSHIP_PATTERNS = [
        ("PERSON", ["called", "contacted", "emailed", "texted", "messaged"], "PERSON"),
        ("PERSON", ["paid", "transferred", "wired", "sent money to"], "PERSON"),
        ("PERSON", ["works for", "employed by", "works at"], "ORG"),
        ("PERSON", ["met with", "met", "visited"], "PERSON"),
        ("ORG", ["acquired", "purchased", "bought"], "ORG"),
        ("PERSON", ["lives at", "resides at", "located at"], "GPE"),
    ]

    def __init__(self, agent_id: str | None = None, config: EntityConfig | None = None):
        self.entity_config = config or EntityConfig()
        super().__init__(
            agent_id=agent_id,
            agent_type="entity",
            config={
                "spacy_model": self.entity_config.spacy_model,
                "extract_pii": self.entity_config.extract_pii,
                "extract_relationships": self.entity_config.extract_relationships,
            },
        )
        self._nlp = None
        self._device = "cpu"
        self._spacy_available = False
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

    async def _setup(self) -> None:
        """Initialize NLP models and compile patterns."""
        # Compile regex patterns
        for entity_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[entity_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Check GPU availability
        if self.entity_config.device == "auto":
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = self.entity_config.device

        # Load spaCy model
        try:
            import spacy

            # Try transformer model first, fall back to smaller models
            models_to_try = [
                self.entity_config.spacy_model,
                "en_core_web_lg",
                "en_core_web_md",
                "en_core_web_sm",
            ]

            for model_name in models_to_try:
                try:
                    self._nlp = spacy.load(model_name)
                    self._spacy_available = True
                    logger.info(f"spaCy model '{model_name}' loaded on {self._device}")
                    break
                except OSError:
                    continue

            if not self._spacy_available:
                logger.warning("No spaCy model available. Pattern matching only.")

        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """Process evidence for entity extraction.

        Steps:
        1. Get text content
        2. Extract standard NER entities (spaCy)
        3. Extract custom forensic entities (patterns)
        4. Extract PII if enabled
        5. Resolve coreferences
        6. Extract relationships
        7. Deduplicate and link entities
        """
        text = packet.get_text_content()

        if not text:
            raise ValueError("No text content for entity extraction")

        # Truncate if too long
        if len(text) > self.entity_config.max_text_length:
            text = text[: self.entity_config.max_text_length]
            logger.warning(f"Text truncated to {self.entity_config.max_text_length} chars")

        # Extract entities
        entities: list[Entity] = []

        # 1. spaCy NER
        spacy_entities = await self._extract_spacy_entities(text)
        entities.extend(spacy_entities)

        # 2. Pattern-based extraction
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)

        # 3. PII extraction
        pii_entities = []
        if self.entity_config.extract_pii:
            pii_entities = self._extract_pii(text)
            entities.extend(pii_entities)

        # 4. Deduplicate entities
        unique_entities = self._deduplicate_entities(entities)

        # 5. Extract relationships
        relationships = []
        if self.entity_config.extract_relationships and self._spacy_available:
            relationships = await self._extract_relationships(text, unique_entities)

        # 6. Coreference resolution (group mentions)
        entity_clusters = self._cluster_entities(unique_entities)

        # Build entity summary
        entity_counts = defaultdict(int)
        for entity in unique_entities:
            entity_counts[entity.label] += 1

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=self._calculate_confidence(unique_entities),
            findings={
                "total_entities": len(unique_entities),
                "unique_entity_types": len(entity_counts),
                "entity_counts": dict(entity_counts),
                "relationship_count": len(relationships),
                "pii_detected": len(pii_entities) > 0,
                "pii_count": len(pii_entities),
                "persons": [e.text for e in unique_entities if e.label == "PERSON"][:20],
                "organizations": [e.text for e in unique_entities if e.label == "ORG"][:20],
                "locations": [e.text for e in unique_entities if e.label in ("GPE", "LOC")][:20],
                "dates": [e.text for e in unique_entities if e.label == "DATE"][:20],
                "money": [e.text for e in unique_entities if e.label == "MONEY"][:20],
            },
            raw_output={
                "entities": [e.to_dict() for e in unique_entities[:200]],
                "relationships": [r.to_dict() for r in relationships[:50]],
                "entity_clusters": entity_clusters[:50],
                "pii_summary": self._summarize_pii(pii_entities),
            },
        )

        return packet.with_updates(
            stage=ProcessingStage.BEHAVIORAL_ANALYZED,
            analysis_results=packet.analysis_results + [analysis],
            source_metadata={
                **packet.source_metadata,
                "entity_extraction": {
                    "total_entities": len(unique_entities),
                    "entity_types": list(entity_counts.keys()),
                    "has_pii": len(pii_entities) > 0,
                },
            },
        )

    async def _extract_spacy_entities(self, text: str) -> list[Entity]:
        """Extract entities using spaCy NER."""
        if not self._spacy_available:
            return []

        entities = []

        try:
            doc = self._nlp(text)

            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.9,  # spaCy doesn't provide confidence
                        metadata={"source": "spacy"},
                    )
                )

        except Exception as e:
            logger.warning(f"spaCy extraction error: {e}")

        return entities

    def _extract_pattern_entities(self, text: str) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(
                        Entity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.95,
                            metadata={"source": "pattern"},
                        )
                    )

        return entities

    def _extract_pii(self, text: str) -> list[Entity]:
        """Extract PII entities."""
        pii_types = ["PHONE", "EMAIL", "SSN", "CREDIT_CARD"]
        entities = []

        for pii_type in pii_types:
            patterns = self._compiled_patterns.get(pii_type, [])
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Validate matches
                    matched_text = match.group()

                    if pii_type == "SSN":
                        # Validate SSN format
                        if not self._validate_ssn(matched_text):
                            continue

                    if pii_type == "CREDIT_CARD":
                        # Validate with Luhn algorithm
                        if not self._validate_credit_card(matched_text):
                            continue

                    entities.append(
                        Entity(
                            text=matched_text,
                            label=f"PII_{pii_type}",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9,
                            metadata={"source": "pii_pattern", "pii_type": pii_type},
                        )
                    )

        return entities

    def _validate_ssn(self, ssn: str) -> bool:
        """Validate SSN format."""
        clean = re.sub(r"[-\s]", "", ssn)
        if len(clean) != 9:
            return False
        # Check for invalid area numbers
        area = int(clean[:3])
        return area not in {0, 666} and area < 900

    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        clean = re.sub(r"[-\s]", "", number)
        if not clean.isdigit():
            return False

        digits = [int(d) for d in clean]
        checksum = 0

        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities, preferring higher confidence."""
        seen: dict[tuple[str, str], Entity] = {}

        for entity in entities:
            key = (entity.text.lower(), entity.label)
            existing = seen.get(key)

            if existing is None or entity.confidence > existing.confidence:
                seen[key] = entity

        return list(seen.values())

    async def _extract_relationships(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """Extract relationships between entities."""
        if not self._spacy_available:
            return []

        relationships = []

        try:
            doc = self._nlp(text)

            # Create entity lookup by span
            entity_by_span: dict[tuple[int, int], Entity] = {(e.start, e.end): e for e in entities}

            # Check for relationship patterns in sentences
            for sent in doc.sents:
                sent_text = sent.text.lower()

                for source_type, verbs, target_type in self.RELATIONSHIP_PATTERNS:
                    # Find source entities
                    sources = [
                        e
                        for e in entities
                        if e.label == source_type
                        and e.start >= sent.start_char
                        and e.end <= sent.end_char
                    ]

                    # Find target entities
                    targets = [
                        e
                        for e in entities
                        if e.label == target_type
                        and e.start >= sent.start_char
                        and e.end <= sent.end_char
                    ]

                    # Check if any verb pattern matches
                    for verb in verbs:
                        if verb.lower() in sent_text:
                            for source in sources:
                                for target in targets:
                                    if source != target:
                                        relationships.append(
                                            Relationship(
                                                source=source,
                                                target=target,
                                                relation_type=verb,
                                                confidence=0.7,
                                                context=sent.text[:200],
                                            )
                                        )

        except Exception as e:
            logger.warning(f"Relationship extraction error: {e}")

        return relationships

    def _cluster_entities(self, entities: list[Entity]) -> list[dict[str, Any]]:
        """Cluster entities that likely refer to the same real-world entity."""
        clusters: list[list[Entity]] = []

        for entity in entities:
            added = False

            for cluster in clusters:
                # Check if entity matches any in cluster
                for existing in cluster:
                    if self._entities_match(entity, existing):
                        cluster.append(entity)
                        added = True
                        break

                if added:
                    break

            if not added:
                clusters.append([entity])

        # Convert to dict format
        return [
            {
                "canonical": cluster[0].text,
                "label": cluster[0].label,
                "mentions": [e.text for e in cluster],
                "count": len(cluster),
            }
            for cluster in clusters
            if len(cluster) > 1
        ]

    def _entities_match(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities likely refer to the same thing."""
        if e1.label != e2.label:
            return False

        t1 = e1.text.lower()
        t2 = e2.text.lower()

        # Exact match
        if t1 == t2:
            return True

        # One is substring of other (for names)
        if e1.label == "PERSON":
            words1 = set(t1.split())
            words2 = set(t2.split())
            # Share at least one word (last name)
            if words1 & words2:
                return True

        return False

    def _summarize_pii(self, pii_entities: list[Entity]) -> dict[str, Any]:
        """Create summary of detected PII."""
        by_type: dict[str, int] = defaultdict(int)

        for entity in pii_entities:
            pii_type = entity.metadata.get("pii_type", "unknown")
            by_type[pii_type] += 1

        return {
            "total_pii": len(pii_entities),
            "by_type": dict(by_type),
            "types_found": list(by_type.keys()),
        }

    def _calculate_confidence(self, entities: list[Entity]) -> float:
        """Calculate overall extraction confidence."""
        if not entities:
            return 0.5

        confidences = [e.confidence for e in entities]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost for variety of entity types
        types = set(e.label for e in entities)
        type_bonus = min(0.2, len(types) * 0.02)

        return min(1.0, avg_confidence + type_bonus)

    async def self_critique(self, result: AnalysisResult) -> float:
        """Self-critique entity extraction quality."""
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Boost for finding diverse entities
        entity_counts = result.findings.get("entity_counts", {})
        if len(entity_counts) >= 3:
            score = min(1.0, score + 0.1)

        # Boost for finding relationships
        if result.findings.get("relationship_count", 0) > 0:
            score = min(1.0, score + 0.05)

        return score


class PIIRedactor:
    """Utility for redacting PII from text."""

    REDACTION_MAP = {
        "PII_PHONE": "[PHONE REDACTED]",
        "PII_EMAIL": "[EMAIL REDACTED]",
        "PII_SSN": "[SSN REDACTED]",
        "PII_CREDIT_CARD": "[CC REDACTED]",
        "PERSON": "[NAME REDACTED]",
    }

    @classmethod
    def redact(cls, text: str, entities: list[Entity]) -> str:
        """Redact PII from text based on extracted entities."""
        # Sort by position (reverse to maintain indices)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        redacted = text
        for entity in sorted_entities:
            replacement = cls.REDACTION_MAP.get(entity.label, "[REDACTED]")
            redacted = redacted[: entity.start] + replacement + redacted[entity.end :]

        return redacted

    @classmethod
    def mask(cls, text: str, entities: list[Entity]) -> str:
        """Mask PII with asterisks, preserving format hints."""
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        masked = text
        for entity in sorted_entities:
            original = entity.text

            if entity.label == "PII_PHONE":
                # Keep last 4 digits
                replacement = "***-***-" + original[-4:] if len(original) >= 4 else "****"
            elif entity.label == "PII_EMAIL":
                # Keep domain
                parts = original.split("@")
                replacement = "****@" + parts[1] if len(parts) == 2 else "****"
            elif entity.label == "PII_SSN":
                replacement = "***-**-" + original[-4:] if len(original) >= 4 else "****"
            elif entity.label == "PII_CREDIT_CARD":
                replacement = "****-****-****-" + original[-4:] if len(original) >= 4 else "****"
            else:
                replacement = "*" * len(original)

            masked = masked[: entity.start] + replacement + masked[entity.end :]

        return masked
