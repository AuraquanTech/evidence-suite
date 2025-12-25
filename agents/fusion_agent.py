"""
Evidence Suite - Fusion Agent
Hybrid late fusion of multi-modal analysis results.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np

from core.models import (
    EvidencePacket,
    AnalysisResult,
    ProcessingStage,
    BehavioralIndicators
)
from core.config import FusionConfig, default_config
from agents.base import BaseAgent


class FusionAgent(BaseAgent):
    """
    Fusion layer agent for combining multi-modal analysis results.

    Features:
    - Hybrid late fusion strategy
    - Confidence-weighted aggregation
    - Cross-modal consistency validation
    - Ensemble decision making
    - Anomaly detection via disagreement analysis
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[FusionConfig] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="fusion",
            config=(config or default_config.fusion).model_dump()
        )
        self.fusion_config = config or default_config.fusion

    async def _setup(self) -> None:
        """Initialize fusion weights and thresholds."""
        logger.info(
            f"Fusion agent initialized with strategy: "
            f"{self.fusion_config.strategy}"
        )

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Fuse all analysis results into a unified assessment.
        """
        if not packet.analysis_results:
            raise ValueError("No analysis results to fuse")

        # Extract results by agent type
        results_by_type = self._group_results_by_type(packet.analysis_results)

        # Run fusion based on strategy
        if self.fusion_config.strategy == "hybrid_late":
            fused = self._hybrid_late_fusion(packet, results_by_type)
        elif self.fusion_config.strategy == "weighted_average":
            fused = self._weighted_average_fusion(results_by_type)
        elif self.fusion_config.strategy == "attention":
            fused = self._attention_fusion(results_by_type)
        else:
            fused = self._weighted_average_fusion(results_by_type)

        # Validate cross-modal consistency
        consistency = self._check_cross_modal_consistency(results_by_type)

        # Detect anomalies (high disagreement)
        anomalies = self._detect_anomalies(results_by_type)

        # Calculate final confidence
        confidence = self._calculate_fusion_confidence(
            fused, consistency, results_by_type
        )

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            findings={
                "fused_score": fused["score"],
                "fused_classification": fused["classification"],
                "consistency_score": consistency["score"],
                "consistency_details": consistency["details"],
                "anomalies": anomalies,
                "agents_fused": list(results_by_type.keys()),
                "fusion_strategy": self.fusion_config.strategy
            }
        )

        return packet.with_updates(
            fused_score=fused["score"],
            fused_classification=fused["classification"],
            fusion_metadata={
                "consistency": consistency,
                "anomalies": anomalies,
                "weights_used": fused.get("weights", {})
            },
            stage=ProcessingStage.FUSED,
            analysis_results=packet.analysis_results + [analysis]
        )

    def _group_results_by_type(
        self,
        results: List[AnalysisResult]
    ) -> Dict[str, List[AnalysisResult]]:
        """Group analysis results by agent type."""
        grouped = {}
        for result in results:
            if result.agent_type not in grouped:
                grouped[result.agent_type] = []
            grouped[result.agent_type].append(result)
        return grouped

    def _hybrid_late_fusion(
        self,
        packet: EvidencePacket,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """
        Hybrid late fusion strategy.

        Combines:
        1. OCR confidence as quality gate
        2. Behavioral scores with confidence weighting
        3. Cross-modal boosting when modalities agree
        """
        alpha = self.fusion_config.alpha

        # Extract key scores
        ocr_confidence = packet.ocr_confidence or 0.5
        behavioral = packet.behavioral_indicators

        if behavioral is None:
            return {
                "score": ocr_confidence * 0.5,
                "classification": "insufficient_data",
                "weights": {"ocr": 1.0}
            }

        # Calculate behavioral severity score
        behavioral_severity = self._calculate_behavioral_severity(behavioral)

        # Fuse with alpha weighting
        # Primary weight on behavioral, modulated by OCR quality
        ocr_quality_factor = min(1.0, ocr_confidence + 0.2)

        fused_score = (
            alpha * behavioral_severity * ocr_quality_factor +
            (1 - alpha) * self._aggregate_result_confidences(results_by_type)
        )

        # Determine classification
        classification = self._determine_classification(
            behavioral, fused_score
        )

        return {
            "score": fused_score,
            "classification": classification,
            "weights": {
                "behavioral": alpha * ocr_quality_factor,
                "confidence": 1 - alpha
            }
        }

    def _weighted_average_fusion(
        self,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """
        Simple weighted average fusion based on confidence scores.
        """
        weights = []
        scores = []

        for agent_type, results in results_by_type.items():
            for result in results:
                if result.is_successful:
                    weights.append(result.confidence)
                    scores.append(result.confidence)

        if not weights:
            return {"score": 0.0, "classification": "no_data", "weights": {}}

        # Weighted average
        weights = np.array(weights)
        scores = np.array(scores)

        if weights.sum() > 0:
            fused_score = np.average(scores, weights=weights)
        else:
            fused_score = np.mean(scores)

        return {
            "score": float(fused_score),
            "classification": self._score_to_classification(fused_score),
            "weights": {t: 1.0 / len(results_by_type) for t in results_by_type}
        }

    def _attention_fusion(
        self,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """
        Attention-based fusion (simplified without learned attention).

        Uses self-critique scores to determine attention weights.
        """
        attention_scores = {}
        values = {}

        for agent_type, results in results_by_type.items():
            # Average confidence as attention proxy
            confidences = [r.confidence for r in results if r.is_successful]
            if confidences:
                attention_scores[agent_type] = np.mean(confidences)
                values[agent_type] = np.mean(confidences)

        if not attention_scores:
            return {"score": 0.0, "classification": "no_data", "weights": {}}

        # Softmax over attention scores
        attn_values = np.array(list(attention_scores.values()))
        exp_attn = np.exp(attn_values - np.max(attn_values))
        softmax_attn = exp_attn / exp_attn.sum()

        # Weighted combination
        value_array = np.array(list(values.values()))
        fused_score = float(np.dot(softmax_attn, value_array))

        weights = {
            t: float(softmax_attn[i])
            for i, t in enumerate(attention_scores.keys())
        }

        return {
            "score": fused_score,
            "classification": self._score_to_classification(fused_score),
            "weights": weights
        }

    def _calculate_behavioral_severity(
        self,
        indicators: BehavioralIndicators
    ) -> float:
        """
        Calculate overall behavioral severity score.
        """
        # Weighted combination of behavioral indicators
        severity = (
            indicators.darvo_score * 0.3 +
            indicators.gaslighting_score * 0.3 +
            indicators.manipulation_score * 0.25 +
            indicators.deception_indicators * 0.15
        )

        # Boost if multiple patterns co-occur
        pattern_count = sum([
            1 if indicators.darvo_score > 0.3 else 0,
            1 if indicators.gaslighting_score > 0.3 else 0,
            1 if indicators.manipulation_score > 0.3 else 0
        ])

        if pattern_count >= 2:
            severity = min(1.0, severity * 1.2)

        return severity

    def _aggregate_result_confidences(
        self,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> float:
        """Aggregate all result confidences."""
        all_confidences = []
        for results in results_by_type.values():
            for r in results:
                if r.is_successful:
                    all_confidences.append(r.confidence)

        return np.mean(all_confidences) if all_confidences else 0.5

    def _determine_classification(
        self,
        indicators: BehavioralIndicators,
        fused_score: float
    ) -> str:
        """
        Determine the final classification based on indicators and score.
        """
        # If we have behavior probabilities, use the highest
        if indicators.behavior_probabilities:
            probs = indicators.behavior_probabilities
            max_class = max(probs, key=probs.get)
            if probs[max_class] > 0.5:
                return max_class

        # Fall back to threshold-based classification
        if fused_score < 0.2:
            return "low_concern"
        elif fused_score < 0.4:
            return "moderate_concern"
        elif fused_score < 0.6:
            return "elevated_concern"
        elif fused_score < 0.8:
            return "high_concern"
        else:
            return "critical_concern"

    def _score_to_classification(self, score: float) -> str:
        """Convert numeric score to classification label."""
        if score < 0.2:
            return "low"
        elif score < 0.4:
            return "moderate"
        elif score < 0.6:
            return "elevated"
        elif score < 0.8:
            return "high"
        else:
            return "critical"

    def _check_cross_modal_consistency(
        self,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> Dict[str, Any]:
        """
        Check consistency across different modalities/agents.
        """
        confidences_by_type = {}
        for agent_type, results in results_by_type.items():
            confs = [r.confidence for r in results if r.is_successful]
            if confs:
                confidences_by_type[agent_type] = np.mean(confs)

        if len(confidences_by_type) < 2:
            return {"score": 1.0, "details": "single_modality"}

        # Calculate variance in confidences
        values = list(confidences_by_type.values())
        variance = np.var(values)
        mean_conf = np.mean(values)

        # Low variance = high consistency
        consistency_score = max(0.0, 1.0 - variance * 4)

        # Check for outliers
        outliers = []
        for agent_type, conf in confidences_by_type.items():
            if abs(conf - mean_conf) > 0.3:
                outliers.append(agent_type)

        return {
            "score": consistency_score,
            "details": {
                "variance": float(variance),
                "mean_confidence": float(mean_conf),
                "outliers": outliers,
                "per_type": confidences_by_type
            }
        }

    def _detect_anomalies(
        self,
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the analysis results.
        """
        anomalies = []

        # Check for errors
        for agent_type, results in results_by_type.items():
            for result in results:
                if result.errors:
                    anomalies.append({
                        "type": "processing_error",
                        "agent": agent_type,
                        "details": result.errors
                    })

        # Check for very low confidence
        for agent_type, results in results_by_type.items():
            for result in results:
                if result.confidence < 0.3 and result.is_successful:
                    anomalies.append({
                        "type": "low_confidence",
                        "agent": agent_type,
                        "confidence": result.confidence
                    })

        # Check for conflicting classifications
        # (Would need access to individual classifications)

        return anomalies

    def _calculate_fusion_confidence(
        self,
        fused: Dict[str, Any],
        consistency: Dict[str, Any],
        results_by_type: Dict[str, List[AnalysisResult]]
    ) -> float:
        """
        Calculate confidence in the fusion result.
        """
        # Base from fused score distance from ambiguity
        base = 0.5 + abs(fused["score"] - 0.5)

        # Boost from consistency
        consistency_boost = consistency["score"] * 0.2

        # Boost from number of agents
        agent_count_boost = min(0.1, len(results_by_type) * 0.03)

        confidence = base + consistency_boost + agent_count_boost

        return min(1.0, confidence)

    async def self_critique(self, result: AnalysisResult) -> float:
        """
        Self-critique fusion quality.
        """
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize if low consistency
        consistency = result.findings.get("consistency_score", 1.0)
        if consistency < 0.5:
            score *= 0.7

        # Penalize for anomalies
        anomalies = result.findings.get("anomalies", [])
        if len(anomalies) > 0:
            score *= max(0.5, 1.0 - len(anomalies) * 0.1)

        # Bonus for multiple agents fused
        agents_fused = result.findings.get("agents_fused", [])
        if len(agents_fused) >= 2:
            score = min(1.0, score + 0.05)

        return score
