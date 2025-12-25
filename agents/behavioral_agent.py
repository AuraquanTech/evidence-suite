"""
Evidence Suite - Behavioral Analysis Agent
BERT-based behavioral pattern detection for forensic analysis.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np

from core.models import (
    EvidencePacket,
    AnalysisResult,
    ProcessingStage,
    BehavioralIndicators
)
from core.config import BehavioralConfig, default_config, hw_settings
from agents.base import BaseAgent


class BehavioralAgent(BaseAgent):
    """
    Intelligence layer agent for behavioral pattern analysis.

    Features:
    - VADER sentiment analysis
    - BERT-based text classification
    - DARVO pattern detection (Deny, Attack, Reverse Victim/Offender)
    - Gaslighting indicator detection
    - Manipulation scoring
    - Linguistic marker analysis
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[BehavioralConfig] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="behavioral",
            config=(config or default_config.behavioral).model_dump()
        )
        self.behavioral_config = config or default_config.behavioral
        self._vader = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._use_onnx = hw_settings.use_onnx
        self._onnx_bert = None

    async def _setup(self) -> None:
        """Initialize NLP models."""
        # Initialize VADER sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"VADER initialization failed: {e}")

        # Initialize BERT model - prefer ONNX for Blackwell GPU support
        model_name = self.behavioral_config.model_name

        if self._use_onnx:
            # Try ONNX Runtime first (has native Blackwell sm_120 support)
            try:
                from core.inference import get_bert_inference
                self._onnx_bert = get_bert_inference(
                    model_name=model_name,
                    use_gpu=True,
                    use_fp16=hw_settings.use_tensorrt,
                )
                self._device = self._onnx_bert.get_device()
                logger.info(f"ONNX BERT loaded: {model_name} on {self._device}")
                return
            except Exception as e:
                logger.warning(f"ONNX BERT initialization failed: {e}")

        # Fallback to PyTorch (may not work on RTX 5090 Blackwell)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            # Determine device
            if self.behavioral_config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.behavioral_config.device

            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"PyTorch BERT loaded: {model_name} on {self._device}")
        except Exception as e:
            logger.warning(f"PyTorch BERT initialization failed: {e}")
            logger.info("Behavioral analysis will run without BERT embeddings")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Analyze text for behavioral patterns.
        """
        text = packet.get_text_content()
        if not text or len(text.strip()) < 10:
            raise ValueError("Insufficient text content for behavioral analysis")

        # Run all analyses
        sentiment = self._analyze_sentiment(text)
        darvo = self._detect_darvo(text)
        gaslighting = self._detect_gaslighting(text)
        manipulation = self._calculate_manipulation_score(text, darvo, gaslighting)
        linguistic = self._analyze_linguistic_markers(text)
        embeddings = await self._get_bert_embeddings(text)
        classification = self._classify_behavior(embeddings, darvo, gaslighting)

        # Build behavioral indicators
        indicators = BehavioralIndicators(
            # Sentiment
            sentiment_compound=sentiment["compound"],
            sentiment_positive=sentiment["pos"],
            sentiment_negative=sentiment["neg"],
            sentiment_neutral=sentiment["neu"],

            # Pattern scores
            darvo_score=darvo["score"],
            gaslighting_score=gaslighting["score"],
            manipulation_score=manipulation,
            deception_indicators=self._detect_deception_indicators(text),

            # Linguistic markers
            hedging_frequency=linguistic["hedging"],
            certainty_markers=linguistic["certainty"],
            emotional_intensity=linguistic["emotional_intensity"],

            # Classification
            primary_behavior_class=classification["primary_class"],
            behavior_probabilities=classification["probabilities"]
        )

        # Calculate overall confidence
        confidence = self._calculate_confidence(indicators)

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            findings={
                "sentiment": sentiment,
                "darvo_matches": darvo["matches"],
                "gaslighting_matches": gaslighting["matches"],
                "linguistic_markers": linguistic,
                "behavior_classification": classification,
                "text_length": len(text),
                "sentence_count": len(self._split_sentences(text))
            },
            raw_output={
                "embeddings_shape": embeddings.shape if embeddings is not None else None
            }
        )

        return packet.with_updates(
            behavioral_indicators=indicators,
            stage=ProcessingStage.BEHAVIORAL_ANALYZED,
            analysis_results=packet.analysis_results + [analysis]
        )

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Run VADER sentiment analysis."""
        if self._vader is None:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        scores = self._vader.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"]
        }

    def _detect_darvo(self, text: str) -> Dict[str, Any]:
        """
        Detect DARVO patterns (Deny, Attack, Reverse Victim/Offender).

        Returns score (0-1) and matched phrases.
        """
        text_lower = text.lower()
        matches = []

        for keyword in self.behavioral_config.darvo_keywords:
            if keyword.lower() in text_lower:
                matches.append(keyword)

        # Calculate score based on density of matches
        words = text.split()
        density = len(matches) / max(len(words), 1) * 100

        # Normalize to 0-1 scale (cap at 5% density = 1.0)
        score = min(1.0, density / 5.0)

        return {
            "score": score,
            "matches": matches,
            "match_count": len(matches)
        }

    def _detect_gaslighting(self, text: str) -> Dict[str, Any]:
        """
        Detect gaslighting language patterns.
        """
        text_lower = text.lower()
        matches = []

        for phrase in self.behavioral_config.gaslighting_phrases:
            if phrase.lower() in text_lower:
                matches.append(phrase)

        # Additional patterns using regex
        patterns = [
            r"you('re| are) (being )?(too |so )?(sensitive|dramatic|crazy|paranoid)",
            r"(that|it) (never|didn't) happen",
            r"you('re| are) (imagining|making) (things|it) up",
            r"no one (else |ever )?(thinks|believes|said)",
            r"you (always|never) (do|say|remember)"
        ]

        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches.append(f"[pattern: {pattern[:30]}...]")

        # Score based on unique matches
        unique_matches = list(set(matches))
        score = min(1.0, len(unique_matches) / 5.0)

        return {
            "score": score,
            "matches": unique_matches,
            "match_count": len(unique_matches)
        }

    def _calculate_manipulation_score(
        self,
        text: str,
        darvo: Dict,
        gaslighting: Dict
    ) -> float:
        """
        Calculate overall manipulation score combining multiple signals.
        """
        # Base from DARVO and gaslighting
        base_score = (darvo["score"] * 0.4 + gaslighting["score"] * 0.4)

        # Add other manipulation indicators
        text_lower = text.lower()

        manipulation_phrases = [
            "if you loved me", "after all i've done",
            "you owe me", "you made me do",
            "no one else would", "you're lucky i",
            "don't tell anyone", "this is your fault"
        ]

        matches = sum(1 for p in manipulation_phrases if p in text_lower)
        phrase_score = min(0.2, matches * 0.05)

        return min(1.0, base_score + phrase_score)

    def _detect_deception_indicators(self, text: str) -> float:
        """
        Detect linguistic indicators often associated with deception.

        Note: These are probabilistic indicators, not definitive proof.
        """
        text_lower = text.lower()
        indicators = 0

        # Excessive qualifiers
        qualifiers = ["honestly", "truthfully", "to be honest", "believe me",
                      "i swear", "trust me", "frankly"]
        indicators += sum(1 for q in qualifiers if q in text_lower)

        # Distancing language
        distancing = ["that person", "that thing", "those people",
                      "one might", "someone", "they say"]
        indicators += sum(0.5 for d in distancing if d in text_lower)

        # Lack of contractions (more formal = potential deception)
        formal_patterns = [" did not ", " do not ", " was not ", " is not ",
                          " would not ", " could not "]
        indicators += sum(0.3 for f in formal_patterns if f in text_lower)

        # Normalize
        words = len(text.split())
        normalized = indicators / max(words / 50, 1)

        return min(1.0, normalized)

    def _analyze_linguistic_markers(self, text: str) -> Dict[str, float]:
        """
        Analyze linguistic patterns in the text.
        """
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        # Hedging language
        hedges = ["maybe", "perhaps", "possibly", "might", "could be",
                  "sort of", "kind of", "i think", "i guess", "probably"]
        hedge_count = sum(1 for h in hedges if h in text_lower)
        hedging = hedge_count / max(word_count / 20, 1)

        # Certainty markers
        certainty_words = ["definitely", "certainly", "absolutely", "always",
                          "never", "must", "have to", "obviously", "clearly"]
        certainty_count = sum(1 for c in certainty_words if c in text_lower)
        certainty = certainty_count / max(word_count / 20, 1)

        # Emotional intensity (exclamation marks, caps, emotional words)
        exclamations = text.count("!")
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)

        emotional_words = ["hate", "love", "furious", "terrified", "ecstatic",
                          "devastated", "amazing", "horrible", "disgusting"]
        emotion_count = sum(1 for e in emotional_words if e in text_lower)

        emotional_intensity = (
            exclamations * 0.1 + caps_words * 0.1 + emotion_count * 0.2
        ) / max(word_count / 50, 1)

        return {
            "hedging": min(1.0, hedging),
            "certainty": min(1.0, certainty),
            "emotional_intensity": min(1.0, emotional_intensity)
        }

    async def _get_bert_embeddings(self, text: str) -> Optional[np.ndarray]:
        """
        Get BERT embeddings for the text.
        Uses ONNX Runtime when available (Blackwell GPU support),
        falls back to PyTorch otherwise.
        """
        # Try ONNX first
        if self._onnx_bert is not None:
            try:
                embeddings = self._onnx_bert.encode_single(text)
                return embeddings.reshape(1, -1)
            except Exception as e:
                logger.warning(f"ONNX BERT embedding extraction failed: {e}")

        # Fallback to PyTorch
        if self._model is None or self._tokenizer is None:
            return None

        import torch

        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                max_length=self.behavioral_config.max_sequence_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embeddings

        except Exception as e:
            logger.warning(f"PyTorch BERT embedding extraction failed: {e}")
            return None

    def _classify_behavior(
        self,
        embeddings: Optional[np.ndarray],
        darvo: Dict,
        gaslighting: Dict
    ) -> Dict[str, Any]:
        """
        Classify the primary behavioral pattern.

        Uses rule-based classification initially.
        TODO: Replace with trained classifier using embeddings.
        """
        # Simple rule-based classification
        probabilities = {
            "normal": 0.5,
            "darvo": darvo["score"],
            "gaslighting": gaslighting["score"],
            "manipulation": (darvo["score"] + gaslighting["score"]) / 2,
            "aggressive": 0.0,
            "defensive": 0.0
        }

        # Adjust normal probability based on detected patterns
        max_pattern = max(darvo["score"], gaslighting["score"])
        probabilities["normal"] = max(0.0, 1.0 - max_pattern * 1.5)

        # Normalize to sum to 1
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        # Get primary class
        primary_class = max(probabilities, key=probabilities.get)

        return {
            "primary_class": primary_class,
            "probabilities": probabilities
        }

    def _calculate_confidence(self, indicators: BehavioralIndicators) -> float:
        """
        Calculate overall confidence in the behavioral analysis.
        """
        # Higher confidence when patterns are clearly detected or clearly absent
        pattern_scores = [
            indicators.darvo_score,
            indicators.gaslighting_score,
            indicators.manipulation_score
        ]

        # Confidence is higher at extremes (clear detection or clear absence)
        avg_score = np.mean(pattern_scores)
        distance_from_middle = abs(avg_score - 0.5) * 2  # 0-1 scale

        # Base confidence from distance from ambiguity
        confidence = 0.5 + distance_from_middle * 0.3

        # Boost if we have BERT embeddings (ONNX or PyTorch)
        if self._onnx_bert is not None or self._model is not None:
            confidence = min(1.0, confidence + 0.1)

        return confidence

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def self_critique(self, result: AnalysisResult) -> float:
        """
        Self-critique behavioral analysis quality.
        """
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize if text was too short
        text_length = result.findings.get("text_length", 0)
        if text_length < 50:
            score *= 0.6
        elif text_length < 100:
            score *= 0.8

        # Bonus for multiple pattern types detected
        darvo_matches = result.findings.get("darvo_matches", [])
        gaslighting_matches = result.findings.get("gaslighting_matches", [])

        if len(darvo_matches) > 0 and len(gaslighting_matches) > 0:
            score = min(1.0, score + 0.05)

        return score
