"""
Evidence Suite - Comprehensive Agent Tests
Unit tests for OCR, Behavioral, and Fusion agents.
"""
import pytest
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBehavioralAgent:
    """Test behavioral analysis agent."""

    @pytest.fixture
    async def agent(self):
        """Create and initialize behavioral agent."""
        from agents.behavioral_agent import BehavioralAgent
        agent = BehavioralAgent()
        await agent.initialize()
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.initialized

    @pytest.mark.asyncio
    async def test_normal_text_analysis(self, agent):
        """Test analysis of normal, non-abusive text."""
        text = "I appreciate your help with this matter. Thank you for your time."
        result = await agent.analyze(text)

        assert result.success
        assert result.data is not None
        assert "darvo_score" in result.data
        assert result.data["darvo_score"] < 0.3  # Should be low

    @pytest.mark.asyncio
    async def test_darvo_detection(self, agent):
        """Test DARVO pattern detection."""
        text = "You're the one who caused all of this! I'm the real victim here! How dare you blame me!"
        result = await agent.analyze(text)

        assert result.success
        assert result.data["darvo_score"] > 0.5  # Should detect DARVO

    @pytest.mark.asyncio
    async def test_gaslighting_detection(self, agent):
        """Test gaslighting pattern detection."""
        text = "That never happened. You're imagining things. Your memory is completely wrong."
        result = await agent.analyze(text)

        assert result.success
        assert result.data["gaslighting_score"] > 0.3  # Should detect gaslighting

    @pytest.mark.asyncio
    async def test_manipulation_detection(self, agent):
        """Test manipulation pattern detection."""
        text = "If you really loved me, you would do this. After everything I've done for you..."
        result = await agent.analyze(text)

        assert result.success
        assert result.data["manipulation_score"] > 0.0

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, agent):
        """Test sentiment analysis component."""
        positive_text = "I love spending time with you. You make me so happy!"
        negative_text = "I hate this situation. Everything is terrible."

        pos_result = await agent.analyze(positive_text)
        neg_result = await agent.analyze(negative_text)

        assert pos_result.success
        assert neg_result.success
        assert pos_result.data["sentiment"]["compound"] > 0
        assert neg_result.data["sentiment"]["compound"] < 0

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, agent):
        """Test handling of empty text."""
        result = await agent.analyze("")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_long_text_handling(self, agent):
        """Test handling of long text."""
        long_text = "This is a test. " * 500  # Very long text
        result = await agent.analyze(long_text)

        assert result is not None
        assert result.success

    @pytest.mark.asyncio
    async def test_special_characters(self, agent):
        """Test handling of special characters."""
        text = "Test with émojis 😀 and spëcial çharacters! @#$%"
        result = await agent.analyze(text)

        assert result is not None


class TestFusionAgent:
    """Test fusion agent."""

    @pytest.fixture
    async def agent(self):
        """Create and initialize fusion agent."""
        from agents.fusion_agent import FusionAgent
        agent = FusionAgent()
        await agent.initialize()
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.initialized

    @pytest.mark.asyncio
    async def test_basic_fusion(self, agent):
        """Test basic score fusion."""
        input_data = {
            "behavioral": {
                "sentiment": {"compound": -0.5},
                "darvo_score": 0.8,
                "gaslighting_score": 0.6,
                "manipulation_score": 0.4,
            },
            "ocr": {"confidence": 0.95},
        }

        result = await agent.fuse(input_data)

        assert result.success
        assert "fused_score" in result.data
        assert "classification" in result.data
        assert 0 <= result.data["fused_score"] <= 1

    @pytest.mark.asyncio
    async def test_high_concern_classification(self, agent):
        """Test high concern classification."""
        input_data = {
            "behavioral": {
                "sentiment": {"compound": -0.8},
                "darvo_score": 0.9,
                "gaslighting_score": 0.8,
                "manipulation_score": 0.7,
            },
        }

        result = await agent.fuse(input_data)

        assert result.success
        assert result.data["classification"] in ["high_concern", "critical"]

    @pytest.mark.asyncio
    async def test_normal_classification(self, agent):
        """Test normal classification."""
        input_data = {
            "behavioral": {
                "sentiment": {"compound": 0.5},
                "darvo_score": 0.0,
                "gaslighting_score": 0.0,
                "manipulation_score": 0.0,
            },
        }

        result = await agent.fuse(input_data)

        assert result.success
        assert result.data["classification"] == "normal"

    @pytest.mark.asyncio
    async def test_missing_data_handling(self, agent):
        """Test handling of missing input data."""
        input_data = {
            "behavioral": {
                "sentiment": {"compound": 0.0},
            },
        }

        result = await agent.fuse(input_data)

        # Should handle missing fields gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_confidence_score(self, agent):
        """Test confidence score calculation."""
        input_data = {
            "behavioral": {
                "sentiment": {"compound": -0.5},
                "darvo_score": 0.5,
            },
        }

        result = await agent.fuse(input_data)

        assert result.success
        if "confidence" in result.data:
            assert 0 <= result.data["confidence"] <= 1


class TestOCRAgent:
    """Test OCR agent."""

    @pytest.fixture
    async def agent(self):
        """Create and initialize OCR agent."""
        from agents.ocr_agent import OCRAgent
        agent = OCRAgent()
        await agent.initialize()
        return agent

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        # OCR agent may not fully initialize without tesseract
        # Just check it doesn't crash


class TestAgentIntegration:
    """Integration tests for agent pipeline."""

    @pytest.mark.asyncio
    async def test_behavioral_to_fusion_flow(self):
        """Test data flow from behavioral to fusion agent."""
        from agents.behavioral_agent import BehavioralAgent
        from agents.fusion_agent import FusionAgent

        behavioral = BehavioralAgent()
        fusion = FusionAgent()

        await behavioral.initialize()
        await fusion.initialize()

        # Analyze text
        text = "You always blame me for everything! I'm the victim here!"
        behavioral_result = await behavioral.analyze(text)

        assert behavioral_result.success

        # Fuse results
        fusion_input = {
            "behavioral": behavioral_result.data,
        }
        fusion_result = await fusion.fuse(fusion_input)

        assert fusion_result.success
        assert "fused_score" in fusion_result.data

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent agent operations."""
        from agents.behavioral_agent import BehavioralAgent

        agent = BehavioralAgent()
        await agent.initialize()

        texts = [
            "You're always wrong!",
            "Thank you for your help.",
            "That never happened.",
            "I appreciate your time.",
            "How dare you blame me!",
        ]

        # Run concurrent analyses
        tasks = [agent.analyze(text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(texts)
        assert all(r.success for r in results)


class TestPerformanceMetrics:
    """Test performance-related functionality."""

    @pytest.mark.asyncio
    async def test_behavioral_latency(self):
        """Test behavioral agent latency is within bounds."""
        import time
        from agents.behavioral_agent import BehavioralAgent

        agent = BehavioralAgent()
        await agent.initialize()

        text = "Test text for latency measurement."

        # Warmup
        await agent.analyze(text)

        # Measure
        start = time.perf_counter()
        await agent.analyze(text)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete within reasonable time (500ms for CPU)
        assert duration_ms < 1000, f"Latency too high: {duration_ms}ms"

    @pytest.mark.asyncio
    async def test_memory_stability(self):
        """Test agent doesn't leak memory over multiple calls."""
        import psutil
        import os
        from agents.behavioral_agent import BehavioralAgent

        agent = BehavioralAgent()
        await agent.initialize()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Run multiple analyses
        for i in range(50):
            await agent.analyze(f"Test text number {i}")

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded (< 100MB for 50 calls)
        assert memory_increase < 100, f"Memory leak detected: {memory_increase}MB increase"


# Run with: pytest tests/test_agents.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
