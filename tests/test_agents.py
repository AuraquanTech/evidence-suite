"""Evidence Suite - Comprehensive Agent Tests
Tests for all agents including VideoAgent, ImageAgent, AudioAgent, EmailAgent.
"""

import asyncio
import io
import os
import sys
from email.message import EmailMessage
from pathlib import Path

import pytest


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Behavioral Agent Tests
# ============================================================================


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
        assert agent._initialized

    @pytest.mark.asyncio
    async def test_normal_text_analysis(self, agent):
        """Test analysis of normal, non-abusive text."""
        from core.models import EvidencePacket, EvidenceType

        packet = EvidencePacket(
            extracted_text="I appreciate your help with this matter. Thank you for your time.",
            evidence_type=EvidenceType.TEXT,
        )
        result = await agent.process(packet)

        assert result is not None
        assert len(result.analysis_results) > 0

    @pytest.mark.asyncio
    async def test_darvo_detection(self, agent):
        """Test DARVO pattern detection."""
        from core.models import EvidencePacket, EvidenceType

        packet = EvidencePacket(
            extracted_text="You're the one who caused all of this! I'm the real victim here! How dare you blame me!",
            evidence_type=EvidenceType.TEXT,
        )
        result = await agent.process(packet)

        assert result is not None
        behavioral_result = result.analysis_results[-1]
        assert behavioral_result.agent_type == "behavioral"

    @pytest.mark.asyncio
    async def test_gaslighting_detection(self, agent):
        """Test gaslighting pattern detection."""
        from core.models import EvidencePacket, EvidenceType

        packet = EvidencePacket(
            extracted_text="That never happened. You're imagining things. Your memory is completely wrong.",
            evidence_type=EvidenceType.TEXT,
        )
        result = await agent.process(packet)

        assert result is not None
        assert len(result.analysis_results) > 0

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, agent):
        """Test handling of empty text."""
        from core.models import EvidencePacket, EvidenceType

        packet = EvidencePacket(
            extracted_text="",
            evidence_type=EvidenceType.TEXT,
        )
        result = await agent.process(packet)

        assert result is not None

    @pytest.mark.asyncio
    async def test_long_text_handling(self, agent):
        """Test handling of long text."""
        from core.models import EvidencePacket, EvidenceType

        long_text = "This is a test. " * 500
        packet = EvidencePacket(
            extracted_text=long_text,
            evidence_type=EvidenceType.TEXT,
        )
        result = await agent.process(packet)

        assert result is not None


# ============================================================================
# Fusion Agent Tests
# ============================================================================


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
        assert agent._initialized

    @pytest.mark.asyncio
    async def test_basic_fusion(self, agent):
        """Test basic score fusion."""
        from core.models import AnalysisResult, EvidencePacket, EvidenceType

        result1 = AnalysisResult(
            agent_id="behavioral",
            agent_type="behavioral",
            confidence=0.8,
            findings={"sentiment": "negative"},
        )

        packet = EvidencePacket(
            extracted_text="Test content",
            evidence_type=EvidenceType.TEXT,
            analysis_results=[result1],
        )

        result = await agent.process(packet)
        assert result is not None


# ============================================================================
# OCR Agent Tests
# ============================================================================


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


# ============================================================================
# Email Agent Tests
# ============================================================================


class TestEmailAgent:
    """Tests for EmailAgent."""

    def _create_test_email(
        self,
        subject: str = "Test Subject",
        body: str = "Test body content",
        from_addr: str = "sender@example.com",
        to_addr: str = "recipient@example.com",
    ) -> bytes:
        """Create a test email message."""
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
        msg["Message-ID"] = "<test@example.com>"
        msg.set_content(body)
        return msg.as_bytes()

    @pytest.mark.asyncio
    async def test_email_agent_creation(self):
        """Test email agent initialization."""
        from agents.email_agent import EmailAgent

        agent = EmailAgent()
        assert agent.agent_type == "email"
        await agent.initialize()

    @pytest.mark.asyncio
    async def test_email_parsing(self):
        """Test basic email parsing."""
        from agents.email_agent import EmailAgent
        from core.models import EvidencePacket, EvidenceType

        agent = EmailAgent()
        await agent.initialize()

        email_content = self._create_test_email(
            subject="Important Message",
            body="This is the email body.",
        )

        packet = EvidencePacket(
            raw_content=email_content,
            evidence_type=EvidenceType.EMAIL,
        )

        result = await agent.process(packet)

        assert result.extracted_text is not None
        assert "Important Message" in result.extracted_text
        assert len(result.analysis_results) > 0

        email_result = result.analysis_results[-1]
        assert email_result.findings.get("subject") == "Important Message"

    @pytest.mark.asyncio
    async def test_spoofing_detection(self):
        """Test email spoofing detection."""
        from agents.email_agent import EmailAgent
        from core.models import EvidencePacket, EvidenceType

        agent = EmailAgent()
        await agent.initialize()

        email_content = self._create_test_email()

        packet = EvidencePacket(
            raw_content=email_content,
            evidence_type=EvidenceType.EMAIL,
        )

        result = await agent.process(packet)
        email_result = result.analysis_results[-1]

        assert "spoofing_risk" in email_result.findings

    @pytest.mark.asyncio
    async def test_url_extraction(self):
        """Test URL extraction from email."""
        from agents.email_agent import EmailAgent
        from core.models import EvidencePacket, EvidenceType

        agent = EmailAgent()
        await agent.initialize()

        email_content = self._create_test_email(
            body="Check out https://example.com and http://test.org for more info."
        )

        packet = EvidencePacket(
            raw_content=email_content,
            evidence_type=EvidenceType.EMAIL,
        )

        result = await agent.process(packet)
        email_result = result.analysis_results[-1]

        assert email_result.findings.get("url_count", 0) >= 2

    @pytest.mark.asyncio
    async def test_wrong_evidence_type_returns_error(self):
        """Test that wrong evidence type returns error result."""
        from agents.email_agent import EmailAgent
        from core.models import EvidencePacket, EvidenceType

        agent = EmailAgent()
        await agent.initialize()

        packet = EvidencePacket(
            raw_content=b"not an email",
            evidence_type=EvidenceType.TEXT,
        )

        result = await agent.process(packet)
        # BaseAgent.process catches exceptions and returns error in result
        assert len(result.analysis_results) > 0
        assert len(result.analysis_results[-1].errors) > 0
        assert "Expected email evidence" in result.analysis_results[-1].errors[0]


# ============================================================================
# Image Agent Tests
# ============================================================================


class TestImageAgent:
    """Tests for ImageAgent."""

    def _create_test_image(self, width: int = 100, height: int = 100) -> bytes:
        """Create a simple test image."""
        try:
            from PIL import Image

            img = Image.new("RGB", (width, height), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except ImportError:
            pytest.skip("PIL not installed")

    @pytest.mark.asyncio
    async def test_image_agent_creation(self):
        """Test image agent initialization."""
        from agents.image_agent import ImageAgent

        agent = ImageAgent()
        assert agent.agent_type == "image"

    @pytest.mark.asyncio
    async def test_image_processing(self):
        """Test basic image processing."""
        from agents.image_agent import ImageAgent
        from core.models import EvidencePacket, EvidenceType

        agent = ImageAgent()
        await agent.initialize()

        if not agent._pil_available:
            pytest.skip("PIL not available")

        image_data = self._create_test_image()

        packet = EvidencePacket(
            raw_content=image_data,
            evidence_type=EvidenceType.IMAGE,
        )

        result = await agent.process(packet)

        assert len(result.analysis_results) > 0
        image_result = result.analysis_results[-1]

        assert image_result.findings.get("format") == "PNG"
        assert image_result.findings.get("resolution") == "100x100"

    @pytest.mark.asyncio
    async def test_hash_calculation(self):
        """Test image hash calculation."""
        from agents.image_agent import ImageAgent
        from core.models import EvidencePacket, EvidenceType

        agent = ImageAgent()
        await agent.initialize()

        if not agent._pil_available:
            pytest.skip("PIL not available")

        image_data = self._create_test_image()

        packet = EvidencePacket(
            raw_content=image_data,
            evidence_type=EvidenceType.IMAGE,
        )

        result = await agent.process(packet)
        image_result = result.analysis_results[-1]

        hashes = image_result.raw_output.get("hashes", {})
        assert "sha256" in hashes
        assert "md5" in hashes
        assert len(hashes["sha256"]) == 64

    @pytest.mark.asyncio
    async def test_wrong_evidence_type_returns_error(self):
        """Test that wrong evidence type returns error result."""
        from agents.image_agent import ImageAgent
        from core.models import EvidencePacket, EvidenceType

        agent = ImageAgent()
        await agent.initialize()

        packet = EvidencePacket(
            raw_content=b"not an image",
            evidence_type=EvidenceType.TEXT,
        )

        result = await agent.process(packet)
        assert len(result.analysis_results) > 0
        assert len(result.analysis_results[-1].errors) > 0
        assert "Expected image evidence" in result.analysis_results[-1].errors[0]


# ============================================================================
# Video Agent Tests
# ============================================================================


class TestVideoAgent:
    """Tests for VideoAgent."""

    @pytest.mark.asyncio
    async def test_video_agent_creation(self):
        """Test video agent initialization."""
        from agents.video_agent import VideoAgent

        agent = VideoAgent()
        assert agent.agent_type == "video"

    @pytest.mark.asyncio
    async def test_video_agent_initialization(self):
        """Test video agent setup."""
        from agents.video_agent import VideoAgent

        agent = VideoAgent()
        await agent.initialize()

        assert agent._initialized

    @pytest.mark.asyncio
    async def test_wrong_evidence_type_returns_error(self):
        """Test that wrong evidence type returns error result."""
        from agents.video_agent import VideoAgent
        from core.models import EvidencePacket, EvidenceType

        agent = VideoAgent()
        await agent.initialize()

        if not agent._cv2_available:
            pytest.skip("OpenCV not available")

        packet = EvidencePacket(
            raw_content=b"not a video",
            evidence_type=EvidenceType.TEXT,
        )

        result = await agent.process(packet)
        assert len(result.analysis_results) > 0
        assert len(result.analysis_results[-1].errors) > 0
        assert "Expected video evidence" in result.analysis_results[-1].errors[0]


# ============================================================================
# Audio Agent Tests
# ============================================================================


class TestAudioAgent:
    """Tests for AudioAgent."""

    @pytest.mark.asyncio
    async def test_audio_agent_creation(self):
        """Test audio agent initialization."""
        from agents.audio_agent import AudioAgent

        agent = AudioAgent()
        assert agent.agent_type == "audio"

    @pytest.mark.asyncio
    async def test_audio_agent_initialization(self):
        """Test audio agent setup."""
        from agents.audio_agent import AudioAgent

        agent = AudioAgent()
        await agent.initialize()

        assert agent._initialized

    @pytest.mark.asyncio
    async def test_wrong_evidence_type_returns_error(self):
        """Test that wrong evidence type returns error result."""
        from agents.audio_agent import AudioAgent
        from core.models import EvidencePacket, EvidenceType

        agent = AudioAgent()
        await agent.initialize()

        packet = EvidencePacket(
            raw_content=b"not audio",
            evidence_type=EvidenceType.TEXT,
        )

        result = await agent.process(packet)
        assert len(result.analysis_results) > 0
        assert len(result.analysis_results[-1].errors) > 0
        assert "Expected audio evidence" in result.analysis_results[-1].errors[0]


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_creation(self):
        """Test pipeline initialization."""
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        assert not pipeline._initialized

    @pytest.mark.asyncio
    async def test_pipeline_text_routing(self):
        """Test TEXT evidence routing."""
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        routing = pipeline._get_routing_plan(EvidenceType.TEXT)

        assert "behavioral" in routing
        assert "fusion" in routing
        assert "ocr" not in routing

    @pytest.mark.asyncio
    async def test_pipeline_image_routing(self):
        """Test IMAGE evidence routing."""
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        routing = pipeline._get_routing_plan(EvidenceType.IMAGE)

        assert "image" in routing
        assert "ocr" in routing
        assert "behavioral" in routing
        assert "fusion" in routing

    @pytest.mark.asyncio
    async def test_pipeline_audio_routing(self):
        """Test AUDIO evidence routing."""
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        routing = pipeline._get_routing_plan(EvidenceType.AUDIO)

        assert "audio" in routing
        assert "behavioral" in routing
        assert "fusion" in routing

    @pytest.mark.asyncio
    async def test_pipeline_email_routing(self):
        """Test EMAIL evidence routing."""
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        routing = pipeline._get_routing_plan(EvidenceType.EMAIL)

        assert "email" in routing
        assert "behavioral" in routing
        assert "fusion" in routing

    @pytest.mark.asyncio
    async def test_pipeline_video_routing(self):
        """Test VIDEO evidence routing."""
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()
        routing = pipeline._get_routing_plan(EvidenceType.VIDEO)

        assert "video" in routing
        assert "behavioral" in routing
        assert "fusion" in routing

    @pytest.mark.asyncio
    async def test_pipeline_text_processing(self):
        """Test full pipeline for text evidence."""
        from core.models import EvidencePacket, EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()

        packet = EvidencePacket(
            extracted_text="This is a test message for behavioral analysis.",
            evidence_type=EvidenceType.TEXT,
        )

        try:
            result = await pipeline.process(packet)

            assert result.success
            assert len(result.stages_completed) > 0
            assert "behavioral" in result.agents_used
            assert "fusion" in result.agents_used

        finally:
            await pipeline.shutdown()

    @pytest.mark.asyncio
    async def test_pipeline_metrics(self):
        """Test pipeline metrics tracking."""
        from core.models import EvidencePacket, EvidenceType
        from pipeline import EvidencePipeline

        pipeline = EvidencePipeline()

        packet = EvidencePacket(
            extracted_text="Test message",
            evidence_type=EvidenceType.TEXT,
        )

        try:
            await pipeline.process(packet)
            metrics = pipeline.get_metrics()

            assert metrics["packets_processed"] == 1
            assert metrics["successful"] == 1
            assert "text" in metrics["by_type"]

        finally:
            await pipeline.shutdown()


# ============================================================================
# Chain of Custody Tests
# ============================================================================


class TestChainOfCustody:
    """Tests for chain of custody tracking."""

    def test_chain_creation(self):
        """Test chain of custody creation."""
        from core.models import ChainOfCustody

        chain = ChainOfCustody(evidence_id="test-evidence-id")
        assert len(chain.entries) == 0

    def test_chain_entry_addition(self):
        """Test adding entries to chain."""
        from core.models import ChainOfCustody

        chain = ChainOfCustody(evidence_id="test-evidence-id")
        chain.add_entry(
            agent_id="test_agent",
            action="process",
            input_data="input_hash",
            output_data="output_hash",
        )

        assert len(chain.entries) == 1
        assert chain.entries[0].agent_id == "test_agent"

    def test_chain_verification(self):
        """Test chain integrity verification."""
        from core.models import ChainOfCustody

        chain = ChainOfCustody(evidence_id="test-evidence-id")
        chain.add_entry(
            agent_id="agent1",
            action="process",
            input_data="hash1",
            output_data="hash2",
        )
        chain.add_entry(
            agent_id="agent2",
            action="analyze",
            input_data="hash2",
            output_data="hash3",
        )

        assert chain.verify_chain()


# ============================================================================
# Evidence Packet Tests
# ============================================================================


class TestEvidencePacket:
    """Tests for EvidencePacket model."""

    def test_packet_creation(self):
        """Test evidence packet creation."""
        from core.models import EvidencePacket, EvidenceType, ProcessingStage

        packet = EvidencePacket(
            raw_content=b"test content",
            evidence_type=EvidenceType.TEXT,
        )

        assert packet.id is not None
        assert packet.evidence_type == EvidenceType.TEXT
        assert packet.stage == ProcessingStage.RAW

    def test_packet_immutability(self):
        """Test packet immutability via with_updates."""
        from core.models import EvidencePacket, EvidenceType, ProcessingStage

        packet = EvidencePacket(
            raw_content=b"test",
            evidence_type=EvidenceType.TEXT,
        )

        updated = packet.with_updates(
            stage=ProcessingStage.BEHAVIORAL_ANALYZED,
            extracted_text="extracted",
        )

        assert packet.stage == ProcessingStage.RAW
        assert packet.extracted_text is None

        assert updated.stage == ProcessingStage.BEHAVIORAL_ANALYZED
        assert updated.extracted_text == "extracted"

    def test_content_hash(self):
        """Test content hash calculation."""
        from core.models import EvidencePacket, EvidenceType

        packet = EvidencePacket(
            raw_content=b"test content",
            evidence_type=EvidenceType.TEXT,
        )

        assert packet.content_hash is not None
        assert len(packet.content_hash) > 0


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Tests for configuration classes."""

    def test_default_config(self):
        """Test default configuration loading."""
        from core.config import default_config

        assert default_config is not None
        assert default_config.ocr is not None
        assert default_config.behavioral is not None
        assert default_config.fusion is not None

    def test_new_agent_configs_exist(self):
        """Test new agent configurations are in Config."""
        from core.config import Config

        config = Config()

        assert hasattr(config, "audio")
        assert hasattr(config, "email")
        assert hasattr(config, "video")
        assert hasattr(config, "image")

    def test_audio_config(self):
        """Test AudioConfig defaults."""
        from core.config import AudioConfig

        config = AudioConfig()

        assert config.whisper_model == "base"
        assert config.enable_diarization is True

    def test_email_config(self):
        """Test EmailConfig defaults."""
        from core.config import EmailConfig

        config = EmailConfig()

        assert config.extract_attachments is True
        assert config.detect_spoofing is True

    def test_video_config(self):
        """Test VideoConfig defaults."""
        from core.config import VideoConfig

        config = VideoConfig()

        assert config.extract_audio is True
        assert config.scene_detection is True

    def test_image_config(self):
        """Test ImageConfig defaults."""
        from core.config import ImageConfig

        config = ImageConfig()

        assert config.extract_exif is True
        assert config.detect_manipulation is True


# Run with: pytest tests/test_agents.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
