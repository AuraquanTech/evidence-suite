"""
Evidence Suite - Audio Agent
Whisper-based transcription with Pyannote speaker diarization.
"""
from __future__ import annotations
import io
import tempfile
import os
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np

from core.models import (
    EvidencePacket,
    AnalysisResult,
    ProcessingStage,
    EvidenceType
)
from core.config import default_config, hw_settings
from agents.base import BaseAgent


class AudioConfig:
    """Audio Agent configuration."""
    def __init__(
        self,
        whisper_model: str = "base",
        language: Optional[str] = None,
        enable_diarization: bool = True,
        device: str = "auto",
        batch_size: int = 16
    ):
        self.whisper_model = whisper_model  # tiny, base, small, medium, large
        self.language = language  # None = auto-detect
        self.enable_diarization = enable_diarization
        self.device = device
        self.batch_size = batch_size


def _check_gpu_for_whisper() -> Tuple[bool, str]:
    """
    Check if GPU is available for Whisper.
    Whisper uses PyTorch, which doesn't support Blackwell (sm_120).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check compute capability
        device = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(device)

        # sm_120 (Blackwell) is not supported by PyTorch
        if cc[0] >= 12:
            return False, f"Blackwell GPU (sm_{cc[0]}{cc[1]}) - using CPU"

        return True, f"GPU available (sm_{cc[0]}{cc[1]})"

    except Exception as e:
        return False, f"GPU check failed: {e}"


class AudioAgent(BaseAgent):
    """
    Sensory layer agent for audio transcription and analysis.

    Features:
    - OpenAI Whisper transcription (multiple model sizes)
    - Pyannote speaker diarization
    - Speaker-attributed transcripts
    - Language detection
    - Timestamp alignment
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[AudioConfig] = None
    ):
        self.audio_config = config or AudioConfig()
        super().__init__(
            agent_id=agent_id,
            agent_type="audio",
            config={
                "whisper_model": self.audio_config.whisper_model,
                "language": self.audio_config.language,
                "enable_diarization": self.audio_config.enable_diarization,
            }
        )
        self._whisper_model = None
        self._diarization_pipeline = None
        self._device = "cpu"

    async def _setup(self) -> None:
        """Initialize audio processing models."""
        # Check GPU availability
        use_gpu, gpu_reason = _check_gpu_for_whisper()
        self._device = "cuda" if use_gpu else "cpu"
        logger.info(f"Audio Agent GPU check: {gpu_reason}")

        # Initialize Whisper
        await self._init_whisper()

        # Initialize Pyannote diarization
        if self.audio_config.enable_diarization:
            await self._init_diarization()

    async def _init_whisper(self) -> None:
        """Initialize Whisper model."""
        try:
            import whisper

            model_name = self.audio_config.whisper_model
            logger.info(f"Loading Whisper model: {model_name}")

            self._whisper_model = whisper.load_model(
                model_name,
                device=self._device
            )
            logger.info(f"Whisper {model_name} loaded on {self._device}")

        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")
        except Exception as e:
            logger.warning(f"Whisper initialization failed: {e}")

    async def _init_diarization(self) -> None:
        """Initialize Pyannote speaker diarization."""
        try:
            from pyannote.audio import Pipeline

            # Note: Pyannote requires HuggingFace token for some models
            # Using offline mode if token not available
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")

            if hf_token:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if self._device == "cuda":
                    import torch
                    self._diarization_pipeline.to(torch.device("cuda"))
                logger.info("Pyannote diarization pipeline loaded")
            else:
                logger.warning("HUGGINGFACE_TOKEN not set - diarization disabled")
                self.audio_config.enable_diarization = False

        except ImportError:
            logger.warning("Pyannote not installed. Install with: pip install pyannote.audio")
            self.audio_config.enable_diarization = False
        except Exception as e:
            logger.warning(f"Pyannote initialization failed: {e}")
            self.audio_config.enable_diarization = False

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Process audio evidence.

        Steps:
        1. Save audio to temp file
        2. Transcribe with Whisper
        3. Run speaker diarization (if enabled)
        4. Merge transcription with speaker labels
        """
        if packet.evidence_type != EvidenceType.AUDIO:
            raise ValueError(f"Expected audio evidence, got {packet.evidence_type}")

        if not packet.raw_content:
            raise ValueError("No audio content to process")

        if self._whisper_model is None:
            raise RuntimeError("Whisper model not initialized")

        # Save to temp file (Whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(packet.raw_content)
            audio_path = f.name

        try:
            # Transcribe
            transcription = await self._transcribe(audio_path)

            # Diarization
            diarization = None
            if self.audio_config.enable_diarization and self._diarization_pipeline:
                diarization = await self._diarize(audio_path)

            # Merge results
            if diarization:
                merged_transcript = self._merge_transcription_diarization(
                    transcription, diarization
                )
            else:
                merged_transcript = transcription["segments"]

            # Build full text
            full_text = transcription["text"]

            # Create analysis result
            analysis = AnalysisResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(transcription),
                findings={
                    "language": transcription.get("language", "unknown"),
                    "duration_seconds": self._get_audio_duration(transcription),
                    "word_count": len(full_text.split()),
                    "segment_count": len(transcription.get("segments", [])),
                    "speaker_count": len(set(
                        s.get("speaker", "unknown") for s in merged_transcript
                    )) if diarization else 1,
                    "diarization_enabled": self.audio_config.enable_diarization,
                },
                raw_output={
                    "segments": merged_transcript,
                    "language_probability": transcription.get("language_probability", 0),
                }
            )

            return packet.with_updates(
                extracted_text=full_text,
                stage=ProcessingStage.OCR_PROCESSED,  # Reusing OCR stage for transcription
                analysis_results=packet.analysis_results + [analysis],
                metadata={
                    **packet.metadata,
                    "audio_transcription": {
                        "language": transcription.get("language"),
                        "segments": merged_transcript,
                    }
                }
            )

        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    async def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio with Whisper."""
        import whisper

        result = self._whisper_model.transcribe(
            audio_path,
            language=self.audio_config.language,
            word_timestamps=True,
            verbose=False
        )

        return result

    async def _diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Run speaker diarization."""
        diarization = self._diarization_pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        return segments

    def _merge_transcription_diarization(
        self,
        transcription: Dict[str, Any],
        diarization: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge transcription segments with speaker labels.
        Assigns each transcription segment to the speaker with the most overlap.
        """
        merged = []

        for segment in transcription.get("segments", []):
            seg_start = segment["start"]
            seg_end = segment["end"]

            # Find best matching speaker
            best_speaker = "UNKNOWN"
            best_overlap = 0

            for diar_seg in diarization:
                overlap_start = max(seg_start, diar_seg["start"])
                overlap_end = min(seg_end, diar_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg["speaker"]

            merged.append({
                "start": seg_start,
                "end": seg_end,
                "text": segment["text"],
                "speaker": best_speaker,
            })

        return merged

    def _calculate_confidence(self, transcription: Dict[str, Any]) -> float:
        """Calculate transcription confidence."""
        # Use language detection probability as base
        lang_prob = transcription.get("language_probability", 0.5)

        # Average segment confidence if available
        segments = transcription.get("segments", [])
        if segments:
            # Whisper doesn't directly provide per-segment confidence,
            # but we can use the compression ratio as a proxy
            avg_compression = np.mean([
                s.get("compression_ratio", 1.0) for s in segments
            ])
            # Lower compression ratio often means cleaner speech
            compression_score = min(1.0, 2.0 / max(avg_compression, 0.5))
        else:
            compression_score = 0.5

        return (lang_prob * 0.4 + compression_score * 0.6)

    def _get_audio_duration(self, transcription: Dict[str, Any]) -> float:
        """Get audio duration from transcription."""
        segments = transcription.get("segments", [])
        if segments:
            return segments[-1].get("end", 0)
        return 0

    async def self_critique(self, result: AnalysisResult) -> float:
        """Self-critique audio transcription quality."""
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize very short transcriptions
        word_count = result.findings.get("word_count", 0)
        if word_count < 10:
            score *= 0.5
        elif word_count < 50:
            score *= 0.8

        # Bonus for successful diarization
        if result.findings.get("diarization_enabled"):
            speaker_count = result.findings.get("speaker_count", 1)
            if speaker_count > 1:
                score = min(1.0, score + 0.1)

        return score
