"""Evidence Suite - Video Agent
Video processing with frame extraction, audio transcription, and scene detection.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
from loguru import logger

from agents.base import BaseAgent
from core.models import AnalysisResult, EvidencePacket, EvidenceType, ProcessingStage


class VideoConfig:
    """Video Agent configuration."""

    def __init__(
        self,
        extract_audio: bool = True,
        extract_frames: bool = True,
        frame_interval_seconds: float = 1.0,
        max_frames: int = 100,
        whisper_model: str = "base",
        device: str = "auto",
        detect_faces: bool = False,
        scene_detection: bool = True,
        scene_threshold: float = 30.0,
    ):
        self.extract_audio = extract_audio
        self.extract_frames = extract_frames
        self.frame_interval_seconds = frame_interval_seconds
        self.max_frames = max_frames
        self.whisper_model = whisper_model
        self.device = device
        self.detect_faces = detect_faces
        self.scene_detection = scene_detection
        self.scene_threshold = scene_threshold


def _check_gpu_available() -> tuple[bool, str]:
    """Check if GPU is available for video processing."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        device = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(device)

        if cc[0] >= 12:
            return False, f"Blackwell GPU (sm_{cc[0]}{cc[1]}) - using CPU"

        return True, f"GPU available (sm_{cc[0]}{cc[1]})"

    except Exception as e:
        return False, f"GPU check failed: {e}"


class VideoAgent(BaseAgent):
    """Sensory layer agent for video forensic analysis.

    Features:
    - Frame extraction at configurable intervals
    - Audio track extraction and transcription (via Whisper)
    - Scene change detection
    - Video metadata extraction
    - Duration and format analysis
    - Optional face detection (disabled by default for privacy)
    """

    def __init__(self, agent_id: str | None = None, config: VideoConfig | None = None):
        self.video_config = config or VideoConfig()
        super().__init__(
            agent_id=agent_id,
            agent_type="video",
            config={
                "extract_audio": self.video_config.extract_audio,
                "extract_frames": self.video_config.extract_frames,
                "frame_interval": self.video_config.frame_interval_seconds,
                "scene_detection": self.video_config.scene_detection,
            },
        )
        self._whisper_model = None
        self._device = "cpu"
        self._cv2_available = False
        self._ffmpeg_available = False

    async def _setup(self) -> None:
        """Initialize video processing dependencies."""
        use_gpu, gpu_reason = _check_gpu_available()
        self._device = "cuda" if use_gpu else "cpu"
        logger.info(f"Video Agent GPU check: {gpu_reason}")

        # Check OpenCV availability
        try:
            import cv2

            self._cv2_available = True
            logger.info("OpenCV available for video processing")
        except ImportError:
            logger.warning("OpenCV not installed. Install with: pip install opencv-python")

        # Check FFmpeg availability
        try:
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._ffmpeg_available = True
                logger.info("FFmpeg available for audio extraction")
        except Exception:
            logger.warning("FFmpeg not available. Audio extraction disabled.")

        # Initialize Whisper for audio transcription
        if self.video_config.extract_audio and self._ffmpeg_available:
            await self._init_whisper()

    async def _init_whisper(self) -> None:
        """Initialize Whisper model for audio transcription."""
        try:
            import whisper

            model_name = self.video_config.whisper_model
            logger.info(f"Loading Whisper model: {model_name}")

            self._whisper_model = whisper.load_model(model_name, device=self._device)
            logger.info(f"Whisper {model_name} loaded on {self._device}")

        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")
        except Exception as e:
            logger.warning(f"Whisper initialization failed: {e}")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """Process video evidence.

        Steps:
        1. Save video to temp file
        2. Extract video metadata
        3. Extract key frames
        4. Extract and transcribe audio
        5. Detect scene changes
        """
        if packet.evidence_type != EvidenceType.VIDEO:
            raise ValueError(f"Expected video evidence, got {packet.evidence_type}")

        if not packet.raw_content:
            raise ValueError("No video content to process")

        if not self._cv2_available:
            raise RuntimeError("OpenCV not available for video processing")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(packet.raw_content)
            video_path = f.name

        try:
            import cv2

            # Get video info
            cap = cv2.VideoCapture(video_path)
            video_info = self._extract_video_info(cap)

            # Extract frames
            frames_info = []
            if self.video_config.extract_frames:
                frames_info = self._extract_frames(cap, video_info)

            # Detect scene changes
            scene_changes = []
            if self.video_config.scene_detection:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                scene_changes = self._detect_scene_changes(cap, video_info)

            cap.release()

            # Extract and transcribe audio
            transcription = None
            if self.video_config.extract_audio and self._ffmpeg_available:
                transcription = await self._extract_and_transcribe_audio(video_path)

            # Build full text from transcription
            full_text = ""
            if transcription:
                full_text = transcription.get("text", "")

            # Create analysis result
            analysis = AnalysisResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(video_info, transcription),
                findings={
                    "duration_seconds": video_info.get("duration_seconds", 0),
                    "fps": video_info.get("fps", 0),
                    "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                    "total_frames": video_info.get("total_frames", 0),
                    "extracted_frame_count": len(frames_info),
                    "scene_change_count": len(scene_changes),
                    "has_audio": transcription is not None,
                    "audio_language": transcription.get("language") if transcription else None,
                    "word_count": len(full_text.split()) if full_text else 0,
                },
                raw_output={
                    "video_info": video_info,
                    "frames": frames_info[:20],  # Limit stored frame data
                    "scene_changes": scene_changes[:50],
                    "transcription_segments": (
                        transcription.get("segments", [])[:100] if transcription else []
                    ),
                },
            )

            return packet.with_updates(
                extracted_text=full_text,
                stage=ProcessingStage.OCR_PROCESSED,
                analysis_results=packet.analysis_results + [analysis],
                source_metadata={
                    **packet.source_metadata,
                    "video_analysis": {
                        "duration": video_info.get("duration_seconds"),
                        "resolution": f"{video_info.get('width')}x{video_info.get('height')}",
                        "scene_changes": len(scene_changes),
                    },
                },
            )

        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def _extract_video_info(self, cap) -> dict[str, Any]:
        """Extract video metadata."""
        import cv2

        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": (
                cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                if cap.get(cv2.CAP_PROP_FPS) > 0
                else 0
            ),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

    def _extract_frames(self, cap, video_info: dict) -> list[dict[str, Any]]:
        """Extract key frames at specified intervals."""
        import cv2

        frames = []
        fps = video_info.get("fps", 30)
        frame_interval = int(fps * self.video_config.frame_interval_seconds)
        frame_interval = max(1, frame_interval)

        frame_count = 0
        extracted = 0

        while extracted < self.video_config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Calculate frame statistics
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                std_brightness = float(np.std(gray))

                frames.append(
                    {
                        "frame_number": frame_count,
                        "timestamp_seconds": frame_count / fps if fps > 0 else 0,
                        "mean_brightness": round(mean_brightness, 2),
                        "std_brightness": round(std_brightness, 2),
                    }
                )
                extracted += 1

            frame_count += 1

        return frames

    def _detect_scene_changes(self, cap, video_info: dict) -> list[dict[str, Any]]:
        """Detect scene changes using frame difference analysis."""
        import cv2

        scene_changes = []
        fps = video_info.get("fps", 30)
        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = float(np.mean(diff))

                if diff_score > self.video_config.scene_threshold:
                    scene_changes.append(
                        {
                            "frame_number": frame_count,
                            "timestamp_seconds": frame_count / fps if fps > 0 else 0,
                            "diff_score": round(diff_score, 2),
                        }
                    )

            prev_frame = gray
            frame_count += 1

            # Skip frames for efficiency
            if frame_count % 5 != 0:
                continue

        return scene_changes

    async def _extract_and_transcribe_audio(self, video_path: str) -> dict[str, Any] | None:
        """Extract audio and transcribe with Whisper."""
        if not self._whisper_model:
            return None

        # Create temp audio file
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            import subprocess

            # Extract audio with FFmpeg
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    audio_path,
                ],
                check=False,
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0 or not os.path.exists(audio_path):
                logger.warning("Audio extraction failed")
                return None

            # Transcribe
            transcription = self._whisper_model.transcribe(
                audio_path, word_timestamps=True, verbose=False
            )

            return transcription

        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
            return None

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def _calculate_confidence(self, video_info: dict, transcription: dict | None) -> float:
        """Calculate processing confidence."""
        confidence = 0.7  # Base confidence

        # Boost for successful video parsing
        if video_info.get("total_frames", 0) > 0:
            confidence += 0.1

        # Boost for successful audio transcription
        if transcription:
            confidence += 0.1
            lang_prob = transcription.get("language_probability", 0)
            confidence += lang_prob * 0.1

        return min(1.0, confidence)

    async def self_critique(self, result: AnalysisResult) -> float:
        """Self-critique video processing quality."""
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize very short videos
        duration = result.findings.get("duration_seconds", 0)
        if duration < 1:
            score *= 0.5
        elif duration < 5:
            score *= 0.8

        # Bonus for scene detection
        scene_count = result.findings.get("scene_change_count", 0)
        if scene_count > 0:
            score = min(1.0, score + 0.05)

        # Bonus for audio transcription
        if result.findings.get("has_audio"):
            score = min(1.0, score + 0.1)

        return score
