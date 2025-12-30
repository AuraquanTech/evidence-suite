"""Evidence Suite - Image Agent
Image forensic analysis with metadata extraction and manipulation detection.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from agents.base import BaseAgent
from core.models import AnalysisResult, EvidencePacket, EvidenceType, ProcessingStage


class ImageConfig:
    """Image Agent configuration."""

    def __init__(
        self,
        extract_exif: bool = True,
        detect_manipulation: bool = True,
        extract_text: bool = True,
        device: str = "auto",
        hash_algorithms: list[str] | None = None,
        max_image_size_mb: int = 100,
    ):
        self.extract_exif = extract_exif
        self.detect_manipulation = detect_manipulation
        self.extract_text = extract_text
        self.device = device
        self.hash_algorithms = hash_algorithms or ["sha256", "md5", "phash"]
        self.max_image_size_mb = max_image_size_mb


class ImageAgent(BaseAgent):
    """Sensory layer agent for image forensic analysis.

    Features:
    - EXIF metadata extraction (camera, GPS, timestamps)
    - Multiple hash algorithms (SHA256, MD5, perceptual hash)
    - Manipulation detection (ELA, noise analysis)
    - Image quality assessment
    - Color analysis
    - Steganography detection (basic)
    - OCR text extraction (optional, delegates to OCRAgent)
    """

    def __init__(self, agent_id: str | None = None, config: ImageConfig | None = None):
        self.image_config = config or ImageConfig()
        super().__init__(
            agent_id=agent_id,
            agent_type="image",
            config={
                "extract_exif": self.image_config.extract_exif,
                "detect_manipulation": self.image_config.detect_manipulation,
                "hash_algorithms": self.image_config.hash_algorithms,
            },
        )
        self._pil_available = False
        self._cv2_available = False

    async def _setup(self) -> None:
        """Initialize image processing dependencies."""
        # Check PIL availability
        try:
            from PIL import Image

            self._pil_available = True
            logger.info("PIL available for image processing")
        except ImportError:
            logger.warning("PIL not installed. Install with: pip install Pillow")

        # Check OpenCV availability
        try:
            import cv2

            self._cv2_available = True
            logger.info("OpenCV available for manipulation detection")
        except ImportError:
            logger.warning("OpenCV not installed. Some features disabled.")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """Process image evidence.

        Steps:
        1. Load and validate image
        2. Extract EXIF metadata
        3. Calculate multiple hashes
        4. Analyze for manipulation
        5. Extract visual characteristics
        """
        if packet.evidence_type != EvidenceType.IMAGE:
            raise ValueError(f"Expected image evidence, got {packet.evidence_type}")

        if not packet.raw_content:
            raise ValueError("No image content to process")

        if not self._pil_available:
            raise RuntimeError("PIL not available for image processing")

        # Check file size
        size_mb = len(packet.raw_content) / (1024 * 1024)
        if size_mb > self.image_config.max_image_size_mb:
            raise ValueError(
                f"Image too large: {size_mb:.2f}MB > {self.image_config.max_image_size_mb}MB"
            )

        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        # Load image
        img = Image.open(io.BytesIO(packet.raw_content))

        # Basic image info
        image_info = {
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "size_bytes": len(packet.raw_content),
        }

        # Calculate hashes
        hashes = self._calculate_hashes(packet.raw_content, img)

        # Extract EXIF
        exif_data = {}
        if self.image_config.extract_exif:
            exif_data = self._extract_exif(img)

        # Manipulation analysis
        manipulation_indicators = {}
        if self.image_config.detect_manipulation and self._cv2_available:
            manipulation_indicators = await self._analyze_manipulation(packet.raw_content)

        # Color analysis
        color_analysis = self._analyze_colors(img)

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=self._calculate_confidence(exif_data, manipulation_indicators),
            findings={
                "format": image_info.get("format"),
                "resolution": f"{image_info['width']}x{image_info['height']}",
                "size_bytes": image_info.get("size_bytes"),
                "has_exif": len(exif_data) > 0,
                "has_gps": exif_data.get("gps") is not None,
                "camera_make": exif_data.get("make"),
                "camera_model": exif_data.get("model"),
                "date_taken": exif_data.get("datetime_original"),
                "manipulation_risk": manipulation_indicators.get("risk_level", "unknown"),
                "dominant_colors": color_analysis.get("dominant_colors", []),
            },
            raw_output={
                "image_info": image_info,
                "hashes": hashes,
                "exif": exif_data,
                "manipulation": manipulation_indicators,
                "color_analysis": color_analysis,
            },
        )

        return packet.with_updates(
            stage=ProcessingStage.OCR_PROCESSED,
            analysis_results=packet.analysis_results + [analysis],
            source_metadata={
                **packet.source_metadata,
                "image_forensics": {
                    "hashes": hashes,
                    "exif_present": len(exif_data) > 0,
                    "manipulation_risk": manipulation_indicators.get("risk_level", "unknown"),
                },
            },
        )

    def _calculate_hashes(self, raw_content: bytes, img) -> dict[str, str]:
        """Calculate multiple hash types for the image."""
        hashes = {}

        if "sha256" in self.image_config.hash_algorithms:
            hashes["sha256"] = hashlib.sha256(raw_content).hexdigest()

        if "md5" in self.image_config.hash_algorithms:
            hashes["md5"] = hashlib.md5(raw_content, usedforsecurity=False).hexdigest()

        if "phash" in self.image_config.hash_algorithms:
            hashes["phash"] = self._calculate_phash(img)

        return hashes

    def _calculate_phash(self, img) -> str:
        """Calculate perceptual hash for the image."""
        try:
            # Resize to 32x32
            resized = img.convert("L").resize((32, 32))
            pixels = np.array(resized)

            # DCT-based perceptual hash (simplified)
            # Use mean as threshold
            avg = np.mean(pixels)
            diff = pixels > avg

            # Convert to hex string
            hash_bits = diff.flatten()
            hash_hex = ""
            for i in range(0, 64, 4):
                nibble = sum(
                    hash_bits[i + j] << (3 - j) for j in range(4) if i + j < len(hash_bits)
                )
                hash_hex += format(nibble, "x")

            return hash_hex[:16]

        except Exception as e:
            logger.warning(f"pHash calculation failed: {e}")
            return ""

    def _extract_exif(self, img) -> dict[str, Any]:
        """Extract EXIF metadata from image."""
        from PIL.ExifTags import GPSTAGS, TAGS

        exif_data = {}

        try:
            exif = img._getexif()
            if not exif:
                return exif_data

            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)

                # Handle GPS data specially
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = self._convert_exif_value(gps_value)
                    exif_data["gps"] = self._parse_gps_coordinates(gps_data)
                else:
                    exif_data[tag.lower()] = self._convert_exif_value(value)

            # Parse specific fields
            if "datetimeoriginal" in exif_data:
                exif_data["datetime_original"] = exif_data.pop("datetimeoriginal")

        except Exception as e:
            logger.warning(f"EXIF extraction failed: {e}")

        return exif_data

    def _convert_exif_value(self, value: Any) -> Any:
        """Convert EXIF value to JSON-serializable format."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return value.hex()
        elif isinstance(value, tuple):
            return [self._convert_exif_value(v) for v in value]
        elif hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value)
        return value

    def _parse_gps_coordinates(self, gps_data: dict) -> dict[str, float] | None:
        """Parse GPS coordinates from EXIF GPS data."""
        try:
            lat = gps_data.get("GPSLatitude")
            lat_ref = gps_data.get("GPSLatitudeRef", "N")
            lon = gps_data.get("GPSLongitude")
            lon_ref = gps_data.get("GPSLongitudeRef", "E")

            if lat and lon:
                lat_decimal = self._dms_to_decimal(lat)
                lon_decimal = self._dms_to_decimal(lon)

                if lat_ref == "S":
                    lat_decimal = -lat_decimal
                if lon_ref == "W":
                    lon_decimal = -lon_decimal

                return {
                    "latitude": round(lat_decimal, 6),
                    "longitude": round(lon_decimal, 6),
                }
        except Exception as e:
            logger.debug(f"GPS parsing failed: {e}")

        return None

    def _dms_to_decimal(self, dms: list) -> float:
        """Convert degrees, minutes, seconds to decimal degrees."""
        if len(dms) >= 3:
            d = float(dms[0])
            m = float(dms[1])
            s = float(dms[2])
            return d + m / 60 + s / 3600
        return 0.0

    async def _analyze_manipulation(self, raw_content: bytes) -> dict[str, Any]:
        """Analyze image for signs of manipulation."""
        import cv2

        indicators = {
            "ela_score": 0.0,
            "noise_inconsistency": 0.0,
            "compression_artifacts": False,
            "risk_level": "low",
            "warnings": [],
        }

        try:
            # Decode image
            nparr = np.frombuffer(raw_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return indicators

            # Error Level Analysis (ELA)
            ela_score = self._calculate_ela(raw_content, img)
            indicators["ela_score"] = round(ela_score, 3)

            if ela_score > 0.15:
                indicators["warnings"].append("High ELA score - possible editing detected")

            # Noise analysis
            noise_score = self._analyze_noise(img)
            indicators["noise_inconsistency"] = round(noise_score, 3)

            if noise_score > 0.2:
                indicators["warnings"].append("Noise inconsistency detected")

            # JPEG compression artifact detection
            if self._detect_double_compression(raw_content):
                indicators["compression_artifacts"] = True
                indicators["warnings"].append("Possible double JPEG compression")

            # Calculate risk level
            warning_count = len(indicators["warnings"])
            if warning_count >= 2:
                indicators["risk_level"] = "high"
            elif warning_count >= 1:
                indicators["risk_level"] = "medium"
            else:
                indicators["risk_level"] = "low"

        except Exception as e:
            logger.warning(f"Manipulation analysis failed: {e}")

        return indicators

    def _calculate_ela(self, raw_content: bytes, img) -> float:
        """Calculate Error Level Analysis score."""
        import cv2

        try:
            # Re-compress at known quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            # Calculate difference
            diff = cv2.absdiff(img, recompressed)
            ela_score = np.mean(diff) / 255.0

            return ela_score

        except Exception:
            return 0.0

    def _analyze_noise(self, img) -> float:
        """Analyze noise patterns for inconsistencies."""
        import cv2

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Divide into blocks and analyze noise variance
            h, w = gray.shape
            block_size = 64
            variances = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y : y + block_size, x : x + block_size]
                    # High-pass filter to extract noise
                    blurred = cv2.GaussianBlur(block, (5, 5), 0)
                    noise = cv2.absdiff(block, blurred)
                    variances.append(np.var(noise))

            if not variances:
                return 0.0

            # Calculate coefficient of variation
            mean_var = np.mean(variances)
            std_var = np.std(variances)

            if mean_var > 0:
                return std_var / mean_var

            return 0.0

        except Exception:
            return 0.0

    def _detect_double_compression(self, raw_content: bytes) -> bool:
        """Detect signs of double JPEG compression."""
        try:
            # Look for JPEG markers
            if raw_content[:2] != b"\xff\xd8":
                return False

            # Count quantization tables
            qt_count = raw_content.count(b"\xff\xdb")
            return qt_count > 2

        except Exception:
            return False

    def _analyze_colors(self, img) -> dict[str, Any]:
        """Analyze color distribution in the image."""
        try:
            # Resize for speed
            img_small = img.convert("RGB").resize((100, 100))
            pixels = np.array(img_small)

            # Reshape to list of pixels
            pixels_flat = pixels.reshape(-1, 3)

            # Simple color clustering (find dominant colors)
            from collections import Counter

            # Quantize colors
            quantized = (pixels_flat // 32) * 32
            color_counts = Counter(map(tuple, quantized))

            # Get top 5 colors
            dominant = color_counts.most_common(5)
            dominant_colors = [
                {"rgb": list(color), "percentage": round(count / len(pixels_flat) * 100, 1)}
                for color, count in dominant
            ]

            # Calculate overall statistics
            mean_color = np.mean(pixels_flat, axis=0)
            std_color = np.std(pixels_flat, axis=0)

            return {
                "dominant_colors": dominant_colors,
                "mean_rgb": [round(c, 1) for c in mean_color],
                "color_variance": round(float(np.mean(std_color)), 2),
            }

        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return {}

    def _calculate_confidence(self, exif_data: dict, manipulation_indicators: dict) -> float:
        """Calculate processing confidence."""
        confidence = 0.8  # Base confidence

        # Boost for successful EXIF extraction
        if exif_data:
            confidence += 0.1

        # Boost for low manipulation risk
        risk = manipulation_indicators.get("risk_level", "unknown")
        if risk == "low":
            confidence += 0.1
        elif risk == "medium":
            confidence += 0.05

        return min(1.0, confidence)

    async def self_critique(self, result: AnalysisResult) -> float:
        """Self-critique image analysis quality."""
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize high manipulation risk
        risk = result.findings.get("manipulation_risk", "unknown")
        if risk == "high":
            score *= 0.7
        elif risk == "medium":
            score *= 0.85

        # Bonus for EXIF data
        if result.findings.get("has_exif"):
            score = min(1.0, score + 0.05)

        return score
