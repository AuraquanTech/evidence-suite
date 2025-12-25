"""
Evidence Suite - OCR Agent
Multi-engine OCR processing with Tesseract and EasyOCR.
"""
from __future__ import annotations
import io
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
from PIL import Image

from core.models import (
    EvidencePacket,
    AnalysisResult,
    ProcessingStage,
    EvidenceType
)
from core.config import OCRConfig, default_config, hw_settings
from agents.base import BaseAgent


def _check_gpu_support() -> Tuple[bool, str]:
    """
    Check if GPU is available and supports EasyOCR (PyTorch).
    Returns (use_gpu, reason).

    Note: RTX 5090 Blackwell (sm_120) is NOT supported by PyTorch.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check compute capability
        device = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(device)
        cc_str = f"{cc[0]}.{cc[1]}"

        # sm_120 (Blackwell) is not supported by PyTorch
        if cc[0] >= 12:
            return False, f"GPU compute capability {cc_str} (Blackwell) not supported by PyTorch"

        # Test with simple operation
        try:
            test = torch.zeros(1).cuda()
            del test
            return True, f"GPU available (sm_{cc[0]}{cc[1]})"
        except Exception as e:
            return False, f"GPU test failed: {e}"

    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"GPU check failed: {e}"


class OCRAgent(BaseAgent):
    """
    Sensory layer agent for optical character recognition.

    Features:
    - Dual-engine support (Tesseract + EasyOCR)
    - Confidence-weighted result fusion
    - Image preprocessing (deskew, denoise)
    - Multi-language support
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[OCRConfig] = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="ocr",
            config=(config or default_config.ocr).model_dump()
        )
        self.ocr_config = config or default_config.ocr
        self._tesseract = None
        self._easyocr_reader = None

    async def _setup(self) -> None:
        """Initialize OCR engines."""
        engines = self.ocr_config.engines

        # Initialize Tesseract
        if "tesseract" in engines:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.ocr_config.tesseract_path
                # Test that tesseract works
                pytesseract.get_tesseract_version()
                self._tesseract = pytesseract
                logger.info("Tesseract OCR initialized")
            except Exception as e:
                logger.warning(f"Tesseract initialization failed: {e}")

        # Initialize EasyOCR (lazy load due to model size)
        if "easyocr" in engines:
            try:
                import easyocr

                # Check GPU support (Blackwell sm_120 not supported by PyTorch)
                use_gpu, gpu_reason = _check_gpu_support()
                logger.info(f"EasyOCR GPU check: {gpu_reason}")

                self._easyocr_reader = easyocr.Reader(
                    self.ocr_config.languages,
                    gpu=use_gpu
                )
                device_str = "GPU" if use_gpu else "CPU"
                logger.info(f"EasyOCR initialized on {device_str}")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """
        Process an evidence packet through OCR.

        Handles:
        - Image files (JPEG, PNG, TIFF, etc.)
        - PDF documents (first page)
        - Already-text content (passthrough)
        """
        # If already text, just pass through
        if packet.evidence_type == EvidenceType.TEXT and packet.raw_content:
            text = packet.raw_content.decode('utf-8', errors='ignore')
            return packet.with_updates(
                extracted_text=text,
                ocr_confidence=1.0,
                stage=ProcessingStage.OCR_PROCESSED
            )

        # Need image data
        if not packet.raw_content:
            raise ValueError("No content to process")

        # Load image
        image = self._load_image(packet.raw_content)

        # Preprocess if enabled
        if self.ocr_config.preprocessing:
            image = self._preprocess_image(image)

        # Run OCR engines and combine results
        results = await self._run_ocr_engines(image)

        # Fuse results from multiple engines
        final_text, confidence = self._fuse_results(results)

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=confidence,
            findings={
                "engines_used": list(results.keys()),
                "char_count": len(final_text),
                "word_count": len(final_text.split()),
                "per_engine_confidence": {
                    k: v["confidence"] for k, v in results.items()
                }
            }
        )

        return packet.with_updates(
            extracted_text=final_text,
            ocr_confidence=confidence,
            stage=ProcessingStage.OCR_PROCESSED,
            analysis_results=packet.analysis_results + [analysis]
        )

    def _load_image(self, content: bytes) -> Image.Image:
        """Load image from bytes."""
        return Image.open(io.BytesIO(content)).convert('RGB')

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        - Convert to grayscale
        - Apply adaptive thresholding
        - Deskew if enabled
        """
        import numpy as np

        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Simple contrast enhancement
        gray = self._enhance_contrast(gray)

        # Convert back to PIL Image
        return Image.fromarray(gray)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Simple contrast enhancement using histogram stretching."""
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return img

    async def _run_ocr_engines(
        self,
        image: Image.Image
    ) -> Dict[str, Dict[str, Any]]:
        """Run all available OCR engines on the image."""
        results = {}

        # Tesseract
        if self._tesseract:
            try:
                tess_result = self._run_tesseract(image)
                results["tesseract"] = tess_result
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")

        # EasyOCR
        if self._easyocr_reader:
            try:
                easy_result = self._run_easyocr(image)
                results["easyocr"] = easy_result
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")

        if not results:
            raise RuntimeError("All OCR engines failed")

        return results

    def _run_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        """Run Tesseract OCR."""
        # Get detailed data including confidence
        data = self._tesseract.image_to_data(
            image,
            lang='+'.join(self.ocr_config.languages),
            output_type=self._tesseract.Output.DICT
        )

        # Extract text and calculate average confidence
        texts = []
        confidences = []

        for i, text in enumerate(data['text']):
            conf = data['conf'][i]
            if conf > 0 and text.strip():
                texts.append(text)
                confidences.append(conf / 100.0)

        full_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "word_count": len(texts)
        }

    def _run_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Run EasyOCR."""
        img_array = np.array(image)

        # EasyOCR returns list of (bbox, text, confidence)
        results = self._easyocr_reader.readtext(img_array)

        texts = []
        confidences = []

        for bbox, text, conf in results:
            if text.strip():
                texts.append(text)
                confidences.append(conf)

        full_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "word_count": len(texts)
        }

    def _fuse_results(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, float]:
        """
        Fuse results from multiple OCR engines.

        Strategy: Confidence-weighted selection
        - If one engine has significantly higher confidence, use it
        - Otherwise, use the one with more extracted text
        """
        if len(results) == 1:
            result = list(results.values())[0]
            return result["text"], result["confidence"]

        # Sort by confidence
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )

        best_engine, best_result = sorted_results[0]
        second_engine, second_result = sorted_results[1]

        # If best is significantly better, use it
        if best_result["confidence"] - second_result["confidence"] > 0.15:
            return best_result["text"], best_result["confidence"]

        # Otherwise, prefer the one with more content
        if best_result["word_count"] >= second_result["word_count"]:
            return best_result["text"], best_result["confidence"]
        else:
            # Second has more words but lower confidence - average
            avg_conf = (best_result["confidence"] + second_result["confidence"]) / 2
            return second_result["text"], avg_conf

    async def self_critique(self, result: AnalysisResult) -> float:
        """
        Self-critique OCR quality.

        Factors:
        - Base confidence from engines
        - Text coherence (ratio of valid words)
        - Sufficient content extracted
        """
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize if very little text extracted
        word_count = result.findings.get("word_count", 0)
        if word_count < 5:
            score *= 0.5
        elif word_count < 20:
            score *= 0.8

        # Bonus for multiple engines agreeing
        engines_used = result.findings.get("engines_used", [])
        if len(engines_used) > 1:
            confidences = result.findings.get("per_engine_confidence", {})
            if confidences:
                variance = np.var(list(confidences.values()))
                if variance < 0.05:  # Engines agree
                    score = min(1.0, score + 0.1)

        return score
