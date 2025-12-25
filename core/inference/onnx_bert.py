"""
Evidence Suite - ONNX BERT Inference
GPU-accelerated BERT inference using ONNX Runtime for RTX 5090 Blackwell.

ONNX Runtime has native support for Blackwell architecture (sm_120) via cuDNN,
unlike PyTorch which lacks sm_120 kernel support.
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from loguru import logger


class ONNXBertInference:
    """
    ONNX Runtime-based BERT inference for Blackwell GPUs.

    Features:
    - Native sm_120 support via cuDNN
    - CUDAExecutionProvider for GPU acceleration
    - Automatic fallback to CPU
    - Dynamic batching
    - FP16 inference option for 2x speedup
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        use_gpu: bool = True,
        use_fp16: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "evidence_suite" / "onnx")

        self._session = None
        self._tokenizer = None
        self._device = "cpu"
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize ONNX session and tokenizer."""
        if self._is_initialized:
            return True

        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)

            # Initialize tokenizer
            self._init_tokenizer()

            # Initialize ONNX session
            self._init_session()

            self._is_initialized = True
            logger.info(f"ONNX BERT initialized on {self._device}")
            return True

        except Exception as e:
            logger.error(f"ONNX BERT initialization failed: {e}")
            return False

    def _init_tokenizer(self):
        """Initialize HuggingFace tokenizer."""
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        logger.debug("Tokenizer initialized")

    def _init_session(self):
        """Initialize ONNX Runtime session."""
        import onnxruntime as ort

        # Get or export ONNX model
        onnx_path = self._get_or_export_onnx()

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 4

        # Select execution providers
        providers = []

        if self.use_gpu:
            # Check CUDA availability
            try:
                cuda_available = ort.get_available_providers()
                if "CUDAExecutionProvider" in cuda_available:
                    cuda_options = {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB limit
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    }
                    providers.append(("CUDAExecutionProvider", cuda_options))
                    self._device = "cuda"
                    logger.info("Using CUDA Execution Provider")

                elif "TensorrtExecutionProvider" in cuda_available:
                    trt_options = {
                        "device_id": 0,
                        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                        "trt_fp16_enable": self.use_fp16,
                    }
                    providers.append(("TensorrtExecutionProvider", trt_options))
                    self._device = "tensorrt"
                    logger.info("Using TensorRT Execution Provider")

            except Exception as e:
                logger.warning(f"GPU provider initialization failed: {e}")

        # Always add CPU fallback
        providers.append("CPUExecutionProvider")

        # Create session
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )

        # Log actual providers being used
        active_providers = self._session.get_providers()
        logger.info(f"Active providers: {active_providers}")

    def _get_or_export_onnx(self) -> str:
        """Get cached ONNX model or export from PyTorch."""
        model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
        onnx_path = os.path.join(self.cache_dir, f"{model_safe_name}.onnx")

        if os.path.exists(onnx_path):
            logger.debug(f"Using cached ONNX model: {onnx_path}")
            return onnx_path

        logger.info(f"Exporting {self.model_name} to ONNX...")
        self._export_to_onnx(onnx_path)

        return onnx_path

    def _export_to_onnx(self, output_path: str):
        """Export HuggingFace model to ONNX format."""
        import torch
        from transformers import AutoModel

        # Load PyTorch model
        model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        model.eval()

        # Create dummy inputs
        batch_size = 1
        seq_length = self.max_length

        dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "token_type_ids": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
                "pooler_output": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        logger.info(f"ONNX model exported to: {output_path}")

        # Optional: Optimize ONNX model
        self._optimize_onnx(output_path)

    def _optimize_onnx(self, onnx_path: str):
        """Optimize ONNX model for inference."""
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions

            opt_options = FusionOptions("bert")
            opt_options.enable_gelu_approximation = True

            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type="bert",
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options,
            )

            # Overwrite with optimized model
            optimized_model.save_model_to_file(onnx_path)
            logger.info("ONNX model optimized")

        except ImportError:
            logger.warning("onnxruntime-tools not available, skipping optimization")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for inference

        Returns:
            Embeddings array of shape (n_texts, hidden_size)
        """
        if not self._is_initialized:
            if not self.initialize():
                raise RuntimeError("ONNX BERT initialization failed")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            # Run inference
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs.get(
                    "token_type_ids",
                    np.zeros_like(inputs["input_ids"])
                ).astype(np.int64),
            }

            outputs = self._session.run(None, ort_inputs)

            # Get [CLS] token embedding (first token)
            cls_embeddings = outputs[0][:, 0, :]
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text])[0]

    def get_device(self) -> str:
        """Return current device."""
        return self._device

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._device in ["cuda", "tensorrt"]


# Singleton instance
_bert_instance: Optional[ONNXBertInference] = None


def get_bert_inference(
    model_name: str = "bert-base-uncased",
    use_gpu: bool = True,
    use_fp16: bool = False,
) -> ONNXBertInference:
    """
    Get singleton BERT inference instance.

    Args:
        model_name: HuggingFace model name
        use_gpu: Whether to use GPU acceleration
        use_fp16: Whether to use FP16 inference

    Returns:
        Initialized ONNXBertInference instance
    """
    global _bert_instance

    if _bert_instance is None:
        _bert_instance = ONNXBertInference(
            model_name=model_name,
            use_gpu=use_gpu,
            use_fp16=use_fp16,
        )
        _bert_instance.initialize()

    return _bert_instance
