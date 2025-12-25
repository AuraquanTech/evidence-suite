# Evidence Suite: Savant Genesis Edition

Forensic behavioral intelligence platform for analyzing communication patterns and detecting manipulation tactics in digital evidence.

## Features

- **Multi-Agent Pipeline**: OCR → Behavioral Analysis → Fusion
- **Pattern Detection**: DARVO, gaslighting, manipulation, deception indicators
- **Sentiment Analysis**: VADER + BERT embeddings with CPU fallback
- **Chain of Custody**: SHA-256 verification (FRE 707 compliant)
- **RTX 5090 Blackwell Support**: Native sm_120 via CuPy/cuBLAS
- **Thermal Management**: Laptop GPU throttling and power awareness
- **Court-Ready Exports**: Visualization charts for legal proceedings

## Quick Start

```bash
# Clone
git clone https://github.com/AuraquanTech/evidence-suite.git
cd evidence-suite

# Install dependencies
pip install -r requirements.txt

# Run pipeline with sample data
python run_pipeline.py

# Run GPU benchmark
python scripts/benchmark_pipeline.py --iterations 50

# Run stress test
python scripts/tensor_core_stress_cupy.py --sustained-minutes 1
```

## Architecture

```
evidence-suite/
├── agents/
│   ├── ocr_agent.py        # Tesseract + EasyOCR dual-engine
│   ├── behavioral_agent.py  # VADER + BERT + keyword patterns
│   └── fusion_agent.py      # Hybrid late fusion scoring
├── core/
│   ├── models.py           # EvidencePacket, ChainOfCustody
│   ├── hardware_monitor.py # GPU thermal/VRAM monitoring
│   └── visualizations.py   # Court presentation charts
├── scripts/
│   ├── benchmark_pipeline.py      # Performance benchmarking
│   └── tensor_core_stress_cupy.py # GPU stress test
├── pipeline.py             # Main orchestrator
└── run_pipeline.py         # Sample data runner
```

## Behavioral Detection

The system detects these manipulation patterns:

| Pattern | Description | Detection Method |
|---------|-------------|------------------|
| **DARVO** | Deny, Attack, Reverse Victim/Offender | Keyword patterns + sentiment inversion |
| **Gaslighting** | Reality denial, memory questioning | Phrase matching + negation analysis |
| **Manipulation** | Guilt-tripping, emotional leverage | Conditional phrases + obligation markers |
| **Deception** | Inconsistency indicators | Cross-reference analysis |

## RTX 5090 Blackwell Support

PyTorch lacks sm_120 kernels for Blackwell architecture. This project uses CuPy with cuBLAS for native GPU acceleration:

```python
# CuPy uses cuBLAS which has native Blackwell support
import cupy as cp
a = cp.random.randn(4096, 4096, dtype=cp.float32)
b = cp.random.randn(4096, 4096, dtype=cp.float32)
c = cp.matmul(a, b)  # Runs on RTX 5090 at 6+ TFLOPS
```

**Benchmark Results (RTX 5090 Laptop GPU):**
- Compute Capability: 12.0 (sm_120)
- FP32 Performance: 6.3 TFLOPS
- FP16 Performance: 12.6 TFLOPS (2x speedup)
- Memory Bandwidth: 568 GB/s
- Thermals: 55-62°C under load

## Pipeline Output

```
============================================================
Evidence ID: 824e2294-1b41-4bb0-8de1-abc4e992c14b
Case ID: test_darvo_pattern_20251225
Status: SUCCESS
------------------------------------------------------------
BEHAVIORAL INDICATORS:
  Sentiment (compound): -0.488
  DARVO Score: 1.000
  Gaslighting Score: 0.000
  Manipulation Score: 0.400
  Primary Behavior: darvo
------------------------------------------------------------
FUSION RESULTS:
  Fused Score: 0.379
  Classification: darvo
------------------------------------------------------------
CHAIN OF CUSTODY:
  Entries: 2
  Chain Valid: True
  Chain Hash: 0fdc8619d55d64e7...
============================================================
```

## Configuration

Edit `config.yaml` for hardware-specific settings:

```yaml
hardware:
  thermal_limits:
    warning_c: 82
    hot_c: 85
    critical_c: 90
  vram_budget_mb: 20480

pipeline:
  batch_size: 96
  max_workers: 16
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x
- For RTX 5090: CuPy 13.x (PyTorch not supported yet)

## License

MIT

## Acknowledgments

Built with Claude Code on an RTX 5090 Laptop GPU.
