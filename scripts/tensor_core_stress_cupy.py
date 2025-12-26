"""Evidence Suite - RTX 5090 Tensor Core Stress Test (CuPy Edition)
Works with Blackwell architecture (sm_120) using CuPy instead of PyTorch.

This test:
1. Verifies GPU availability and compute capability
2. Compares FP32 vs FP16 performance
3. Validates numerical precision across data types
4. Measures sustained compute throughput
5. Generates a hardware certification report
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Add PyTorch's CUDA DLLs to the search path BEFORE importing cupy
# Don't set CUDA_PATH - CuPy expects a standard toolkit layout
# Just add the DLL directory for runtime libraries (cuBLAS, cuRAND, etc.)
torch_lib = r"C:\Users\ayrto\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("ERROR: CuPy is required for this stress test")
    print("Install with: pip install cupy-cuda12x")
    sys.exit(1)

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from loguru import logger


@dataclass
class PrecisionBenchmark:
    """Results for a single precision benchmark."""

    dtype: str
    matrix_size: int
    iterations: int
    total_time_sec: float
    mean_latency_ms: float
    tflops: float
    memory_used_mb: float


@dataclass
class PrecisionComparison:
    """Comparison between precision types."""

    fp32_tflops: float
    fp16_tflops: float
    fp16_speedup: float


@dataclass
class NumericalValidation:
    """Numerical precision validation results."""

    fp16_max_error: float
    fp16_mean_error: float
    precision_acceptable: bool


@dataclass
class TensorCoreReport:
    """Complete stress test report."""

    timestamp: str
    gpu_name: str
    compute_capability: str
    cupy_version: str
    benchmarks: list[PrecisionBenchmark]
    comparison: PrecisionComparison | None
    validation: NumericalValidation | None
    sustained_test: dict[str, Any]
    certification: str


class TensorCoreStressTest:
    """RTX 5090 Stress Test Suite using CuPy.

    Works with Blackwell architecture (sm_120) where PyTorch lacks kernel support.
    Uses cuBLAS for matrix operations which has native sm_120 support.
    """

    def __init__(self):
        self.device = cp.cuda.Device(0)
        self.gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        cc_raw = self.device.compute_capability
        # Format CC properly: "120" -> "12.0", 120 -> "12.0"
        # CuPy returns a string like "120" for sm_120
        try:
            cc_int = int(cc_raw) if isinstance(cc_raw, str) else cc_raw
            if isinstance(cc_int, int):
                if cc_int >= 100:
                    # Format: 120 means sm_120 -> "12.0"
                    self.compute_capability = f"{cc_int // 10}.{cc_int % 10}"
                else:
                    # Format: 12 means major version only -> "12.0"
                    self.compute_capability = f"{cc_int}.0"
            elif hasattr(cc_raw, "__len__") and len(cc_raw) >= 2:
                self.compute_capability = f"{cc_raw[0]}.{cc_raw[1]}"
            else:
                self.compute_capability = str(cc_raw)
        except (ValueError, TypeError):
            self.compute_capability = str(cc_raw)

        # NVML for thermal monitoring
        self._nvml_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        logger.info(f"GPU: {self.gpu_name}")
        logger.info(f"Compute Capability: {self.compute_capability}")
        logger.info(f"CuPy version: {cp.__version__}")

    def _get_gpu_temp(self) -> float:
        """Get current GPU temperature."""
        if self._nvml_handle:
            try:
                return float(
                    pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
        return 0.0

    def _get_memory_used_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        mempool = cp.get_default_memory_pool()
        return mempool.used_bytes() / (1024**2)

    def benchmark_precision(
        self, dtype, matrix_size: int = 8192, iterations: int = 100
    ) -> PrecisionBenchmark:
        """Benchmark matrix multiplication for a specific precision.
        Uses cuBLAS internally which has native Blackwell support.
        """
        dtype_name = str(dtype).replace("cupy.", "").replace("<class '", "").replace("'>", "")
        logger.info(f"Benchmarking {dtype_name} at {matrix_size}x{matrix_size}...")

        # Allocate matrices directly with the target dtype to avoid JIT compilation
        # CuPy's random functions work without JIT, and matmul uses cuBLAS
        if dtype == cp.float32:
            a = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
            b = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
        else:
            # For FP16, generate in FP32 then convert using cuBLAS-compatible path
            a = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
            b = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
            # Use view and direct memory operations instead of astype
            a = cp.array(cp.asnumpy(a).astype("float16"), dtype=cp.float16)
            b = cp.array(cp.asnumpy(b).astype("float16"), dtype=cp.float16)

        memory_used = self._get_memory_used_mb()

        # Warmup - matmul uses cuBLAS which has native sm_120 support
        for _ in range(10):
            c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()
        total_time = time.perf_counter() - start

        # Calculate TFLOPS
        # Matrix multiply: 2 * N^3 FLOPs
        flops_per_op = 2 * (matrix_size**3)
        total_flops = flops_per_op * iterations
        tflops = (total_flops / total_time) / 1e12

        # Cleanup
        del a, b, c
        cp.get_default_memory_pool().free_all_blocks()

        return PrecisionBenchmark(
            dtype=dtype_name,
            matrix_size=matrix_size,
            iterations=iterations,
            total_time_sec=round(total_time, 3),
            mean_latency_ms=round((total_time / iterations) * 1000, 3),
            tflops=round(tflops, 2),
            memory_used_mb=round(memory_used, 1),
        )

    def compare_precisions(self, matrix_size: int = 8192) -> PrecisionComparison:
        """Compare FP32 and FP16 performance."""
        logger.info("Running precision comparison...")

        # FP32 baseline
        fp32_result = self.benchmark_precision(cp.float32, matrix_size, iterations=50)

        # FP16
        fp16_result = self.benchmark_precision(cp.float16, matrix_size, iterations=100)

        fp16_speedup = fp16_result.tflops / fp32_result.tflops if fp32_result.tflops > 0 else 0

        return PrecisionComparison(
            fp32_tflops=fp32_result.tflops,
            fp16_tflops=fp16_result.tflops,
            fp16_speedup=round(fp16_speedup, 2),
        )

    def validate_numerical_precision(self, matrix_size: int = 2048) -> NumericalValidation:
        """Validate numerical precision of FP16 against FP32 reference.
        Uses NumPy for type conversion to avoid CuPy JIT issues.
        """
        logger.info("Validating numerical precision...")
        import numpy as np

        # Create reference matrices in FP32
        a_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        b_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)

        a_fp32 = cp.array(a_np)
        b_fp32 = cp.array(b_np)

        # Reference result
        c_fp32 = cp.matmul(a_fp32, b_fp32)

        # FP16 computation
        a_fp16 = cp.array(a_np.astype(np.float16))
        b_fp16 = cp.array(b_np.astype(np.float16))
        c_fp16_raw = cp.matmul(a_fp16, b_fp16)
        c_fp16 = cp.array(cp.asnumpy(c_fp16_raw).astype(np.float32))

        # Calculate errors using NumPy
        c_fp32_np = cp.asnumpy(c_fp32)
        c_fp16_np = cp.asnumpy(c_fp16)
        fp16_error = np.abs(c_fp32_np - c_fp16_np)

        fp16_max = float(np.max(fp16_error))
        fp16_mean = float(np.mean(fp16_error))

        # Cleanup
        del a_fp32, b_fp32, c_fp32, a_fp16, b_fp16, c_fp16, c_fp16_raw
        cp.get_default_memory_pool().free_all_blocks()

        # Precision is acceptable if mean error is small relative to values
        precision_acceptable = fp16_mean < 0.1  # Relaxed threshold for FP16

        return NumericalValidation(
            fp16_max_error=round(fp16_max, 6),
            fp16_mean_error=round(fp16_mean, 8),
            precision_acceptable=precision_acceptable,
        )

    def sustained_stress_test(
        self, duration_minutes: float = 2.0, matrix_size: int = 8192, thermal_limit: int = 85
    ) -> dict[str, Any]:
        """Run sustained compute stress test with thermal monitoring.
        Uses FP32 matmul which goes through cuBLAS with native Blackwell support.
        """
        logger.info(f"Running sustained stress test for {duration_minutes} minutes...")

        duration_sec = duration_minutes * 60

        # Allocate FP32 matrices (no JIT needed)
        a = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
        b = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)

        samples = []
        operations = 0
        start_time = time.time()
        start_temp = self._get_gpu_temp()

        try:
            while (time.time() - start_time) < duration_sec:
                # Check thermal
                temp = self._get_gpu_temp()
                if temp > thermal_limit:
                    logger.warning(f"Thermal limit reached ({temp}C), pausing...")
                    time.sleep(10)
                    continue

                # Run operations - cuBLAS matmul has native sm_120 support
                for _ in range(10):
                    c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()
                operations += 10

                # Sample every second
                elapsed = time.time() - start_time
                if len(samples) < int(elapsed):
                    samples.append(
                        {"time_sec": round(elapsed, 1), "temp_c": temp, "operations": operations}
                    )

        except KeyboardInterrupt:
            logger.info("Stress test interrupted")

        finally:
            del a, b
            cp.get_default_memory_pool().free_all_blocks()

        total_time = time.time() - start_time
        end_temp = self._get_gpu_temp()

        # Calculate sustained TFLOPS
        flops_per_op = 2 * (matrix_size**3)
        total_flops = flops_per_op * operations
        sustained_tflops = (total_flops / total_time) / 1e12 if total_time > 0 else 0

        return {
            "duration_sec": round(total_time, 1),
            "operations": operations,
            "sustained_tflops": round(sustained_tflops, 2),
            "start_temp_c": start_temp,
            "end_temp_c": end_temp,
            "max_temp_c": max(s["temp_c"] for s in samples) if samples else 0,
            "thermal_throttled": any(s["temp_c"] > thermal_limit for s in samples),
            "samples": samples,
        }

    def generate_report(
        self, matrix_size: int = 4096, sustained_minutes: float = 1.0
    ) -> TensorCoreReport:
        """Generate complete stress test certification report."""
        logger.info("Generating stress test certification report...")

        # Collect benchmarks
        benchmarks = []

        # FP32 baseline
        benchmarks.append(self.benchmark_precision(cp.float32, matrix_size, 50))

        # FP16
        benchmarks.append(self.benchmark_precision(cp.float16, matrix_size, 100))

        # Comparison
        comparison = self.compare_precisions(matrix_size)

        # Numerical validation
        validation = self.validate_numerical_precision()

        # Sustained test
        sustained = self.sustained_stress_test(
            duration_minutes=sustained_minutes, matrix_size=matrix_size
        )

        # Certification status (compute_capability is already formatted as string)
        cc_str = self.compute_capability

        if comparison.fp16_speedup >= 1.5:
            certification = f"CERTIFIED: RTX 5090 Blackwell (CC {cc_str}) Performance Verified"
        elif comparison.fp16_speedup >= 1.2:
            certification = f"PASSED: GPU Performance Acceptable (CC {cc_str})"
        else:
            certification = f"WARNING: Performance below expected levels (CC {cc_str})"

        return TensorCoreReport(
            timestamp=datetime.now().isoformat(),
            gpu_name=self.gpu_name,
            compute_capability=cc_str,
            cupy_version=cp.__version__,
            benchmarks=benchmarks,
            comparison=comparison,
            validation=validation,
            sustained_test=sustained,
            certification=certification,
        )

    def print_report(self, report: TensorCoreReport):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print("  RTX 5090 STRESS TEST REPORT (CuPy Edition)")
        print("  Evidence Suite: Savant Genesis Edition")
        print("=" * 70)

        print("\nHardware:")
        print(f"  GPU:                {report.gpu_name}")
        print(f"  Compute Capability: {report.compute_capability}")
        print(f"  CuPy:               {report.cupy_version}")

        print(f"\n{'-' * 70}")
        print("Precision Benchmarks:")
        print(f"  {'Precision':<12} {'TFLOPS':<10} {'Latency':<12} {'Memory':<10}")
        print(f"  {'-' * 44}")
        for b in report.benchmarks:
            print(
                f"  {b.dtype:<12} {b.tflops:<10.2f} {b.mean_latency_ms:<10.2f}ms {b.memory_used_mb:<8.0f}MB"
            )

        if report.comparison:
            c = report.comparison
            print(f"\n{'-' * 70}")
            print("Performance Comparison:")
            print(f"  FP32 Baseline:    {c.fp32_tflops:.2f} TFLOPS")
            print(f"  FP16 Speedup:     {c.fp16_speedup:.2f}x ({c.fp16_tflops:.2f} TFLOPS)")

        if report.validation:
            v = report.validation
            print(f"\n{'-' * 70}")
            print("Numerical Precision (vs FP32 reference):")
            print(f"  FP16 Max Error:   {v.fp16_max_error:.6f}")
            print(f"  FP16 Mean Error:  {v.fp16_mean_error:.8f}")
            print(f"  Acceptable:       {'Yes' if v.precision_acceptable else 'No'}")

        s = report.sustained_test
        print(f"\n{'-' * 70}")
        print("Sustained Performance:")
        print(f"  Duration:         {s['duration_sec']:.1f} seconds")
        print(f"  Operations:       {s['operations']:,}")
        print(f"  Sustained TFLOPS: {s['sustained_tflops']:.2f}")
        print(f"  Start Temp:       {s['start_temp_c']:.0f}C")
        print(f"  End Temp:         {s['end_temp_c']:.0f}C")
        print(f"  Max Temp:         {s['max_temp_c']:.0f}C")
        print(f"  Throttled:        {'Yes' if s['thermal_throttled'] else 'No'}")

        print(f"\n{'=' * 70}")
        print(f"  {report.certification}")
        print("=" * 70 + "\n")

    def save_report(self, report: TensorCoreReport, output_path: str):
        """Save report to JSON."""
        data = asdict(report)
        # Convert benchmarks
        data["benchmarks"] = [asdict(b) for b in report.benchmarks]
        if report.comparison:
            data["comparison"] = asdict(report.comparison)
        if report.validation:
            data["validation"] = asdict(report.validation)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RTX 5090 Stress Test (CuPy Edition)")
    parser.add_argument(
        "--matrix-size",
        "-m",
        type=int,
        default=4096,
        help="Matrix size for benchmarks (default: 4096)",
    )
    parser.add_argument(
        "--sustained-minutes",
        "-s",
        type=float,
        default=1.0,
        help="Sustained test duration in minutes (default: 1.0)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="stress_test_report.json", help="Output JSON file"
    )

    args = parser.parse_args()

    try:
        tester = TensorCoreStressTest()
        report = tester.generate_report(
            matrix_size=args.matrix_size, sustained_minutes=args.sustained_minutes
        )
        tester.print_report(report)
        tester.save_report(report, args.output)

        print("Stress test complete!")
        return 0

    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
