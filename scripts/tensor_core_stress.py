"""
Evidence Suite - RTX 5090 Tensor Core Stress Test
Isolates and validates BF16 precision gains on Blackwell architecture.

This test:
1. Verifies Tensor Core availability and BF16 support
2. Compares FP32 vs FP16 vs BF16 performance
3. Validates numerical precision across data types
4. Measures sustained Tensor Core throughput
5. Generates a hardware certification report
"""
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for Tensor Core stress test")
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
    bf16_tflops: float
    fp16_speedup: float
    bf16_speedup: float
    bf16_vs_fp16: float


@dataclass
class NumericalValidation:
    """Numerical precision validation results."""
    fp16_max_error: float
    bf16_max_error: float
    fp16_mean_error: float
    bf16_mean_error: float
    bf16_better_precision: bool


@dataclass
class TensorCoreReport:
    """Complete Tensor Core stress test report."""
    timestamp: str
    gpu_name: str
    cuda_version: str
    torch_version: str
    bf16_supported: bool
    tensor_cores_available: bool
    benchmarks: List[PrecisionBenchmark]
    comparison: Optional[PrecisionComparison]
    validation: Optional[NumericalValidation]
    sustained_test: Dict[str, Any]
    certification: str


class TensorCoreStressTest:
    """
    RTX 5090 Tensor Core Stress Test Suite.

    Validates BF16 performance gains specific to Blackwell architecture.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device = torch.device('cuda')
        self.gpu_name = torch.cuda.get_device_name(0)
        self.bf16_supported = torch.cuda.is_bf16_supported()
        self.tensor_cores = self._detect_tensor_cores()

        # NVML for thermal monitoring
        self._nvml_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        logger.info(f"GPU: {self.gpu_name}")
        logger.info(f"BF16 Supported: {self.bf16_supported}")
        logger.info(f"Tensor Cores: {self.tensor_cores}")

    def _detect_tensor_cores(self) -> bool:
        """Detect if GPU has Tensor Cores (Compute Capability >= 7.0)."""
        props = torch.cuda.get_device_properties(0)
        major, minor = props.major, props.minor
        # Tensor Cores available on Volta (7.0) and newer
        return major >= 7

    def _get_gpu_temp(self) -> float:
        """Get current GPU temperature."""
        if self._nvml_handle:
            try:
                return float(pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU
                ))
            except Exception:
                pass
        return 0.0

    def _get_memory_used_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def benchmark_precision(
        self,
        dtype: torch.dtype,
        matrix_size: int = 8192,
        iterations: int = 100
    ) -> PrecisionBenchmark:
        """
        Benchmark matrix multiplication for a specific precision.
        """
        dtype_name = str(dtype).replace("torch.", "")
        logger.info(f"Benchmarking {dtype_name} at {matrix_size}x{matrix_size}...")

        # Allocate matrices
        a = torch.randn(matrix_size, matrix_size, device=self.device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=self.device, dtype=dtype)

        memory_used = self._get_memory_used_mb()

        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        # Calculate TFLOPS
        # Matrix multiply: 2 * N^3 FLOPs
        flops_per_op = 2 * (matrix_size ** 3)
        total_flops = flops_per_op * iterations
        tflops = (total_flops / total_time) / 1e12

        # Cleanup
        del a, b
        torch.cuda.empty_cache()

        return PrecisionBenchmark(
            dtype=dtype_name,
            matrix_size=matrix_size,
            iterations=iterations,
            total_time_sec=round(total_time, 3),
            mean_latency_ms=round((total_time / iterations) * 1000, 3),
            tflops=round(tflops, 2),
            memory_used_mb=round(memory_used, 1)
        )

    def compare_precisions(self, matrix_size: int = 8192) -> PrecisionComparison:
        """
        Compare FP32, FP16, and BF16 performance.
        """
        logger.info("Running precision comparison...")

        # FP32 baseline
        fp32_result = self.benchmark_precision(torch.float32, matrix_size, iterations=50)

        # FP16
        fp16_result = self.benchmark_precision(torch.float16, matrix_size, iterations=100)

        # BF16 (if supported)
        if self.bf16_supported:
            bf16_result = self.benchmark_precision(torch.bfloat16, matrix_size, iterations=100)
            bf16_tflops = bf16_result.tflops
        else:
            bf16_tflops = 0.0
            logger.warning("BF16 not supported on this GPU")

        fp16_speedup = fp16_result.tflops / fp32_result.tflops if fp32_result.tflops > 0 else 0
        bf16_speedup = bf16_tflops / fp32_result.tflops if fp32_result.tflops > 0 else 0
        bf16_vs_fp16 = bf16_tflops / fp16_result.tflops if fp16_result.tflops > 0 else 0

        return PrecisionComparison(
            fp32_tflops=fp32_result.tflops,
            fp16_tflops=fp16_result.tflops,
            bf16_tflops=bf16_tflops,
            fp16_speedup=round(fp16_speedup, 2),
            bf16_speedup=round(bf16_speedup, 2),
            bf16_vs_fp16=round(bf16_vs_fp16, 2)
        )

    def validate_numerical_precision(self, matrix_size: int = 2048) -> NumericalValidation:
        """
        Validate numerical precision of FP16 vs BF16 against FP32 reference.

        BF16 has same exponent range as FP32 (better for large values)
        but less mantissa precision than FP16.
        """
        logger.info("Validating numerical precision...")

        # Create reference matrices in FP32
        a_fp32 = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float32)
        b_fp32 = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float32)

        # Reference result
        c_fp32 = torch.matmul(a_fp32, b_fp32)

        # FP16 computation
        a_fp16 = a_fp32.to(torch.float16)
        b_fp16 = b_fp32.to(torch.float16)
        c_fp16 = torch.matmul(a_fp16, b_fp16).to(torch.float32)

        # BF16 computation (if supported)
        if self.bf16_supported:
            a_bf16 = a_fp32.to(torch.bfloat16)
            b_bf16 = b_fp32.to(torch.bfloat16)
            c_bf16 = torch.matmul(a_bf16, b_bf16).to(torch.float32)
        else:
            c_bf16 = c_fp16  # Fallback

        # Calculate errors
        fp16_error = torch.abs(c_fp32 - c_fp16)
        bf16_error = torch.abs(c_fp32 - c_bf16)

        fp16_max = float(fp16_error.max())
        bf16_max = float(bf16_error.max())
        fp16_mean = float(fp16_error.mean())
        bf16_mean = float(bf16_error.mean())

        # Cleanup
        del a_fp32, b_fp32, c_fp32, a_fp16, b_fp16, c_fp16
        if self.bf16_supported:
            del a_bf16, b_bf16, c_bf16
        torch.cuda.empty_cache()

        return NumericalValidation(
            fp16_max_error=round(fp16_max, 6),
            bf16_max_error=round(bf16_max, 6),
            fp16_mean_error=round(fp16_mean, 8),
            bf16_mean_error=round(bf16_mean, 8),
            bf16_better_precision=bf16_mean < fp16_mean
        )

    def sustained_stress_test(
        self,
        duration_minutes: float = 2.0,
        matrix_size: int = 10240,
        thermal_limit: int = 85
    ) -> Dict[str, Any]:
        """
        Run sustained Tensor Core stress test with thermal monitoring.
        """
        logger.info(f"Running sustained stress test for {duration_minutes} minutes...")

        dtype = torch.bfloat16 if self.bf16_supported else torch.float16
        duration_sec = duration_minutes * 60

        # Allocate matrices
        a = torch.randn(matrix_size, matrix_size, device=self.device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=self.device, dtype=dtype)

        samples = []
        operations = 0
        start_time = time.time()
        start_temp = self._get_gpu_temp()

        try:
            while (time.time() - start_time) < duration_sec:
                # Check thermal
                temp = self._get_gpu_temp()
                if temp > thermal_limit:
                    logger.warning(f"Thermal limit reached ({temp}°C), pausing...")
                    time.sleep(10)
                    continue

                # Run operations
                for _ in range(10):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                operations += 10

                # Sample every second
                elapsed = time.time() - start_time
                if len(samples) < int(elapsed):
                    samples.append({
                        "time_sec": round(elapsed, 1),
                        "temp_c": temp,
                        "operations": operations
                    })

        except KeyboardInterrupt:
            logger.info("Stress test interrupted")

        finally:
            del a, b
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        end_temp = self._get_gpu_temp()

        # Calculate sustained TFLOPS
        flops_per_op = 2 * (matrix_size ** 3)
        total_flops = flops_per_op * operations
        sustained_tflops = (total_flops / total_time) / 1e12

        return {
            "duration_sec": round(total_time, 1),
            "operations": operations,
            "sustained_tflops": round(sustained_tflops, 2),
            "start_temp_c": start_temp,
            "end_temp_c": end_temp,
            "max_temp_c": max(s["temp_c"] for s in samples) if samples else 0,
            "thermal_throttled": any(s["temp_c"] > thermal_limit for s in samples),
            "samples": samples
        }

    def generate_report(
        self,
        matrix_size: int = 8192,
        sustained_minutes: float = 1.0
    ) -> TensorCoreReport:
        """
        Generate complete Tensor Core certification report.
        """
        logger.info("Generating Tensor Core certification report...")

        # Collect benchmarks
        benchmarks = []

        # FP32 baseline
        benchmarks.append(self.benchmark_precision(torch.float32, matrix_size, 50))

        # FP16
        benchmarks.append(self.benchmark_precision(torch.float16, matrix_size, 100))

        # BF16
        if self.bf16_supported:
            benchmarks.append(self.benchmark_precision(torch.bfloat16, matrix_size, 100))

        # Comparison
        comparison = self.compare_precisions(matrix_size)

        # Numerical validation
        validation = self.validate_numerical_precision()

        # Sustained test
        sustained = self.sustained_stress_test(duration_minutes=sustained_minutes)

        # Certification status
        if self.bf16_supported and comparison.bf16_speedup >= 1.8:
            certification = "CERTIFIED: Blackwell Tensor Core Performance Verified"
        elif comparison.fp16_speedup >= 1.5:
            certification = "CERTIFIED: Tensor Core Performance Verified (FP16)"
        else:
            certification = "WARNING: Tensor Core performance below expected levels"

        return TensorCoreReport(
            timestamp=datetime.now().isoformat(),
            gpu_name=self.gpu_name,
            cuda_version=torch.version.cuda or "N/A",
            torch_version=torch.__version__,
            bf16_supported=self.bf16_supported,
            tensor_cores_available=self.tensor_cores,
            benchmarks=benchmarks,
            comparison=comparison,
            validation=validation,
            sustained_test=sustained,
            certification=certification
        )

    def print_report(self, report: TensorCoreReport):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print("  RTX 5090 TENSOR CORE STRESS TEST REPORT")
        print("  Evidence Suite: Savant Genesis Edition")
        print("=" * 70)

        print(f"\nHardware:")
        print(f"  GPU:           {report.gpu_name}")
        print(f"  CUDA:          {report.cuda_version}")
        print(f"  PyTorch:       {report.torch_version}")
        print(f"  Tensor Cores:  {'Yes' if report.tensor_cores_available else 'No'}")
        print(f"  BF16 Support:  {'Yes' if report.bf16_supported else 'No'}")

        print(f"\n{'─' * 70}")
        print("Precision Benchmarks:")
        print(f"  {'Precision':<12} {'TFLOPS':<10} {'Latency':<12} {'Memory':<10}")
        print(f"  {'─' * 44}")
        for b in report.benchmarks:
            print(f"  {b.dtype:<12} {b.tflops:<10.2f} {b.mean_latency_ms:<10.2f}ms {b.memory_used_mb:<8.0f}MB")

        if report.comparison:
            c = report.comparison
            print(f"\n{'─' * 70}")
            print("Performance Comparison:")
            print(f"  FP32 Baseline:    {c.fp32_tflops:.2f} TFLOPS")
            print(f"  FP16 Speedup:     {c.fp16_speedup:.2f}x ({c.fp16_tflops:.2f} TFLOPS)")
            print(f"  BF16 Speedup:     {c.bf16_speedup:.2f}x ({c.bf16_tflops:.2f} TFLOPS)")
            print(f"  BF16 vs FP16:     {c.bf16_vs_fp16:.2f}x")

        if report.validation:
            v = report.validation
            print(f"\n{'─' * 70}")
            print("Numerical Precision (vs FP32 reference):")
            print(f"  FP16 Max Error:   {v.fp16_max_error:.6f}")
            print(f"  BF16 Max Error:   {v.bf16_max_error:.6f}")
            print(f"  FP16 Mean Error:  {v.fp16_mean_error:.8f}")
            print(f"  BF16 Mean Error:  {v.bf16_mean_error:.8f}")
            print(f"  BF16 Better:      {'Yes' if v.bf16_better_precision else 'No'}")

        s = report.sustained_test
        print(f"\n{'─' * 70}")
        print("Sustained Performance:")
        print(f"  Duration:         {s['duration_sec']:.1f} seconds")
        print(f"  Operations:       {s['operations']:,}")
        print(f"  Sustained TFLOPS: {s['sustained_tflops']:.2f}")
        print(f"  Start Temp:       {s['start_temp_c']:.0f}°C")
        print(f"  End Temp:         {s['end_temp_c']:.0f}°C")
        print(f"  Max Temp:         {s['max_temp_c']:.0f}°C")
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

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RTX 5090 Tensor Core Stress Test"
    )
    parser.add_argument(
        "--matrix-size", "-m",
        type=int,
        default=8192,
        help="Matrix size for benchmarks (default: 8192)"
    )
    parser.add_argument(
        "--sustained-minutes", "-s",
        type=float,
        default=1.0,
        help="Sustained test duration in minutes (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tensor_core_report.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    try:
        tester = TensorCoreStressTest()
        report = tester.generate_report(
            matrix_size=args.matrix_size,
            sustained_minutes=args.sustained_minutes
        )
        tester.print_report(report)
        tester.save_report(report, args.output)

        print("Tensor Core stress test complete!")
        return 0

    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
