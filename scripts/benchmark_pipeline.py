"""
Evidence Suite - Pipeline Benchmark (CuPy Edition)
Measures RTX 5090 Blackwell performance across OCR and AI inference layers.
Uses CuPy for GPU acceleration (PyTorch lacks sm_120 support).
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add PyTorch's CUDA DLLs for CuPy to use
torch_lib = r"C:\Users\ayrto\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from loguru import logger
from core.hardware_monitor import HardwareMonitor, get_monitor

# Try to import visualizations (optional)
try:
    from core.visualizations import PipelinePerformanceVisualizer
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


class PipelineBenchmark:
    """
    Comprehensive benchmark suite for Evidence Suite pipeline.
    Uses CuPy for RTX 5090 Blackwell (sm_120) GPU acceleration.
    """

    def __init__(self):
        self.monitor = get_monitor()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "benchmarks": {},
            "summary": {}
        }
        self.gpu_available = CUPY_AVAILABLE

    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            "python_version": sys.version,
            "cupy_available": CUPY_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE
        }

        if CUPY_AVAILABLE:
            info["cupy_version"] = cp.__version__
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)

            info["gpu_name"] = props['name'].decode()
            info["gpu_memory_gb"] = props['totalGlobalMem'] / (1024**3)

            # Format compute capability
            cc_raw = device.compute_capability
            if isinstance(cc_raw, str):
                cc_int = int(cc_raw)
                info["compute_capability"] = f"{cc_int // 10}.{cc_int % 10}" if cc_int >= 100 else f"{cc_int}.0"
            else:
                info["compute_capability"] = str(cc_raw)

        gpu_status = self.monitor.get_gpu_status()
        if gpu_status:
            info["gpu_temp_c"] = gpu_status.temperature_c
            info["vram_used_mb"] = gpu_status.vram_used_mb
            info["vram_total_mb"] = gpu_status.vram_total_mb

        self.results["system"] = info
        return info

    def benchmark_inference_latency(
        self,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark simulated inference latency using GPU matrix operations.
        """
        logger.info(f"Running inference latency benchmark ({iterations} iterations)...")

        latencies = []

        # Warmup
        for _ in range(warmup):
            start = time.perf_counter()
            self._simulate_inference()
            _ = time.perf_counter() - start

        # Benchmark
        for i in range(iterations):
            # Check thermal
            throttle = self.monitor.should_throttle()
            if throttle["throttle"]:
                logger.warning(f"Throttling: {throttle['reason']}")
                time.sleep(throttle.get("recommended_delay", 1))

            start = time.perf_counter()
            self._simulate_inference()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{iterations}")

        # Calculate statistics
        if NUMPY_AVAILABLE:
            latencies_arr = np.array(latencies)
            stats = {
                "mean_ms": float(np.mean(latencies_arr)),
                "std_ms": float(np.std(latencies_arr)),
                "min_ms": float(np.min(latencies_arr)),
                "max_ms": float(np.max(latencies_arr)),
                "p50_ms": float(np.percentile(latencies_arr, 50)),
                "p95_ms": float(np.percentile(latencies_arr, 95)),
                "p99_ms": float(np.percentile(latencies_arr, 99)),
                "throughput_per_sec": iterations / (sum(latencies) / 1000)
            }
        else:
            sorted_lat = sorted(latencies)
            stats = {
                "mean_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": sorted_lat[len(sorted_lat) // 2],
                "p95_ms": sorted_lat[int(len(sorted_lat) * 0.95)],
                "p99_ms": sorted_lat[int(len(sorted_lat) * 0.99)],
                "throughput_per_sec": iterations / (sum(latencies) / 1000)
            }

        stats["raw_latencies"] = latencies
        self.results["benchmarks"]["inference"] = stats
        return stats

    def benchmark_matmul_performance(
        self,
        sizes: List[int] = [1024, 2048, 4096, 8192],
        iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark matrix multiplication (simulates BERT attention).
        Uses CuPy with cuBLAS for native Blackwell support.
        """
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available, skipping matmul benchmark")
            return {}

        logger.info("Running matrix multiplication benchmark...")

        results = {}

        for size in sizes:
            logger.info(f"  Matrix size: {size}x{size}")

            try:
                # Use FP32 for reliable benchmarking
                a = cp.random.randn(size, size, dtype=cp.float32)
                b = cp.random.randn(size, size, dtype=cp.float32)

                # Warmup
                for _ in range(5):
                    c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()

                # Benchmark
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

                # TFLOPS calculation: matmul is 2*N^3 FLOPs
                flops = 2 * (size ** 3)
                mean_time_sec = (sum(times) / len(times)) / 1000
                tflops = (flops / mean_time_sec) / 1e12

                results[f"{size}x{size}"] = {
                    "mean_ms": round(sum(times) / len(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "tflops": round(tflops, 2)
                }

                # Cleanup
                del a, b, c
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                logger.error(f"  Failed at size {size}: {e}")
                results[f"{size}x{size}"] = {"error": str(e)}

        self.results["benchmarks"]["matmul"] = results
        return results

    def benchmark_memory_bandwidth(self, size_mb: int = 1024) -> Dict[str, Any]:
        """
        Benchmark GPU memory bandwidth using CuPy.
        """
        if not CUPY_AVAILABLE:
            return {}

        logger.info(f"Running memory bandwidth benchmark ({size_mb}MB)...")

        elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes

        try:
            # Allocate
            a = cp.random.randn(elements, dtype=cp.float32)
            b = cp.empty(elements, dtype=cp.float32)

            # Warmup
            for _ in range(5):
                cp.copyto(b, a)
            cp.cuda.Stream.null.synchronize()

            # Benchmark
            iterations = 20
            start = time.perf_counter()
            for _ in range(iterations):
                cp.copyto(b, a)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start

            # Bandwidth = (read + write) * iterations / time
            bytes_transferred = 2 * size_mb * 1024 * 1024 * iterations
            bandwidth_gbps = (bytes_transferred / elapsed) / (1024**3)

            result = {
                "size_mb": size_mb,
                "iterations": iterations,
                "total_time_sec": round(elapsed, 3),
                "bandwidth_gbps": round(bandwidth_gbps, 2)
            }

            del a, b
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            result = {"error": str(e)}

        self.results["benchmarks"]["memory_bandwidth"] = result
        return result

    def _simulate_inference(self):
        """Simulate inference workload using CuPy or NumPy."""
        if CUPY_AVAILABLE:
            # Real GPU work using CuPy (cuBLAS has native sm_120 support)
            size = 512
            a = cp.random.randn(size, size, dtype=cp.float32)
            b = cp.random.randn(size, size, dtype=cp.float32)
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            del a, b, c
        elif NUMPY_AVAILABLE:
            # CPU simulation
            a = np.random.randn(256, 256)
            b = np.random.randn(256, 256)
            _ = np.dot(a, b)
        else:
            time.sleep(0.005)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "timestamp": self.results["timestamp"],
            "gpu": self.results["system"].get("gpu_name", "Unknown"),
            "compute_capability": self.results["system"].get("compute_capability", "Unknown")
        }

        if "inference" in self.results["benchmarks"]:
            inf = self.results["benchmarks"]["inference"]
            summary["inference_mean_ms"] = round(inf.get("mean_ms", 0), 2)
            summary["inference_p95_ms"] = round(inf.get("p95_ms", 0), 2)
            summary["throughput_per_sec"] = round(inf.get("throughput_per_sec", 0), 1)

        if "matmul" in self.results["benchmarks"]:
            mm = self.results["benchmarks"]["matmul"]
            # Get largest successful size
            for size in ["8192x8192", "4096x4096", "2048x2048", "1024x1024"]:
                if size in mm and "tflops" in mm[size]:
                    summary["peak_tflops"] = mm[size]["tflops"]
                    summary["peak_size"] = size
                    break

        if "memory_bandwidth" in self.results["benchmarks"]:
            mb = self.results["benchmarks"]["memory_bandwidth"]
            summary["memory_bandwidth_gbps"] = mb.get("bandwidth_gbps", 0)

        self.results["summary"] = summary
        return summary

    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save results to JSON."""
        # Remove raw latencies for cleaner output
        results_copy = self.results.copy()
        results_copy["benchmarks"] = self.results["benchmarks"].copy()
        if "inference" in results_copy["benchmarks"]:
            results_copy["benchmarks"]["inference"] = {
                k: v for k, v in results_copy["benchmarks"]["inference"].items()
                if k != "raw_latencies"
            }

        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def print_results(self):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("  RTX 5090 MOBILE PIPELINE BENCHMARK RESULTS")
        print("  (CuPy Edition - Native Blackwell Support)")
        print("=" * 60)

        # System info
        sys_info = self.results.get("system", {})
        print(f"\nSystem:")
        print(f"  GPU:      {sys_info.get('gpu_name', 'N/A')}")
        print(f"  CC:       {sys_info.get('compute_capability', 'N/A')}")
        print(f"  VRAM:     {sys_info.get('gpu_memory_gb', 0):.1f} GB")
        print(f"  CuPy:     {sys_info.get('cupy_version', 'N/A')}")

        # Inference
        if "inference" in self.results.get("benchmarks", {}):
            inf = self.results["benchmarks"]["inference"]
            print(f"\nInference Latency:")
            print(f"  Mean:       {inf.get('mean_ms', 0):.2f} ms")
            print(f"  P95:        {inf.get('p95_ms', 0):.2f} ms")
            print(f"  P99:        {inf.get('p99_ms', 0):.2f} ms")
            print(f"  Throughput: {inf.get('throughput_per_sec', 0):.1f} items/sec")

        # Matmul
        if "matmul" in self.results.get("benchmarks", {}):
            print(f"\nMatrix Multiply (TFLOPS):")
            for size, data in self.results["benchmarks"]["matmul"].items():
                if "tflops" in data:
                    print(f"  {size}: {data['tflops']:.2f} TFLOPS ({data['mean_ms']:.2f}ms)")
                elif "error" in data:
                    print(f"  {size}: FAILED")

        # Memory
        if "memory_bandwidth" in self.results.get("benchmarks", {}):
            mb = self.results["benchmarks"]["memory_bandwidth"]
            if "bandwidth_gbps" in mb:
                print(f"\nMemory Bandwidth:")
                print(f"  {mb.get('bandwidth_gbps', 0):.1f} GB/s")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evidence Suite Pipeline Benchmark (CuPy Edition)")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                       help="Number of inference iterations (default: 100)")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.json",
                       help="Output JSON file")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Generate visualization charts")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EVIDENCE SUITE PIPELINE BENCHMARK")
    print("  RTX 5090 Mobile Performance Audit (CuPy Edition)")
    print("=" * 60 + "\n")

    if not CUPY_AVAILABLE:
        print("ERROR: CuPy is required for GPU benchmarks.")
        print("Install with: pip install cupy-cuda12x")
        return 1

    benchmark = PipelineBenchmark()

    # Collect system info
    logger.info("Collecting system information...")
    benchmark.collect_system_info()

    # Run benchmarks
    benchmark.benchmark_inference_latency(iterations=args.iterations)
    benchmark.benchmark_matmul_performance()
    benchmark.benchmark_memory_bandwidth()

    # Generate summary
    benchmark.generate_summary()

    # Print and save
    benchmark.print_results()
    benchmark.save_results(args.output)

    # Visualize
    if args.visualize and VIZ_AVAILABLE:
        logger.info("Generating visualizations...")
        viz = PipelinePerformanceVisualizer()

        if "inference" in benchmark.results.get("benchmarks", {}):
            latencies = benchmark.results["benchmarks"]["inference"].get("raw_latencies", [])
            if latencies:
                path = viz.plot_latency_distribution(latencies)
                if path:
                    print(f"Latency chart: {path}")

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
