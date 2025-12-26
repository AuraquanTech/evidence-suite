"""
Evidence Suite - Comprehensive Benchmark System
Collects detailed performance metrics and generates benchmark reports.
"""
import asyncio
import os
import sys
import time
import json
import statistics
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkMetric:
    """Single benchmark metric."""
    name: str
    value: float
    unit: str
    category: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run."""
    name: str
    iterations: int
    total_time_s: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_sec: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    success_rate: float
    errors: List[str] = field(default_factory=list)


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    platform: str
    python_version: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_version: Optional[str]


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    suite_name: str
    version: str
    timestamp: str
    system_info: SystemInfo
    benchmarks: List[BenchmarkRun]
    metrics: Dict[str, float]
    comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "version": self.version,
            "timestamp": self.timestamp,
            "system_info": asdict(self.system_info),
            "benchmarks": [asdict(b) for b in self.benchmarks],
            "metrics": self.metrics,
            "comparison": self.comparison,
        }

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK REPORT: {self.suite_name}")
        print(f"{'='*70}")
        print(f"Version: {self.version}")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nSystem Info:")
        print(f"  Platform: {self.system_info.platform}")
        print(f"  Python: {self.system_info.python_version}")
        print(f"  CPU: {self.system_info.cpu_model}")
        print(f"  Cores/Threads: {self.system_info.cpu_cores}/{self.system_info.cpu_threads}")
        print(f"  RAM: {self.system_info.ram_total_gb:.1f}GB total, {self.system_info.ram_available_gb:.1f}GB available")
        if self.system_info.gpu_name:
            print(f"  GPU: {self.system_info.gpu_name} ({self.system_info.gpu_memory_gb:.1f}GB)")

        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")

        for bench in self.benchmarks:
            print(f"\n{bench.name}:")
            print(f"  Iterations: {bench.iterations}")
            print(f"  Total Time: {bench.total_time_s:.2f}s")
            print(f"  Avg: {bench.avg_time_ms:.2f}ms | Min: {bench.min_time_ms:.2f}ms | Max: {bench.max_time_ms:.2f}ms")
            print(f"  P50: {bench.p50_ms:.2f}ms | P95: {bench.p95_ms:.2f}ms | P99: {bench.p99_ms:.2f}ms")
            print(f"  Throughput: {bench.ops_per_sec:.2f} ops/sec")
            print(f"  Memory Delta: {bench.memory_delta_mb:+.2f}MB")
            print(f"  Success Rate: {bench.success_rate*100:.1f}%")

        print(f"\n{'='*70}")
        print("AGGREGATE METRICS")
        print(f"{'='*70}")
        for key, value in self.metrics.items():
            print(f"  {key}: {value}")

        if self.comparison:
            print(f"\n{'='*70}")
            print("COMPARISON WITH BASELINE")
            print(f"{'='*70}")
            for key, value in self.comparison.items():
                print(f"  {key}: {value}")


class BenchmarkRunner:
    """Runs benchmarks and collects metrics."""

    def __init__(self, iterations: int = 100, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.benchmarks: List[BenchmarkRun] = []

    def get_system_info(self) -> SystemInfo:
        """Collect system information."""
        gpu_name = None
        gpu_memory = None
        cuda_version = None

        # Try to get GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem_info.total / (1024**3)
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            pynvml.nvmlShutdown()
        except Exception:
            pass

        # Get CPU model
        cpu_model = platform.processor()
        if not cpu_model:
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    cpu_model = lines[1].strip()
            except Exception:
                cpu_model = "Unknown"

        return SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu_model=cpu_model,
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            cpu_threads=psutil.cpu_count(logical=True) or 1,
            ram_total_gb=psutil.virtual_memory().total / (1024**3),
            ram_available_gb=psutil.virtual_memory().available / (1024**3),
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory,
            cuda_version=cuda_version,
        )

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    async def benchmark_pipeline(self) -> BenchmarkRun:
        """Benchmark the analysis pipeline."""
        print("  Benchmarking pipeline...")

        try:
            from pipeline import EvidencePipeline
            pipeline = EvidencePipeline()
            await pipeline.initialize()
        except Exception as e:
            return BenchmarkRun(
                name="pipeline",
                iterations=0,
                total_time_s=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                ops_per_sec=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_delta_mb=0,
                success_rate=0,
                errors=[str(e)]
            )

        test_texts = [
            "You're the one who caused all of this, not me! I'm the victim here!",
            "That never happened, you're imagining things. Your memory is wrong.",
            "If you really loved me, you would do this for me.",
            "I appreciate your help with this matter. Thank you for your time.",
            "Stop blaming me! You always attack me when you're the problem!",
        ]

        # Warmup
        for _ in range(self.warmup):
            await pipeline.process_text(test_texts[0], "warmup")

        memory_before = self._get_memory_mb()
        times = []
        errors = []
        successes = 0

        start_total = time.time()
        for i in range(self.iterations):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            try:
                result = await pipeline.process_text(text, f"bench_{i}")
                if result.status == "success":
                    successes += 1
            except Exception as e:
                errors.append(str(e))
            times.append((time.perf_counter() - start) * 1000)

        total_time = time.time() - start_total
        memory_after = self._get_memory_mb()

        times_sorted = sorted(times)
        return BenchmarkRun(
            name="pipeline",
            iterations=self.iterations,
            total_time_s=total_time,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            p50_ms=times_sorted[len(times_sorted) // 2],
            p95_ms=times_sorted[int(len(times_sorted) * 0.95)],
            p99_ms=times_sorted[int(len(times_sorted) * 0.99)],
            ops_per_sec=self.iterations / total_time if total_time > 0 else 0,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_after - memory_before,
            success_rate=successes / self.iterations,
            errors=errors[:5],
        )

    async def benchmark_behavioral_agent(self) -> BenchmarkRun:
        """Benchmark behavioral analysis agent."""
        print("  Benchmarking behavioral agent...")

        try:
            from agents.behavioral_agent import BehavioralAgent
            agent = BehavioralAgent()
            await agent.initialize()
        except Exception as e:
            return BenchmarkRun(
                name="behavioral_agent",
                iterations=0,
                total_time_s=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                ops_per_sec=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_delta_mb=0,
                success_rate=0,
                errors=[str(e)]
            )

        test_text = "You're always wrong and everything is your fault! I'm the victim here!"

        # Warmup
        for _ in range(self.warmup):
            await agent.analyze(test_text)

        memory_before = self._get_memory_mb()
        times = []
        errors = []
        successes = 0

        start_total = time.time()
        for i in range(self.iterations):
            start = time.perf_counter()
            try:
                result = await agent.analyze(test_text)
                if result.success:
                    successes += 1
            except Exception as e:
                errors.append(str(e))
            times.append((time.perf_counter() - start) * 1000)

        total_time = time.time() - start_total
        memory_after = self._get_memory_mb()

        times_sorted = sorted(times)
        return BenchmarkRun(
            name="behavioral_agent",
            iterations=self.iterations,
            total_time_s=total_time,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            p50_ms=times_sorted[len(times_sorted) // 2],
            p95_ms=times_sorted[int(len(times_sorted) * 0.95)],
            p99_ms=times_sorted[int(len(times_sorted) * 0.99)],
            ops_per_sec=self.iterations / total_time if total_time > 0 else 0,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_after - memory_before,
            success_rate=successes / self.iterations,
            errors=errors[:5],
        )

    async def benchmark_fusion_agent(self) -> BenchmarkRun:
        """Benchmark fusion agent."""
        print("  Benchmarking fusion agent...")

        try:
            from agents.fusion_agent import FusionAgent
            agent = FusionAgent()
            await agent.initialize()
        except Exception as e:
            return BenchmarkRun(
                name="fusion_agent",
                iterations=0,
                total_time_s=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                ops_per_sec=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_delta_mb=0,
                success_rate=0,
                errors=[str(e)]
            )

        # Create mock input
        test_input = {
            "behavioral": {
                "sentiment": {"compound": -0.5},
                "darvo_score": 0.8,
                "gaslighting_score": 0.6,
            },
            "ocr": {"confidence": 0.95},
        }

        # Warmup
        for _ in range(self.warmup):
            await agent.fuse(test_input)

        memory_before = self._get_memory_mb()
        times = []
        errors = []
        successes = 0

        start_total = time.time()
        for i in range(self.iterations):
            start = time.perf_counter()
            try:
                result = await agent.fuse(test_input)
                if result.success:
                    successes += 1
            except Exception as e:
                errors.append(str(e))
            times.append((time.perf_counter() - start) * 1000)

        total_time = time.time() - start_total
        memory_after = self._get_memory_mb()

        times_sorted = sorted(times)
        return BenchmarkRun(
            name="fusion_agent",
            iterations=self.iterations,
            total_time_s=total_time,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            p50_ms=times_sorted[len(times_sorted) // 2],
            p95_ms=times_sorted[int(len(times_sorted) * 0.95)],
            p99_ms=times_sorted[int(len(times_sorted) * 0.99)],
            ops_per_sec=self.iterations / total_time if total_time > 0 else 0,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_after - memory_before,
            success_rate=successes / self.iterations,
            errors=errors[:5],
        )

    def compare_with_baseline(
        self,
        current: Dict[str, float],
        baseline_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Compare current metrics with baseline."""
        if not baseline_path or not Path(baseline_path).exists():
            return None

        with open(baseline_path) as f:
            baseline = json.load(f)

        baseline_metrics = baseline.get("metrics", {})
        comparison = {}

        for key in current:
            if key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                current_val = current[key]
                if baseline_val > 0:
                    change_pct = ((current_val - baseline_val) / baseline_val) * 100
                    comparison[key] = {
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": round(change_pct, 2),
                        "improved": change_pct < 0 if "time" in key.lower() else change_pct > 0,
                    }

        return comparison if comparison else None

    async def run_all_benchmarks(
        self,
        baseline_path: Optional[str] = None
    ) -> BenchmarkReport:
        """Run all benchmarks and generate report."""
        print(f"\n{'='*70}")
        print("EVIDENCE SUITE BENCHMARK")
        print(f"{'='*70}")
        print(f"Iterations: {self.iterations}")
        print(f"Warmup: {self.warmup}")
        print(f"{'='*70}\n")

        system_info = self.get_system_info()
        print("Collecting system info...")
        print(f"  CPU: {system_info.cpu_model}")
        if system_info.gpu_name:
            print(f"  GPU: {system_info.gpu_name}")

        print("\nRunning benchmarks...")

        # Run benchmarks
        pipeline_bench = await self.benchmark_pipeline()
        self.benchmarks.append(pipeline_bench)

        behavioral_bench = await self.benchmark_behavioral_agent()
        self.benchmarks.append(behavioral_bench)

        fusion_bench = await self.benchmark_fusion_agent()
        self.benchmarks.append(fusion_bench)

        # Calculate aggregate metrics
        all_times = []
        for bench in self.benchmarks:
            if bench.avg_time_ms > 0:
                all_times.append(bench.avg_time_ms)

        metrics = {
            "total_benchmarks": len(self.benchmarks),
            "avg_latency_ms": round(statistics.mean(all_times), 2) if all_times else 0,
            "total_ops": sum(b.iterations for b in self.benchmarks),
            "avg_throughput_ops_sec": round(statistics.mean([b.ops_per_sec for b in self.benchmarks if b.ops_per_sec > 0]), 2),
            "total_memory_delta_mb": round(sum(b.memory_delta_mb for b in self.benchmarks), 2),
            "avg_success_rate": round(statistics.mean([b.success_rate for b in self.benchmarks]) * 100, 2),
        }

        # Compare with baseline
        comparison = self.compare_with_baseline(metrics, baseline_path)

        # Get version
        try:
            from core.config import api_settings
            version = api_settings.version
        except Exception:
            version = "unknown"

        report = BenchmarkReport(
            suite_name="Evidence Suite",
            version=version,
            timestamp=datetime.utcnow().isoformat(),
            system_info=system_info,
            benchmarks=self.benchmarks,
            metrics=metrics,
            comparison=comparison,
        )

        return report


def run_benchmarks(
    iterations: int = 100,
    warmup: int = 5,
    output_file: Optional[str] = None,
    baseline_file: Optional[str] = None,
) -> Tuple[bool, BenchmarkReport]:
    """Run benchmarks and optionally save report."""
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    report = asyncio.run(runner.run_all_benchmarks(baseline_path=baseline_file))

    report.print_summary()

    if output_file:
        report.save(output_file)
        print(f"\nReport saved to: {output_file}")

    # Check if any benchmark had 0% success rate
    all_passed = all(b.success_rate > 0 for b in report.benchmarks)
    return all_passed, report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evidence Suite Benchmarks")
    parser.add_argument("-i", "--iterations", type=int, default=100,
                        help="Number of iterations per benchmark")
    parser.add_argument("-w", "--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file for JSON report")
    parser.add_argument("-b", "--baseline", type=str, default=None,
                        help="Baseline file for comparison")

    args = parser.parse_args()

    success, _ = run_benchmarks(
        iterations=args.iterations,
        warmup=args.warmup,
        output_file=args.output,
        baseline_file=args.baseline,
    )

    sys.exit(0 if success else 1)
