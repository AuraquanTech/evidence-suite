"""
Evidence Suite - Comprehensive Stress Testing Framework
Stress tests for pipeline, API, and agents with metrics collection.
"""
import asyncio
import os
import sys
import time
import random
import string
import statistics
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StressTestResult:
    """Result from a single stress test run."""
    test_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestReport:
    """Aggregated stress test report."""
    suite_name: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    total_duration_s: float
    results: List[StressTestResult]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / self.total_tests if self.total_tests > 0 else 0,
            "total_duration_s": round(self.total_duration_s, 2),
            "metrics": self.metrics,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class StressTestRunner:
    """Runs stress tests and collects metrics."""

    def __init__(self, concurrency: int = 10, iterations: int = 100):
        self.concurrency = concurrency
        self.iterations = iterations
        self.results: List[StressTestResult] = []

    def _generate_test_text(self, pattern: str = "mixed") -> str:
        """Generate test text for behavioral analysis."""
        patterns = {
            "normal": [
                "I appreciate your help with this matter.",
                "Thank you for your time and consideration.",
                "I understand your perspective on this issue.",
                "Let's work together to find a solution.",
            ],
            "darvo": [
                "You're the one who caused all of this, not me!",
                "I'm the real victim here, you always blame me!",
                "How dare you accuse me when you're the problem!",
                "Stop playing the victim, I'm the one suffering!",
            ],
            "gaslighting": [
                "That never happened, you're imagining things.",
                "You're crazy if you think I said that.",
                "Everyone knows you're too sensitive.",
                "Your memory is completely wrong about this.",
            ],
            "manipulation": [
                "If you really loved me, you would do this.",
                "After everything I've done for you...",
                "Fine, I guess I'm just not good enough for you.",
                "You'll regret this decision forever.",
            ],
        }

        if pattern == "mixed":
            all_texts = []
            for texts in patterns.values():
                all_texts.extend(texts)
            return random.choice(all_texts)
        return random.choice(patterns.get(pattern, patterns["normal"]))

    async def stress_test_pipeline(self) -> List[StressTestResult]:
        """Stress test the analysis pipeline."""
        results = []

        try:
            from pipeline import EvidencePipeline
            pipeline = EvidencePipeline()
            await pipeline.initialize()
        except Exception as e:
            return [StressTestResult(
                test_name="pipeline_init",
                success=False,
                duration_ms=0,
                error=str(e)
            )]

        async def run_single_analysis(idx: int) -> StressTestResult:
            start = time.perf_counter()
            try:
                text = self._generate_test_text("mixed")
                result = await pipeline.process_text(text, f"stress_test_{idx}")
                duration = (time.perf_counter() - start) * 1000

                return StressTestResult(
                    test_name=f"pipeline_analysis_{idx}",
                    success=result.status == "success",
                    duration_ms=duration,
                    metadata={
                        "text_length": len(text),
                        "fused_score": result.fused_score,
                        "classification": result.fused_classification,
                    }
                )
            except Exception as e:
                return StressTestResult(
                    test_name=f"pipeline_analysis_{idx}",
                    success=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error=str(e)
                )

        # Run concurrent tests
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_analysis(idx: int) -> StressTestResult:
            async with semaphore:
                return await run_single_analysis(idx)

        tasks = [bounded_analysis(i) for i in range(self.iterations)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def stress_test_agents(self) -> List[StressTestResult]:
        """Stress test individual agents."""
        results = []

        # Test behavioral agent
        try:
            from agents.behavioral_agent import BehavioralAgent
            agent = BehavioralAgent()
            await agent.initialize()

            for i in range(min(self.iterations, 50)):  # Limit agent tests
                start = time.perf_counter()
                try:
                    text = self._generate_test_text("mixed")
                    result = await agent.analyze(text)
                    duration = (time.perf_counter() - start) * 1000

                    results.append(StressTestResult(
                        test_name=f"behavioral_agent_{i}",
                        success=result.success,
                        duration_ms=duration,
                        metadata={
                            "darvo_score": result.data.get("darvo_score"),
                            "gaslighting_score": result.data.get("gaslighting_score"),
                        }
                    ))
                except Exception as e:
                    results.append(StressTestResult(
                        test_name=f"behavioral_agent_{i}",
                        success=False,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        error=str(e)
                    ))
        except Exception as e:
            results.append(StressTestResult(
                test_name="behavioral_agent_init",
                success=False,
                duration_ms=0,
                error=str(e)
            ))

        return results

    def stress_test_cache(self) -> List[StressTestResult]:
        """Stress test Redis cache operations."""
        results = []

        try:
            import asyncio
            from core.cache import CacheManager

            async def run_cache_tests():
                cache = CacheManager()
                await cache.connect()

                if not cache.is_connected:
                    return [StressTestResult(
                        test_name="cache_connection",
                        success=False,
                        duration_ms=0,
                        error="Redis not available"
                    )]

                cache_results = []

                # Test set/get operations
                for i in range(min(self.iterations, 100)):
                    start = time.perf_counter()
                    try:
                        key = f"stress_test_{i}"
                        value = {"test": i, "data": "x" * 1000}

                        await cache.set_analysis(key, value, ttl=60)
                        retrieved = await cache.get_analysis(key)
                        duration = (time.perf_counter() - start) * 1000

                        cache_results.append(StressTestResult(
                            test_name=f"cache_set_get_{i}",
                            success=retrieved is not None,
                            duration_ms=duration,
                        ))
                    except Exception as e:
                        cache_results.append(StressTestResult(
                            test_name=f"cache_set_get_{i}",
                            success=False,
                            duration_ms=(time.perf_counter() - start) * 1000,
                            error=str(e)
                        ))

                await cache.close()
                return cache_results

            return asyncio.run(run_cache_tests())

        except Exception as e:
            return [StressTestResult(
                test_name="cache_init",
                success=False,
                duration_ms=0,
                error=str(e)
            )]

    def run_all_stress_tests(self) -> StressTestReport:
        """Run all stress tests and generate report."""
        print(f"\n{'='*60}")
        print("EVIDENCE SUITE STRESS TEST")
        print(f"{'='*60}")
        print(f"Concurrency: {self.concurrency}")
        print(f"Iterations: {self.iterations}")
        print(f"{'='*60}\n")

        start_time = time.time()
        all_results = []

        # Pipeline stress test
        print("Running pipeline stress tests...")
        try:
            pipeline_results = asyncio.run(self.stress_test_pipeline())
            all_results.extend(pipeline_results)
            passed = sum(1 for r in pipeline_results if r.success)
            print(f"  Pipeline: {passed}/{len(pipeline_results)} passed")
        except Exception as e:
            print(f"  Pipeline: FAILED - {e}")

        # Agent stress test
        print("Running agent stress tests...")
        try:
            agent_results = asyncio.run(self.stress_test_agents())
            all_results.extend(agent_results)
            passed = sum(1 for r in agent_results if r.success)
            print(f"  Agents: {passed}/{len(agent_results)} passed")
        except Exception as e:
            print(f"  Agents: FAILED - {e}")

        # Cache stress test
        print("Running cache stress tests...")
        try:
            cache_results = self.stress_test_cache()
            all_results.extend(cache_results)
            passed = sum(1 for r in cache_results if r.success)
            print(f"  Cache: {passed}/{len(cache_results)} passed")
        except Exception as e:
            print(f"  Cache: FAILED - {e}")

        total_duration = time.time() - start_time
        passed = sum(1 for r in all_results if r.success)
        failed = len(all_results) - passed

        # Calculate metrics
        durations = [r.duration_ms for r in all_results if r.duration_ms > 0]
        metrics = {}
        if durations:
            metrics = {
                "avg_duration_ms": round(statistics.mean(durations), 2),
                "median_duration_ms": round(statistics.median(durations), 2),
                "min_duration_ms": round(min(durations), 2),
                "max_duration_ms": round(max(durations), 2),
                "std_dev_ms": round(statistics.stdev(durations), 2) if len(durations) > 1 else 0,
                "p95_duration_ms": round(sorted(durations)[int(len(durations) * 0.95)], 2) if durations else 0,
                "p99_duration_ms": round(sorted(durations)[int(len(durations) * 0.99)], 2) if durations else 0,
                "throughput_ops_per_sec": round(len(durations) / total_duration, 2) if total_duration > 0 else 0,
            }

        report = StressTestReport(
            suite_name="Evidence Suite Stress Test",
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(all_results),
            passed=passed,
            failed=failed,
            total_duration_s=total_duration,
            results=all_results,
            metrics=metrics,
        )

        # Print summary
        print(f"\n{'='*60}")
        print("STRESS TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed}")
        print(f"Failed: {report.failed}")
        print(f"Pass Rate: {report.passed/report.total_tests*100:.1f}%")
        print(f"Total Duration: {report.total_duration_s:.2f}s")
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        return report


def run_stress_tests(
    concurrency: int = 10,
    iterations: int = 100,
    output_file: Optional[str] = None
) -> bool:
    """Run stress tests and optionally save report."""
    runner = StressTestRunner(concurrency=concurrency, iterations=iterations)
    report = runner.run_all_stress_tests()

    if output_file:
        report.save(output_file)
        print(f"Report saved to: {output_file}")

    return report.failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evidence Suite Stress Tests")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="Number of concurrent operations")
    parser.add_argument("-i", "--iterations", type=int, default=100,
                        help="Number of iterations per test")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file for JSON report")

    args = parser.parse_args()

    success = run_stress_tests(
        concurrency=args.concurrency,
        iterations=args.iterations,
        output_file=args.output
    )

    sys.exit(0 if success else 1)
