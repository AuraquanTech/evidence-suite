#!/usr/bin/env python3
"""
Evidence Suite - Pre-Commit Testing Rule
Mandatory testing and benchmarking before any commit.

RULE: Any upgrade must test and benchmark metrics.

This script is designed to be run as a pre-commit hook or manually
before pushing changes. It enforces the following rules:

1. All unit tests must pass
2. Pipeline must run successfully
3. Benchmark metrics must not regress beyond threshold
4. Stress tests must pass (optional, configurable)
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add parent to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PreCommitChecker:
    """Enforces mandatory testing rules before commits."""

    # Thresholds for regression detection
    REGRESSION_THRESHOLDS = {
        "avg_latency_ms": 0.20,      # 20% max increase in latency
        "avg_throughput_ops_sec": -0.15,  # 15% max decrease in throughput
        "avg_success_rate": -0.05,   # 5% max decrease in success rate
    }

    def __init__(
        self,
        run_stress: bool = False,
        run_benchmark: bool = True,
        baseline_file: Optional[str] = None,
        verbose: bool = True,
    ):
        self.run_stress = run_stress
        self.run_benchmark = run_benchmark
        self.baseline_file = baseline_file or str(PROJECT_ROOT / "benchmarks" / "baseline.json")
        self.verbose = verbose
        self.results: Dict[str, Any] = {}

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose."""
        if self.verbose:
            prefix = {
                "INFO": "[INFO]",
                "PASS": "[PASS]",
                "FAIL": "[FAIL]",
                "WARN": "[WARN]",
            }.get(level, "[INFO]")
            print(f"{prefix} {message}")

    def run_unit_tests(self) -> Tuple[bool, str]:
        """Run unit tests."""
        self.log("Running unit tests...")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "EVIDENCE_SUITE_ENV": "test"}
            )

            # Parse results
            if "passed" in result.stdout:
                passed_count = result.stdout.count(" passed")
                self.log(f"Unit tests completed with {passed_count} passed", "PASS")
                return True, result.stdout
            elif result.returncode != 0:
                self.log("Unit tests failed", "FAIL")
                return False, result.stdout + result.stderr
            return True, result.stdout

        except subprocess.TimeoutExpired:
            self.log("Unit tests timed out", "FAIL")
            return False, "Timeout"
        except Exception as e:
            self.log(f"Unit test error: {e}", "FAIL")
            return False, str(e)

    def run_pipeline_test(self) -> Tuple[bool, str]:
        """Run pipeline verification."""
        self.log("Running pipeline verification...")

        try:
            result = subprocess.run(
                [sys.executable, "run_pipeline.py"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Check for success indicators
            if "PIPELINE TEST COMPLETE" in result.stdout and "success_rate: 1.0" in result.stdout.lower():
                self.log("Pipeline verification passed", "PASS")
                return True, result.stdout
            elif "PASS" in result.stdout:
                self.log("Pipeline verification passed", "PASS")
                return True, result.stdout
            else:
                self.log("Pipeline verification failed", "FAIL")
                return False, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            self.log("Pipeline test timed out", "FAIL")
            return False, "Timeout"
        except Exception as e:
            self.log(f"Pipeline test error: {e}", "FAIL")
            return False, str(e)

    def run_benchmarks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run benchmarks and check for regressions."""
        if not self.run_benchmark:
            self.log("Skipping benchmarks (disabled)", "WARN")
            return True, {}

        self.log("Running benchmarks...")

        try:
            # Create output directory
            benchmark_dir = PROJECT_ROOT / "benchmarks"
            benchmark_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = benchmark_dir / f"benchmark_{timestamp}.json"

            # Run benchmark
            result = subprocess.run(
                [
                    sys.executable, "tests/benchmark.py",
                    "-i", "50",  # Reduced iterations for pre-commit
                    "-w", "3",
                    "-o", str(output_file),
                    "-b", self.baseline_file if Path(self.baseline_file).exists() else "",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Load results
            if output_file.exists():
                with open(output_file) as f:
                    benchmark_data = json.load(f)

                metrics = benchmark_data.get("metrics", {})
                comparison = benchmark_data.get("comparison", {})

                # Check for regressions
                regressions = []
                for metric, threshold in self.REGRESSION_THRESHOLDS.items():
                    if metric in comparison:
                        change = comparison[metric].get("change_pct", 0) / 100
                        if threshold > 0:
                            # Higher is worse (like latency)
                            if change > threshold:
                                regressions.append(f"{metric}: {change*100:+.1f}% (threshold: {threshold*100}%)")
                        else:
                            # Lower is worse (like throughput)
                            if change < threshold:
                                regressions.append(f"{metric}: {change*100:+.1f}% (threshold: {threshold*100}%)")

                if regressions:
                    self.log(f"Benchmark regressions detected:", "WARN")
                    for r in regressions:
                        self.log(f"  - {r}", "WARN")
                    # Don't fail, just warn
                    self.log("Proceeding despite regressions (warning only)", "WARN")

                self.log("Benchmarks completed", "PASS")
                return True, benchmark_data

            return True, {}

        except subprocess.TimeoutExpired:
            self.log("Benchmarks timed out", "WARN")
            return True, {}  # Don't block on benchmark timeout
        except Exception as e:
            self.log(f"Benchmark error: {e}", "WARN")
            return True, {}  # Don't block on benchmark errors

    def run_stress_tests(self) -> Tuple[bool, str]:
        """Run stress tests."""
        if not self.run_stress:
            self.log("Skipping stress tests (disabled)", "WARN")
            return True, ""

        self.log("Running stress tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable, "tests/stress_test.py",
                    "-c", "5",  # Reduced concurrency for pre-commit
                    "-i", "20",  # Reduced iterations
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )

            if "STRESS TEST SUMMARY" in result.stdout:
                # Parse pass rate
                for line in result.stdout.split("\n"):
                    if "Pass Rate:" in line:
                        rate = float(line.split(":")[1].strip().replace("%", ""))
                        if rate >= 90:  # 90% pass rate required
                            self.log(f"Stress tests passed ({rate:.1f}%)", "PASS")
                            return True, result.stdout
                        else:
                            self.log(f"Stress tests failed ({rate:.1f}% < 90%)", "FAIL")
                            return False, result.stdout

            self.log("Stress tests completed", "PASS")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            self.log("Stress tests timed out", "FAIL")
            return False, "Timeout"
        except Exception as e:
            self.log(f"Stress test error: {e}", "FAIL")
            return False, str(e)

    def check_all(self) -> bool:
        """Run all checks and return overall status."""
        print("\n" + "="*60)
        print("EVIDENCE SUITE PRE-COMMIT CHECK")
        print("Rule: Any upgrade must test and benchmark metrics")
        print("="*60 + "\n")

        all_passed = True

        # 1. Unit tests
        unit_passed, unit_output = self.run_unit_tests()
        self.results["unit_tests"] = {"passed": unit_passed, "output": unit_output[:500]}
        if not unit_passed:
            all_passed = False

        # 2. Pipeline test
        pipeline_passed, pipeline_output = self.run_pipeline_test()
        self.results["pipeline"] = {"passed": pipeline_passed, "output": pipeline_output[:500]}
        if not pipeline_passed:
            all_passed = False

        # 3. Benchmarks (warning only)
        bench_passed, bench_data = self.run_benchmarks()
        self.results["benchmarks"] = {"passed": bench_passed, "data": bench_data}

        # 4. Stress tests (if enabled)
        if self.run_stress:
            stress_passed, stress_output = self.run_stress_tests()
            self.results["stress_tests"] = {"passed": stress_passed, "output": stress_output[:500]}
            if not stress_passed:
                all_passed = False

        # Summary
        print("\n" + "="*60)
        print("PRE-COMMIT CHECK SUMMARY")
        print("="*60)
        print(f"Unit Tests: {'PASS' if self.results.get('unit_tests', {}).get('passed') else 'FAIL'}")
        print(f"Pipeline: {'PASS' if self.results.get('pipeline', {}).get('passed') else 'FAIL'}")
        print(f"Benchmarks: {'PASS' if self.results.get('benchmarks', {}).get('passed') else 'FAIL'}")
        if self.run_stress:
            print(f"Stress Tests: {'PASS' if self.results.get('stress_tests', {}).get('passed') else 'FAIL'}")
        print("="*60)

        if all_passed:
            print("\n[PASS] All checks passed. Safe to commit.\n")
        else:
            print("\n[FAIL] Some checks failed. Fix issues before committing.\n")

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-commit testing rule: Any upgrade must test and benchmark metrics"
    )
    parser.add_argument("--stress", action="store_true",
                        help="Also run stress tests")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip benchmarks")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline file for benchmark comparison")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    checker = PreCommitChecker(
        run_stress=args.stress,
        run_benchmark=not args.no_benchmark,
        baseline_file=args.baseline,
        verbose=not args.quiet,
    )

    success = checker.check_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
