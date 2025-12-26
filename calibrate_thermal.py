"""Evidence Suite - RTX 5090 Mobile Thermal Calibration Suite
Finds the optimal "Goldilocks Zone" for sustained forensic processing.

This script:
1. Runs controlled GPU stress tests at 80% target load
2. Monitors thermal response in real-time
3. Detects throttling points
4. Generates a calibration report
5. Saves thermal profile for production use
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed - using synthetic load")

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("WARNING: pynvml not installed - using nvidia-smi fallback")

from loguru import logger

from core.hardware_monitor import HardwareMonitor, ThermalState


@dataclass
class ThermalSample:
    """Single thermal measurement."""

    timestamp: str
    elapsed_sec: float
    gpu_temp_c: float
    gpu_util_percent: float
    vram_used_mb: float
    vram_total_mb: float
    power_draw_w: float
    thermal_state: str


@dataclass
class CalibrationResult:
    """Complete calibration result."""

    system_name: str
    gpu_name: str
    test_duration_sec: float
    target_load: float
    samples: list[ThermalSample]

    # Derived metrics
    avg_temp: float
    max_temp: float
    min_temp: float
    temp_stability: float  # Standard deviation
    avg_power: float
    throttle_events: int
    recommended_load: float
    thermal_headroom: float


class MobileThermalCalibrator:
    """RTX 5090 Mobile Thermal Calibration Suite.

    Stress tests the GPU at controlled load levels to find the optimal
    sustained processing threshold that avoids thermal throttling.
    """

    # RTX 5090 Mobile thresholds
    TEMP_SAFE = 80
    TEMP_WARNING = 85
    TEMP_CRITICAL = 90

    def __init__(self, target_load: float = 0.80):
        """Initialize calibrator.

        Args:
            target_load: Target GPU utilization (0.0-1.0), default 80%
        """
        self.target_load = target_load
        self.samples: list[ThermalSample] = []
        self.throttle_events = 0

        # Initialize NVML if available
        self._nvml_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                logger.info(f"NVML initialized: {gpu_name}")
            except Exception as e:
                logger.warning(f"NVML init failed: {e}")

        # Fallback to HardwareMonitor
        self.monitor = HardwareMonitor()

        logger.info(f"Thermal Calibrator initialized (target: {target_load * 100:.0f}%)")

    def get_gpu_stats(self) -> dict:
        """Get current GPU statistics."""
        if self._nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0

                return {
                    "temp": float(temp),
                    "util": float(util.gpu),
                    "vram_used": float(mem.used) / (1024**2),
                    "vram_total": float(mem.total) / (1024**2),
                    "power": float(power),
                }
            except Exception as e:
                logger.debug(f"NVML query error: {e}")

        # Fallback to HardwareMonitor
        status = self.monitor.get_gpu_status()
        if status:
            return {
                "temp": status.temperature_c,
                "util": status.utilization_percent,
                "vram_used": status.vram_used_mb,
                "vram_total": status.vram_total_mb,
                "power": status.power_draw_w,
            }

        return {"temp": 0, "util": 0, "vram_used": 0, "vram_total": 0, "power": 0}

    def _create_workload(self, size: int = 8192):
        """Create GPU matrices for stress testing."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None, None

        # Use bfloat16 for Blackwell architecture efficiency
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)

        logger.info(f"Created {size}x{size} matrices ({dtype})")
        return a, b

    def _run_compute_cycle(self, a, b, iterations: int = 10):
        """Run compute iterations to generate load."""
        if a is None or b is None:
            time.sleep(0.1)  # Synthetic delay
            return

        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    def calibrate(
        self,
        duration_minutes: float = 2.0,
        sample_interval: float = 1.0,
        matrix_size: int = 10240,  # Optimized for RTX 5090 24GB
    ) -> CalibrationResult:
        """Run thermal calibration test.

        Args:
            duration_minutes: Test duration in minutes
            sample_interval: Seconds between samples
            matrix_size: Size of stress matrices (larger = more load)

        Returns:
            CalibrationResult with thermal profile
        """
        duration_sec = duration_minutes * 60
        self.samples = []
        self.throttle_events = 0

        print("\n" + "=" * 60)
        print("  RTX 5090 MOBILE THERMAL CALIBRATION")
        print("=" * 60)
        print(f"  Target Load:    {self.target_load * 100:.0f}%")
        print(f"  Duration:       {duration_minutes:.1f} minutes")
        print(f"  Matrix Size:    {matrix_size}x{matrix_size}")
        print("=" * 60 + "\n")

        # Pre-test status
        stats = self.get_gpu_stats()
        print(f"Initial State: {stats['temp']:.0f}°C, {stats['util']:.0f}% util\n")

        # Create workload
        a, b = self._create_workload(matrix_size)

        start_time = time.time()
        last_sample = 0

        print("Starting stress test...")
        print("-" * 60)
        print(f"{'Time':>8} | {'Temp':>6} | {'Util':>6} | {'VRAM':>10} | {'Power':>7} | Status")
        print("-" * 60)

        try:
            while (time.time() - start_time) < duration_sec:
                elapsed = time.time() - start_time

                # Get current stats
                stats = self.get_gpu_stats()
                temp = stats["temp"]
                util = stats["util"]

                # Determine thermal state
                if temp >= self.TEMP_CRITICAL:
                    thermal_state = ThermalState.CRITICAL
                elif temp >= self.TEMP_WARNING:
                    thermal_state = ThermalState.HOT
                elif temp >= self.TEMP_SAFE:
                    thermal_state = ThermalState.WARM
                else:
                    thermal_state = ThermalState.NORMAL

                # Sample at interval
                if elapsed - last_sample >= sample_interval:
                    sample = ThermalSample(
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        elapsed_sec=round(elapsed, 1),
                        gpu_temp_c=temp,
                        gpu_util_percent=util,
                        vram_used_mb=stats["vram_used"],
                        vram_total_mb=stats["vram_total"],
                        power_draw_w=stats["power"],
                        thermal_state=thermal_state.value,
                    )
                    self.samples.append(sample)
                    last_sample = elapsed

                    # Status indicator
                    status = ""
                    if thermal_state == ThermalState.CRITICAL:
                        status = "CRITICAL!"
                        self.throttle_events += 1
                    elif thermal_state == ThermalState.HOT:
                        status = "HOT"
                    elif thermal_state == ThermalState.WARM:
                        status = "WARM"
                    else:
                        status = "OK"

                    print(
                        f"{elapsed:>7.1f}s | {temp:>5.1f}C | {util:>5.1f}% | "
                        f"{stats['vram_used']:>6.0f}MB | {stats['power']:>6.1f}W | {status}"
                    )

                # Thermal protection
                if temp >= self.TEMP_CRITICAL:
                    print(f"\n⚠️  THERMAL SPIKE: {temp}°C - Injecting 10s cooldown...")
                    time.sleep(10)
                    continue

                if temp >= self.TEMP_WARNING:
                    print(f"⚠️  Thermal warning: {temp}°C - Reducing load...")
                    time.sleep(2)
                    continue

                # Dynamic load adjustment
                if util < (self.target_load * 100):
                    # Increase load
                    iterations = int(30 * (self.target_load - util / 100 + 0.1))
                    self._run_compute_cycle(a, b, max(5, iterations))
                else:
                    # Maintain current load
                    self._run_compute_cycle(a, b, 10)

                time.sleep(0.1)  # Prevent busy loop

        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user.")

        finally:
            # Cleanup
            if PYNVML_AVAILABLE and self._nvml_handle:
                pynvml.nvmlShutdown()

            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate result
        return self._analyze_results(duration_sec)

    def _analyze_results(self, duration_sec: float) -> CalibrationResult:
        """Analyze calibration samples and generate report."""
        if not self.samples:
            logger.error("No samples collected!")
            return None

        temps = [s.gpu_temp_c for s in self.samples]
        powers = [s.power_draw_w for s in self.samples]

        avg_temp = sum(temps) / len(temps)
        max_temp = max(temps)
        min_temp = min(temps)

        # Temperature stability (std dev)
        mean_temp = avg_temp
        variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
        temp_stability = variance**0.5

        avg_power = sum(powers) / len(powers) if powers else 0

        # Recommend load based on thermal headroom
        thermal_headroom = self.TEMP_WARNING - max_temp

        if thermal_headroom < 0:
            recommended_load = max(0.5, self.target_load - 0.15)
        elif thermal_headroom < 5:
            recommended_load = max(0.6, self.target_load - 0.10)
        elif thermal_headroom < 10:
            recommended_load = self.target_load
        else:
            recommended_load = min(0.95, self.target_load + 0.05)

        # Get GPU name
        stats = self.get_gpu_stats()
        gpu_name = "RTX 5090 Laptop GPU"  # Default

        result = CalibrationResult(
            system_name="Evidence Suite Mobile",
            gpu_name=gpu_name,
            test_duration_sec=duration_sec,
            target_load=self.target_load,
            samples=self.samples,
            avg_temp=round(avg_temp, 1),
            max_temp=round(max_temp, 1),
            min_temp=round(min_temp, 1),
            temp_stability=round(temp_stability, 2),
            avg_power=round(avg_power, 1),
            throttle_events=self.throttle_events,
            recommended_load=round(recommended_load, 2),
            thermal_headroom=round(thermal_headroom, 1),
        )

        self._print_report(result)
        self._save_results(result)

        return result

    def _print_report(self, result: CalibrationResult):
        """Print calibration report."""
        print("\n" + "=" * 60)
        print("  CALIBRATION RESULTS")
        print("=" * 60)
        print(f"  GPU:              {result.gpu_name}")
        print(f"  Test Duration:    {result.test_duration_sec:.0f} seconds")
        print(f"  Samples:          {len(result.samples)}")
        print("-" * 60)
        print("  THERMAL PROFILE:")
        print(f"    Average Temp:   {result.avg_temp}°C")
        print(f"    Max Temp:       {result.max_temp}°C")
        print(f"    Min Temp:       {result.min_temp}°C")
        print(f"    Stability (σ):  {result.temp_stability}°C")
        print(f"    Headroom:       {result.thermal_headroom}°C to warning")
        print("-" * 60)
        print("  POWER:")
        print(f"    Average Draw:   {result.avg_power}W")
        print("-" * 60)
        print("  EVENTS:")
        print(f"    Throttle Events: {result.throttle_events}")
        print("-" * 60)
        print("  RECOMMENDATION:")
        print(f"    Target Load:    {result.target_load * 100:.0f}%")
        print(f"    Recommended:    {result.recommended_load * 100:.0f}%")

        if result.recommended_load < result.target_load:
            print(f"\n  ⚠️  REDUCE LOAD to {result.recommended_load * 100:.0f}% for stability")
        elif result.recommended_load > result.target_load:
            print(
                f"\n  ✅  HEADROOM AVAILABLE - can increase to {result.recommended_load * 100:.0f}%"
            )
        else:
            print("\n  ✅  OPTIMAL - 80% load is appropriate for your cooling")

        print("=" * 60 + "\n")

    def _save_results(self, result: CalibrationResult):
        """Save calibration results to files."""
        output_dir = Path(__file__).parent / "calibration_output"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results as JSON
        json_path = output_dir / f"thermal_calibration_{timestamp}.json"
        with open(json_path, "w") as f:
            data = asdict(result)
            # Convert samples to dicts
            data["samples"] = [asdict(s) for s in result.samples]
            json.dump(data, f, indent=2)

        # Save CSV for analysis
        csv_path = output_dir / f"thermal_samples_{timestamp}.csv"
        with open(csv_path, "w") as f:
            headers = "timestamp,elapsed_sec,gpu_temp_c,gpu_util_percent,vram_used_mb,power_draw_w,thermal_state\n"
            f.write(headers)
            f.writelines(
                f"{s.timestamp},{s.elapsed_sec},{s.gpu_temp_c},{s.gpu_util_percent},"
                f"{s.vram_used_mb},{s.power_draw_w},{s.thermal_state}\n"
                for s in result.samples
            )

        # Save recommended config
        config_path = output_dir / "recommended_config.yaml"
        with open(config_path, "w") as f:
            f.write(f"""# RTX 5090 Mobile Calibration Results
# Generated: {datetime.now().isoformat()}

hardware:
  thermal:
    max_sustained_load: {result.recommended_load}
    calibrated_max_temp: {result.max_temp}
    thermal_headroom: {result.thermal_headroom}
    stability_score: {max(0, 10 - result.temp_stability):.1f}/10

  # Copy these values to your main config.yaml
  calibration_status: "complete"
  calibration_date: "{datetime.now().strftime("%Y-%m-%d")}"
""")

        print(f"Results saved to: {output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {csv_path.name}")
        print(f"  - {config_path.name}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RTX 5090 Mobile Thermal Calibration Suite")
    parser.add_argument(
        "--duration", "-d", type=float, default=2.0, help="Test duration in minutes (default: 2)"
    )
    parser.add_argument(
        "--target-load",
        "-t",
        type=float,
        default=0.80,
        help="Target GPU load 0.0-1.0 (default: 0.80)",
    )
    parser.add_argument(
        "--matrix-size",
        "-m",
        type=int,
        default=10240,
        help="Matrix size for stress test (default: 10240)",
    )

    args = parser.parse_args()

    calibrator = MobileThermalCalibrator(target_load=args.target_load)
    result = calibrator.calibrate(duration_minutes=args.duration, matrix_size=args.matrix_size)

    if result:
        print("\n✅ Calibration complete!")
        print(f"Recommended sustained load: {result.recommended_load * 100:.0f}%")
    else:
        print("\n❌ Calibration failed - check GPU availability")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
