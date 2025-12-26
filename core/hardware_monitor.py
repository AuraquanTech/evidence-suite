"""Evidence Suite - Hardware Monitor
RTX 5090 Mobile Edition thermal, power, and VRAM safeguards.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed - power monitoring disabled")


class PowerState(str, Enum):
    """Laptop power states."""

    AC = "ac"
    BATTERY = "battery"
    UNKNOWN = "unknown"


class ThermalState(str, Enum):
    """Thermal condition states."""

    NORMAL = "normal"  # < 70°C
    WARM = "warm"  # 70-80°C
    HOT = "hot"  # 80-85°C
    CRITICAL = "critical"  # > 85°C


@dataclass
class GPUStatus:
    """Current GPU status."""

    name: str
    temperature_c: float
    utilization_percent: float
    vram_used_mb: float
    vram_total_mb: float
    power_draw_w: float
    thermal_state: ThermalState

    @property
    def vram_free_mb(self) -> float:
        return self.vram_total_mb - self.vram_used_mb

    @property
    def vram_percent(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb) * 100 if self.vram_total_mb > 0 else 0


@dataclass
class SystemStatus:
    """Combined system status."""

    gpu: GPUStatus | None
    power_state: PowerState
    cpu_percent: float
    ram_percent: float
    battery_percent: float | None


class HardwareMonitor:
    """Real-time hardware monitoring for RTX 5090 Mobile.

    Features:
    - VRAM usage tracking with alerts
    - Thermal state detection
    - AC/Battery power awareness
    - Automatic throttling recommendations
    """

    # RTX 5090 Mobile thermal thresholds
    TEMP_NORMAL = 70
    TEMP_WARM = 80
    TEMP_HOT = 85
    TEMP_CRITICAL = 90

    # VRAM thresholds (24GB total)
    VRAM_TOTAL_MB = 24576  # 24GB
    VRAM_MODEL_LIMIT_MB = 20480  # 20GB for models
    VRAM_WARNING_MB = 21504  # 21GB triggers pause
    VRAM_CRITICAL_MB = 23040  # 22.5GB emergency stop

    def __init__(
        self,
        check_interval: float = 5.0,
        on_thermal_warning: Callable[[ThermalState], None] | None = None,
        on_vram_warning: Callable[[float], None] | None = None,
        on_power_change: Callable[[PowerState], None] | None = None,
    ):
        self.check_interval = check_interval
        self._on_thermal_warning = on_thermal_warning
        self._on_vram_warning = on_vram_warning
        self._on_power_change = on_power_change

        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._last_power_state = PowerState.UNKNOWN
        self._last_status: SystemStatus | None = None

        # Cooldown tracking
        self._cooldown_active = False
        self._cooldown_until = 0.0

    def get_gpu_status(self) -> GPUStatus | None:
        """Query NVIDIA GPU status via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 6:
                return None

            temp = float(parts[1])
            thermal_state = self._classify_thermal(temp)

            return GPUStatus(
                name=parts[0],
                temperature_c=temp,
                utilization_percent=float(parts[2]),
                vram_used_mb=float(parts[3]),
                vram_total_mb=float(parts[4]),
                power_draw_w=float(parts[5]) if parts[5] != "[N/A]" else 0.0,
                thermal_state=thermal_state,
            )

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.debug(f"GPU query failed: {e}")
            return None

    def get_power_state(self) -> PowerState:
        """Check if laptop is on AC or battery power."""
        if not PSUTIL_AVAILABLE:
            return PowerState.UNKNOWN

        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return PowerState.AC  # Desktop or no battery
            return PowerState.AC if battery.power_plugged else PowerState.BATTERY
        except Exception:
            return PowerState.UNKNOWN

    def get_battery_percent(self) -> float | None:
        """Get battery percentage if available."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery else None
        except Exception:
            return None

    def get_system_status(self) -> SystemStatus:
        """Get complete system status."""
        gpu = self.get_gpu_status()
        power = self.get_power_state()

        cpu_percent = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
        ram_percent = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0
        battery = self.get_battery_percent()

        status = SystemStatus(
            gpu=gpu,
            power_state=power,
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            battery_percent=battery,
        )

        self._last_status = status
        return status

    def _classify_thermal(self, temp_c: float) -> ThermalState:
        """Classify GPU temperature into thermal state."""
        if temp_c < self.TEMP_NORMAL:
            return ThermalState.NORMAL
        if temp_c < self.TEMP_WARM:
            return ThermalState.WARM
        if temp_c < self.TEMP_HOT:
            return ThermalState.HOT
        return ThermalState.CRITICAL

    def should_throttle(self) -> dict[str, Any]:
        """Determine if processing should be throttled.

        Returns dict with:
        - throttle: bool - whether to throttle
        - reason: str - why throttling is recommended
        - severity: str - low/medium/high/critical
        - recommended_delay: float - seconds to wait
        """
        status = self.get_system_status()

        # Check cooldown
        if self._cooldown_active and time.time() < self._cooldown_until:
            remaining = self._cooldown_until - time.time()
            return {
                "throttle": True,
                "reason": "cooldown_active",
                "severity": "medium",
                "recommended_delay": remaining,
            }

        self._cooldown_active = False

        # Check power state
        if status.power_state == PowerState.BATTERY:
            return {
                "throttle": True,
                "reason": "battery_power",
                "severity": "medium",
                "recommended_delay": 0,
                "reduce_workers": 0.5,  # 50% reduction
            }

        # Check GPU status
        if status.gpu:
            # VRAM critical
            if status.gpu.vram_used_mb > self.VRAM_CRITICAL_MB:
                return {
                    "throttle": True,
                    "reason": "vram_critical",
                    "severity": "critical",
                    "recommended_delay": 30,
                    "pause_refinement": True,
                }

            # VRAM warning
            if status.gpu.vram_used_mb > self.VRAM_WARNING_MB:
                if self._on_vram_warning:
                    self._on_vram_warning(status.gpu.vram_used_mb)
                return {
                    "throttle": True,
                    "reason": "vram_high",
                    "severity": "high",
                    "recommended_delay": 10,
                    "pause_refinement": True,
                }

            # Thermal critical
            if status.gpu.thermal_state == ThermalState.CRITICAL:
                if self._on_thermal_warning:
                    self._on_thermal_warning(status.gpu.thermal_state)
                return {
                    "throttle": True,
                    "reason": "thermal_critical",
                    "severity": "critical",
                    "recommended_delay": 30,
                }

            # Thermal hot
            if status.gpu.thermal_state == ThermalState.HOT:
                return {
                    "throttle": True,
                    "reason": "thermal_hot",
                    "severity": "high",
                    "recommended_delay": 10,
                }

        return {"throttle": False, "reason": None, "severity": None, "recommended_delay": 0}

    def activate_cooldown(self, duration: float = 30.0):
        """Activate a cooldown period after intensive processing."""
        self._cooldown_active = True
        self._cooldown_until = time.time() + duration
        logger.info(f"Cooldown activated for {duration}s")

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Hardware monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Hardware monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                status = self.get_system_status()

                # Check for power state changes
                if status.power_state != self._last_power_state:
                    if self._on_power_change:
                        self._on_power_change(status.power_state)
                    logger.info(f"Power state changed: {status.power_state}")
                    self._last_power_state = status.power_state

                # Log warnings
                if status.gpu:
                    if status.gpu.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
                        logger.warning(
                            f"GPU thermal warning: {status.gpu.temperature_c}°C "
                            f"({status.gpu.thermal_state})"
                        )

                    if status.gpu.vram_used_mb > self.VRAM_WARNING_MB:
                        logger.warning(
                            f"VRAM usage high: {status.gpu.vram_used_mb:.0f}MB / "
                            f"{status.gpu.vram_total_mb:.0f}MB"
                        )

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.check_interval)

    def get_optimal_workers(self, base_workers: int = 16) -> int:
        """Get optimal worker count based on current conditions."""
        status = self.get_system_status()
        workers = base_workers

        # Reduce for battery
        if status.power_state == PowerState.BATTERY:
            workers = workers // 2

        # Reduce for thermal
        if status.gpu:
            if status.gpu.thermal_state == ThermalState.CRITICAL:
                workers = workers // 4
            elif status.gpu.thermal_state == ThermalState.HOT:
                workers = workers // 2
            elif status.gpu.thermal_state == ThermalState.WARM:
                workers = int(workers * 0.75)

        return max(1, workers)

    def format_status(self) -> str:
        """Format current status for display."""
        status = self.get_system_status()
        lines = [
            "┌─────────────────────────────────────────┐",
            "│         HARDWARE STATUS                 │",
            "├─────────────────────────────────────────┤",
        ]

        if status.gpu:
            lines.extend(
                [
                    f"│ GPU: {status.gpu.name[:35]:<35} │",
                    f"│ Temp: {status.gpu.temperature_c:.0f}°C ({status.gpu.thermal_state.value:<8})      │",
                    f"│ VRAM: {status.gpu.vram_used_mb:.0f}/{status.gpu.vram_total_mb:.0f} MB ({status.gpu.vram_percent:.1f}%)     │",
                    f"│ Util: {status.gpu.utilization_percent:.0f}%  Power: {status.gpu.power_draw_w:.0f}W          │",
                ]
            )
        else:
            lines.append("│ GPU: Not detected                       │")

        lines.extend(
            [
                "├─────────────────────────────────────────┤",
                f"│ Power: {status.power_state.value.upper():<10}                      │",
                f"│ CPU: {status.cpu_percent:.1f}%  RAM: {status.ram_percent:.1f}%              │",
            ]
        )

        if status.battery_percent is not None:
            lines.append(f"│ Battery: {status.battery_percent:.0f}%                          │")

        lines.append("└─────────────────────────────────────────┘")

        return "\n".join(lines)


# Singleton instance for easy access
_monitor: HardwareMonitor | None = None


def get_monitor() -> HardwareMonitor:
    """Get or create the global hardware monitor."""
    global _monitor
    if _monitor is None:
        _monitor = HardwareMonitor()
    return _monitor
