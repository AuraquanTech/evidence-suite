"""Evidence Suite - Forensic Visualizations
RTX 5090 Mobile telemetry and chain of custody visualization for court presentations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class SavantVisualizer:
    """Forensic Visualization Engine for the RTX 5090 Suite.

    Generates high-fidelity charts for:
    - Hardware telemetry audits
    - Behavioral analysis results
    - Chain of custody verification
    """

    # Color scheme for consistent forensic branding
    COLORS = {
        "primary": "#1a73e8",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "success": "#28a745",
        "neutral": "#6c757d",
        "thermal_safe": "#28a745",
        "thermal_warm": "#ffc107",
        "thermal_hot": "#fd7e14",
        "thermal_critical": "#dc3545",
    }

    def __init__(self, output_dir: str | None = None):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_hardware_telemetry(
        self,
        samples: list[dict[str, Any]],
        output_name: str = "gpu_telemetry_audit.png",
        title: str = "RTX 5090 Mobile Thermal & Memory Audit",
    ) -> str | None:
        """Generate hardware telemetry visualization.

        Args:
            samples: List of dicts with keys: elapsed_sec, gpu_temp_c, vram_used_mb, power_draw_w, thermal_state
            output_name: Output filename
            title: Chart title

        Returns:
            Path to saved image or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not samples:
            return None

        # Extract data
        times = [s.get("elapsed_sec", i) for i, s in enumerate(samples)]
        temps = [s.get("gpu_temp_c", 0) for s in samples]
        vram = [s.get("vram_used_mb", 0) for s in samples]
        power = [s.get("power_draw_w", 0) for s in samples]
        states = [s.get("thermal_state", "normal") for s in samples]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Temperature plot
        ax1 = axes[0]
        ax1.plot(times, temps, color=self.COLORS["danger"], linewidth=2, label="GPU Temp")
        ax1.axhline(y=82, color="darkred", linestyle="--", alpha=0.7, label="Thermal Limit (82°C)")
        ax1.axhline(y=70, color="orange", linestyle=":", alpha=0.5, label="Warning Zone (70°C)")
        ax1.fill_between(times, temps, alpha=0.3, color=self.COLORS["danger"])
        ax1.set_ylabel("Temperature (°C)", fontsize=10)
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Mark thermal events
        for i, state in enumerate(states):
            if state in ["hot", "critical"]:
                ax1.axvline(x=times[i], color="red", alpha=0.2, linewidth=3)

        # VRAM plot
        ax2 = axes[1]
        ax2.plot(times, vram, color=self.COLORS["primary"], linewidth=2, label="VRAM Used")
        ax2.axhline(y=20480, color="red", linestyle="--", alpha=0.7, label="Model Limit (20GB)")
        ax2.axhline(y=21504, color="darkred", linestyle="--", alpha=0.7, label="Warning (21GB)")
        ax2.fill_between(times, vram, alpha=0.3, color=self.COLORS["primary"])
        ax2.set_ylabel("VRAM (MB)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Power plot
        ax3 = axes[2]
        ax3.plot(times, power, color=self.COLORS["warning"], linewidth=2, label="Power Draw")
        ax3.fill_between(times, power, alpha=0.3, color=self.COLORS["warning"])
        ax3.set_ylabel("Power (W)", fontsize=10)
        ax3.set_xlabel("Time (seconds)", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_behavioral_confidence(
        self,
        results: list[dict[str, Any]],
        output_name: str = "behavioral_risk_assessment.png",
        title: str = "Behavioral Pattern Confidence Scores",
    ) -> str | None:
        """Generate behavioral analysis bar chart.

        Args:
            results: List of dicts with keys: type, score
            output_name: Output filename
            title: Chart title
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not results:
            return None

        labels = [r.get("type", f"Pattern {i}") for i, r in enumerate(results)]
        scores = [r.get("score", 0) for r in results]

        # Color by risk level
        colors = []
        for s in scores:
            if s > 0.75:
                colors.append(self.COLORS["danger"])
            elif s > 0.4:
                colors.append(self.COLORS["warning"])
            else:
                colors.append(self.COLORS["success"])

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.5)

        # Threshold lines
        ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.7, label="High Risk (0.75)")
        ax.axhline(y=0.40, color="orange", linestyle="--", alpha=0.7, label="Moderate Risk (0.40)")

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylabel("Confidence Score", fontsize=11)
        ax.set_xlabel("Behavioral Pattern", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Rotate labels if many
        if len(labels) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_sentiment_breakdown(
        self, sentiment: dict[str, float], output_name: str = "sentiment_breakdown.png"
    ) -> str | None:
        """Generate sentiment pie chart.

        Args:
            sentiment: Dict with keys: positive, negative, neutral, compound
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        labels = ["Positive", "Negative", "Neutral"]
        sizes = [
            sentiment.get("pos", sentiment.get("positive", 0)),
            sentiment.get("neg", sentiment.get("negative", 0)),
            sentiment.get("neu", sentiment.get("neutral", 0)),
        ]
        colors = [self.COLORS["success"], self.COLORS["danger"], self.COLORS["neutral"]]
        explode = (0.05, 0.05, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax.axis("equal")

        compound = sentiment.get("compound", 0)
        ax.set_title(
            f"Sentiment Analysis (Compound: {compound:.3f})", fontsize=12, fontweight="bold"
        )

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)


class ChainOfCustodyVisualizer:
    """Visualizes SHA-256 hash chains for non-technical stakeholders."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_chain(
        self,
        entries: list[dict[str, Any]],
        output_name: str = "chain_of_custody.png",
        title: str = "Evidence Integrity Verification: SHA-256 Hash Chain",
    ) -> str | None:
        """Generate chain of custody flowchart.

        Args:
            entries: List of dicts with keys: agent_id, action, output_hash, timestamp, verified (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not entries:
            return None

        fig, ax = plt.subplots(figsize=(14, 6))

        # Calculate positions
        n = len(entries)
        x_positions = list(range(n))
        y_position = 0.5

        # Draw nodes
        for i, entry in enumerate(entries):
            x = x_positions[i]

            # Determine color
            verified = entry.get("verified", True)
            color = "#90EE90" if verified else "#FFB6C1"  # Light green / light red

            # Draw box
            box = mpatches.FancyBboxPatch(
                (x - 0.4, y_position - 0.3),
                0.8,
                0.6,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(box)

            # Add text
            agent = entry.get("agent_id", entry.get("agent", f"Step {i + 1}"))[:12]
            action = entry.get("action", "")[:15]
            hash_val = entry.get("output_hash", entry.get("hash", ""))[:8]

            ax.text(
                x, y_position + 0.15, agent, ha="center", va="center", fontsize=8, fontweight="bold"
            )
            ax.text(x, y_position, action, ha="center", va="center", fontsize=7)
            ax.text(
                x,
                y_position - 0.15,
                f"#{hash_val}...",
                ha="center",
                va="center",
                fontsize=6,
                family="monospace",
            )

            # Draw arrow to next
            if i < n - 1:
                ax.annotate(
                    "",
                    xy=(x + 0.5, y_position),
                    xytext=(x + 0.4, y_position),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),
                )

        # Set limits and remove axes
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

        # Legend
        verified_patch = mpatches.Patch(color="#90EE90", label="Verified")
        failed_patch = mpatches.Patch(color="#FFB6C1", label="Verification Failed")
        ax.legend(handles=[verified_patch, failed_patch], loc="upper right", fontsize=9)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def visualize_chain_networkx(
        self, entries: list[dict[str, Any]], output_name: str = "chain_network.png"
    ) -> str | None:
        """Generate chain visualization using NetworkX (if available)."""
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None

        if not entries:
            return None

        G = nx.DiGraph()

        # Add nodes
        for i, entry in enumerate(entries):
            agent = entry.get("agent_id", entry.get("agent", f"Step {i + 1}"))
            hash_val = entry.get("output_hash", entry.get("hash", ""))[:8]
            label = f"{agent}\n#{hash_val}"
            verified = entry.get("verified", True)
            color = "lightgreen" if verified else "lightcoral"
            G.add_node(i, label=label, color=color)

            if i > 0:
                G.add_edge(i - 1, i)

        # Draw
        plt.figure(figsize=(12, 4))
        pos = {i: (i * 2, 0) for i in range(len(entries))}
        colors = [G.nodes[n]["color"] for n in G.nodes()]
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}

        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_color=colors,
            node_size=3000,
            node_shape="s",
            font_size=8,
            arrowsize=20,
            arrows=True,
        )

        plt.title("SHA-256 Hash Chain Integrity", fontsize=12, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)


class PipelinePerformanceVisualizer:
    """Visualizes pipeline performance metrics."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_latency_distribution(
        self,
        latencies_ms: list[float],
        output_name: str = "latency_distribution.png",
        title: str = "Pipeline Latency Distribution",
    ) -> str | None:
        """Generate latency histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not latencies_ms:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(latencies_ms, bins=30, color="skyblue", edgecolor="black", alpha=0.7)

        # Statistics
        mean_lat = sum(latencies_ms) / len(latencies_ms)
        p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]

        ax.axvline(
            x=mean_lat, color="green", linestyle="-", linewidth=2, label=f"Mean: {mean_lat:.1f}ms"
        )
        ax.axvline(x=p95, color="orange", linestyle="--", linewidth=2, label=f"P95: {p95:.1f}ms")
        ax.axvline(x=p99, color="red", linestyle="--", linewidth=2, label=f"P99: {p99:.1f}ms")

        ax.set_xlabel("Latency (ms)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_throughput_over_time(
        self,
        throughputs: list[float],
        interval_sec: float = 1.0,
        output_name: str = "throughput_timeline.png",
    ) -> str | None:
        """Generate throughput timeline."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not throughputs:
            return None

        times = [i * interval_sec for i in range(len(throughputs))]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(times, throughputs, color="blue", linewidth=2, marker="o", markersize=3)
        ax.fill_between(times, throughputs, alpha=0.3, color="blue")

        avg_throughput = sum(throughputs) / len(throughputs)
        ax.axhline(
            y=avg_throughput,
            color="green",
            linestyle="--",
            label=f"Average: {avg_throughput:.1f} items/sec",
        )

        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Throughput (items/sec)", fontsize=11)
        ax.set_title("Pipeline Throughput Over Time", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)
