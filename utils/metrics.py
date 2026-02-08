"""
Latency tracking utilities.

Provides a lightweight stopwatch that records the duration of named steps
in the RAG pipeline (DB queries, embedding, FAISS search, LLM call) so
that performance can be surfaced in the Streamlit UI.
"""

import time


class LatencyTracker:
    """Accumulates wall-clock durations for named processing steps."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start(self, step_name: str) -> None:
        """Record the start time for *step_name*."""
        self._starts[step_name] = time.perf_counter()

    def stop(self, step_name: str) -> None:
        """Record the end time and compute duration for *step_name*."""
        if step_name not in self._starts:
            return
        elapsed = time.perf_counter() - self._starts[step_name]
        self._durations[step_name] = round(elapsed, 4)

    def get_metrics(self) -> dict[str, float]:
        """Return all step durations plus a ``total`` key."""
        metrics = dict(self._durations)
        metrics["total"] = round(sum(self._durations.values()), 4)
        return metrics

    def format_metrics(self) -> str:
        """Return a human-readable summary string."""
        lines = []
        for step, dur in self._durations.items():
            lines.append(f"  {step}: {dur:.3f}s")
        total = sum(self._durations.values())
        lines.append(f"  ─────────────────")
        lines.append(f"  total: {total:.3f}s")
        return "\n".join(lines)
