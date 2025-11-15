"""Metrics tracking: TTFT, TPOT, tokens/sec."""

import time
from collections import deque
from threading import Lock
from typing import Optional


class MetricsStats:
    """Thread-safe metrics collector for inference statistics."""

    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.

        Args:
            window_size: Number of recent samples to keep for rolling averages
        """
        self.window_size = window_size
        self._lock = Lock()

        # TTFT (Time To First Token) metrics
        self._ttft_samples = deque(maxlen=window_size)
        self._ttft_sum = 0.0
        self._ttft_count = 0

        # Tokens/sec metrics
        self._tokens_per_sec_samples = deque(maxlen=window_size)
        self._tokens_per_sec_sum = 0.0
        self._tokens_per_sec_count = 0

    def record_ttft(self, ttft_seconds: float):
        """Record Time To First Token metric.

        Args:
            ttft_seconds: Time to first token in seconds
        """
        with self._lock:
            self._ttft_samples.append(ttft_seconds)
            self._ttft_sum += ttft_seconds
            self._ttft_count += 1

            # Remove oldest if window exceeded
            if len(self._ttft_samples) > self.window_size:
                removed = self._ttft_samples.popleft()
                self._ttft_sum -= removed

    def record_tokens_per_sec(self, tokens_per_sec: float):
        """Record tokens per second metric.

        Args:
            tokens_per_sec: Tokens generated per second
        """
        with self._lock:
            self._tokens_per_sec_samples.append(tokens_per_sec)
            self._tokens_per_sec_sum += tokens_per_sec
            self._tokens_per_sec_count += 1

            # Remove oldest if window exceeded
            if len(self._tokens_per_sec_samples) > self.window_size:
                removed = self._tokens_per_sec_samples.popleft()
                self._tokens_per_sec_sum -= removed

    def get_ttft_stats(self) -> dict:
        """Get TTFT statistics.

        Returns:
            Dictionary with mean, min, max, p50, p95, p99, count
        """
        with self._lock:
            if not self._ttft_samples:
                return {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "count": 0,
                }

            samples = sorted(self._ttft_samples)
            n = len(samples)

            return {
                "mean": self._ttft_sum / n,
                "min": samples[0],
                "max": samples[-1],
                "p50": samples[int(n * 0.50)],
                "p95": samples[int(n * 0.95)] if n > 0 else 0.0,
                "p99": samples[int(n * 0.99)] if n > 0 else 0.0,
                "count": n,
            }

    def get_tokens_per_sec_stats(self) -> dict:
        """Get tokens/sec statistics.

        Returns:
            Dictionary with mean, min, max, p50, p95, p99, count
        """
        with self._lock:
            if not self._tokens_per_sec_samples:
                return {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "count": 0,
                }

            samples = sorted(self._tokens_per_sec_samples)
            n = len(samples)

            return {
                "mean": self._tokens_per_sec_sum / n,
                "min": samples[0],
                "max": samples[-1],
                "p50": samples[int(n * 0.50)],
                "p95": samples[int(n * 0.95)] if n > 0 else 0.0,
                "p99": samples[int(n * 0.99)] if n > 0 else 0.0,
                "count": n,
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._ttft_samples.clear()
            self._ttft_sum = 0.0
            self._ttft_count = 0
            self._tokens_per_sec_samples.clear()
            self._tokens_per_sec_sum = 0.0
            self._tokens_per_sec_count = 0


# Global metrics instance
_global_stats: Optional[MetricsStats] = None


def get_stats() -> MetricsStats:
    """Get global metrics stats instance."""
    global _global_stats
    if _global_stats is None:
        _global_stats = MetricsStats()
    return _global_stats


def reset_stats():
    """Reset global metrics stats."""
    global _global_stats
    if _global_stats is not None:
        _global_stats.reset()
