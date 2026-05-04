"""Shared test fixtures and configuration."""

import pytest

# Reusable benchmark configuration
default_benchmark = pytest.mark.benchmark(
    disable_gc=True,
    warmup=True,
    min_rounds=10,
    max_time=0.005,  # 5 ms timeout
)
