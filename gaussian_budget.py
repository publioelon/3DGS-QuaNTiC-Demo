"""Cheap pre-inference Gaussian scheduling for compute-budgeted NTC playback.

The scheduler selects only a fraction of persistent base Gaussians before NTC
inference. Skipped Gaussians become stale by design; state_age tracks that debt
so a later temporal-rejoin mechanism/controller can decide when repair is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class GaussianBudgetStats:
    timestep: int
    total: int
    selected: int
    fraction: float
    mean_age: float
    p95_age: float
    max_age: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "timestep": float(self.timestep),
            "total": float(self.total),
            "selected": float(self.selected),
            "fraction": float(self.fraction),
            "mean_age": float(self.mean_age),
            "p95_age": float(self.p95_age),
            "max_age": float(self.max_age),
        }


class GaussianBudgetScheduler:
    """Deterministic compute-budget scheduler.

    mode="cyclic" walks a contiguous window through the base-Gaussian array.
    It has O(K) selection cost, no sorting, and deterministic bounded revisit
    time. For a fixed fraction f, every Gaussian is selected approximately once
    every ceil(1/f) update opportunities.

    The scheduler deliberately does not inspect current NTC outputs; doing so
    would require full inference and defeat compute reduction.
    """

    def __init__(
        self,
        fraction: float = 1.0,
        mode: str = "cyclic",
        log_interval: int = 25,
    ):
        self.fraction = float(fraction)
        self.mode = str(mode).strip().lower()
        self.log_interval = max(1, int(log_interval))
        self.state_age: Optional[torch.Tensor] = None
        self._cursor = 0

        if not (0.0 < self.fraction <= 1.0):
            raise ValueError("fraction must satisfy 0 < fraction <= 1")
        if self.mode not in {"cyclic"}:
            raise ValueError("currently supported mode: cyclic")

    def reset(self, n: Optional[int] = None, device=None) -> None:
        self._cursor = 0
        if n is None:
            self.state_age = None
        else:
            self.state_age = torch.zeros(
                int(n),
                dtype=torch.int32,
                device=device if device is not None else "cpu",
            )

    def _ensure_age(self, n: int, device) -> None:
        if (
            self.state_age is None
            or self.state_age.numel() != n
            or self.state_age.device != device
        ):
            self.reset(n=n, device=device)

    @torch.no_grad()
    def select(self, n: int, timestep: int, device) -> torch.Tensor:
        n = int(n)
        if n <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        self._ensure_age(n, device)

        if self.fraction >= 1.0:
            selected = torch.arange(n, dtype=torch.long, device=device)
        else:
            k = max(1, min(n, int(round(n * self.fraction))))
            start = self._cursor % n
            end = start + k

            if end <= n:
                selected = torch.arange(start, end, dtype=torch.long, device=device)
            else:
                first = torch.arange(start, n, dtype=torch.long, device=device)
                second = torch.arange(0, end - n, dtype=torch.long, device=device)
                selected = torch.cat((first, second), dim=0)

            self._cursor = end % n

        # Age means "number of NTC transitions since this Gaussian was last
        # actually evaluated/applied". Selected entries become current at this step.
        self.state_age.add_(1)
        self.state_age[selected] = 0
        return selected

    @torch.no_grad()
    def stats(self, timestep: int, selected: torch.Tensor) -> GaussianBudgetStats:
        if self.state_age is None:
            return GaussianBudgetStats(
                timestep=int(timestep),
                total=0,
                selected=int(selected.numel()),
                fraction=0.0,
                mean_age=0.0,
                p95_age=0.0,
                max_age=0,
            )

        age = self.state_age
        n = int(age.numel())
        k = int(selected.numel())

        if n == 0:
            p95 = 0.0
            mean = 0.0
            max_age = 0
        else:
            age_f = age.float()
            mean = float(age_f.mean().item())
            p95 = float(torch.quantile(age_f, 0.95).item())
            max_age = int(age.max().item())

        return GaussianBudgetStats(
            timestep=int(timestep),
            total=n,
            selected=k,
            fraction=(float(k) / float(n)) if n else 0.0,
            mean_age=mean,
            p95_age=p95,
            max_age=max_age,
        )

    def should_log(self, timestep: int) -> bool:
        return int(timestep) % self.log_interval == 0


def expected_revisit_frames(fraction: float) -> int:
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    return int(math.ceil(1.0 / fraction))
