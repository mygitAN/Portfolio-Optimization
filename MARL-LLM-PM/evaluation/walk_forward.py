"""
Walk-forward split manager.

Implements the fixed evaluation protocol from the thesis:
  - Training:          first 60% of sample
  - Validation:        next 20% of sample
  - Walk-forward test: final 20%, split into non-overlapping 6-month windows
  - Held-out:          last 12 months (SEALED — evaluated exactly once)

The split is defined here as proportions over calendar dates.
Once final data is confirmed, the exact calendar dates will be
printed and logged for reproducibility.

INTEGRITY COMMITMENT:
  The held_out split is exposed as a separate property that must
  only be accessed after all design decisions, hyperparameters, and
  architecture choices are frozen. This is enforced via a simple
  access-control flag; the caller is responsible for setting
  `allow_held_out=True` only at final evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp   # inclusive

    def __str__(self) -> str:
        return f"{self.name}: {self.start.date()} → {self.end.date()}"


@dataclass
class WalkForwardSchedule:
    training: DataSplit
    validation: DataSplit
    walk_forward_windows: list[DataSplit]
    held_out: DataSplit
    _held_out_accessed: bool = field(default=False, init=False, repr=False)

    def print_schedule(self) -> None:
        print("\n=== Walk-Forward Evaluation Schedule ===")
        print(self.training)
        print(self.validation)
        for w in self.walk_forward_windows:
            print(w)
        print(f"HELD-OUT (sealed): {self.held_out.start.date()} → {self.held_out.end.date()}")
        print("=" * 42)

    def get_held_out(self, allow: bool = False) -> DataSplit:
        if not allow:
            raise RuntimeError(
                "Held-out split access is BLOCKED. "
                "Set allow=True only after all design decisions are frozen "
                "and you are ready for final evaluation."
            )
        self._held_out_accessed = True
        return self.held_out


class WalkForwardManager:
    """
    Builds and manages walk-forward data splits.

    Args:
        dates: full DatetimeIndex of available trading dates.
        config: walk_forward section of experiment config.
    """

    def __init__(self, dates: pd.DatetimeIndex, config: dict):
        self.dates = dates
        self.cfg = config
        self._schedule: WalkForwardSchedule | None = None

    def build_schedule(self) -> WalkForwardSchedule:
        dates = self.dates
        n = len(dates)

        train_frac = self.cfg.get("train_frac", 0.60)
        val_frac = self.cfg.get("val_frac", 0.20)
        # test_frac = 1 - train_frac - val_frac = 0.20

        train_end_idx = int(n * train_frac) - 1
        val_end_idx = int(n * (train_frac + val_frac)) - 1

        training = DataSplit("Training", dates[0], dates[train_end_idx])
        validation = DataSplit("Validation", dates[train_end_idx + 1], dates[val_end_idx])

        # Walk-forward windows: 6-month non-overlapping from val_end
        wf_start = dates[val_end_idx + 1]
        wf_end = dates[-1]
        held_out_months = self.cfg.get("held_out_months", 12)
        window_months = self.cfg.get("test_window_months", 6)

        # Held-out = last held_out_months of total sample
        held_out_start = wf_end - pd.DateOffset(months=held_out_months)
        held_out_start = dates[dates >= held_out_start][0]
        held_out = DataSplit("Held-out (SEALED)", held_out_start, wf_end)

        # Walk-forward windows between wf_start and held_out_start
        wf_windows: list[DataSplit] = []
        current = wf_start
        i = 1
        while current < held_out_start:
            window_end = current + pd.DateOffset(months=window_months) - timedelta(days=1)
            window_end = min(window_end, held_out_start - timedelta(days=1))
            window_end_date = dates[dates <= window_end]
            if len(window_end_date) == 0:
                break
            window_end_date = window_end_date[-1]
            wf_windows.append(DataSplit(f"Walk-forward {i}", current, window_end_date))
            next_start = dates[dates > window_end_date]
            if len(next_start) == 0:
                break
            current = next_start[0]
            i += 1

        self._schedule = WalkForwardSchedule(
            training=training,
            validation=validation,
            walk_forward_windows=wf_windows,
            held_out=held_out,
        )
        return self._schedule

    def get_split_data(
        self, data: pd.DataFrame, split: DataSplit
    ) -> pd.DataFrame:
        """Slice data for a given split."""
        return data.loc[split.start: split.end]

    def expanding_train_data(
        self, data: pd.DataFrame, test_window: DataSplit
    ) -> pd.DataFrame:
        """Return all data strictly before the test window start (expanding window)."""
        return data.loc[: test_window.start - timedelta(days=1)]
