"""
Multiple-testing bookkeeping.

The problem this solves
-----------------------
This study has many defensible specifications: several bar widths, two return
definitions, two session-boundary conventions, three variance-ratio horizons,
several placebo dates, two control assets. Running all of them is good practice.
Reporting only the ones that worked, without saying how many were run, is not.

``TestRegistry`` records every test as it is executed, which makes the
denominator explicit. The registry is what allows the report to state the number
of hypotheses examined rather than leaving a reader to guess.

A note on which correction to use
---------------------------------
The pre-registered primary test needs no correction: it is one hypothesis,
declared in advance. Corrections apply to the family of secondary and robustness
tests. Benjamini-Hochberg is the default there because it controls the false
discovery rate, which is the appropriate target for exploratory robustness
checks, whereas Bonferroni control of the family-wise error rate is
unnecessarily severe when dozens of related specifications are being scanned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def adjust_pvalues(
    pvalues: np.ndarray | pd.Series | list[float],
    method: str = "bh",
) -> np.ndarray:
    """Adjust p-values for multiplicity.

    Supported methods are ``"bh"`` (Benjamini-Hochberg false discovery rate),
    ``"holm"`` (step-down family-wise error rate) and ``"bonferroni"``.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    n = p.size
    method = method.lower()

    if method == "bonferroni":
        return np.minimum(p * n, 1.0)

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)

    if method == "bh":
        # Step-up, then enforce monotonicity from the largest p-value downward.
        factors = n / np.arange(1, n + 1)
        raw = ranked * factors
        adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    elif method == "holm":
        factors = n - np.arange(n)
        raw = ranked * factors
        adjusted_sorted = np.maximum.accumulate(raw)
    else:
        raise ValueError(f"unknown method {method!r}; use 'bh', 'holm' or 'bonferroni'")

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted[order] = adjusted_sorted
    return adjusted


@dataclass
class TestRecord:
    name: str
    family: str
    p_value: float
    statistic: float | None = None
    is_primary: bool = False
    n_obs: int | None = None
    direction: str = ""
    note: str = ""


@dataclass
class TestRegistry:
    """Append-only log of every statistical test performed.

    Keeping this honest is the point. Tests should be registered when they are
    run, not selected afterwards, so that the total count reflects what was
    actually examined.
    """

    # Prevents pytest from attempting to collect this as a test class.
    __test__ = False

    records: list[TestRecord] = field(default_factory=list)

    def register(
        self,
        name: str,
        family: str,
        p_value: float,
        statistic: float | None = None,
        is_primary: bool = False,
        n_obs: int | None = None,
        direction: str = "",
        note: str = "",
    ) -> None:
        if any(r.name == name for r in self.records):
            raise ValueError(
                f"a test named {name!r} is already registered; use a distinct name "
                "so that the count of tests performed stays accurate"
            )
        if not np.isfinite(p_value):
            raise ValueError(f"p-value for {name!r} is not finite")
        self.records.append(
            TestRecord(
                name=name,
                family=family,
                p_value=float(p_value),
                statistic=statistic,
                is_primary=is_primary,
                n_obs=n_obs,
                direction=direction,
                note=note,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_primary(self) -> int:
        return sum(1 for r in self.records if r.is_primary)

    def table(self, method: str = "bh", exclude_primary: bool = True) -> pd.DataFrame:
        """All registered tests with adjusted p-values.

        The primary test is excluded from the adjustment family by default,
        because it is a single pre-registered hypothesis and pooling it with
        exploratory checks would penalise it for their number.
        """
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "name",
                    "family",
                    "statistic",
                    "p_value",
                    "p_adjusted",
                    "is_primary",
                    "n_obs",
                    "direction",
                    "note",
                ]
            )

        df = pd.DataFrame([r.__dict__ for r in self.records])
        df["p_adjusted"] = np.nan

        mask = ~df["is_primary"] if exclude_primary else pd.Series(True, index=df.index)
        if mask.any():
            df.loc[mask, "p_adjusted"] = adjust_pvalues(
                df.loc[mask, "p_value"].to_numpy(), method=method
            )
        if exclude_primary:
            df.loc[df["is_primary"], "p_adjusted"] = df.loc[df["is_primary"], "p_value"]

        cols = [
            "name",
            "family",
            "statistic",
            "p_value",
            "p_adjusted",
            "is_primary",
            "n_obs",
            "direction",
            "note",
        ]
        return df[cols].sort_values(["is_primary", "p_value"], ascending=[False, True])

    def summary(self, alpha: float = 0.05, method: str = "bh") -> dict:
        table = self.table(method=method)
        secondary = table[~table["is_primary"]]
        return {
            "n_tests_total": len(table),
            "n_primary": int(table["is_primary"].sum()),
            "n_secondary": len(secondary),
            "n_secondary_significant_raw": int((secondary["p_value"] < alpha).sum()),
            "n_secondary_significant_adjusted": int(
                (secondary["p_adjusted"] < alpha).sum()
            ),
            "expected_false_positives_at_alpha": round(len(secondary) * alpha, 2),
            "adjustment_method": method,
        }

    def to_csv(self, path: str | Path, method: str = "bh") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table(method=method).to_csv(path, index=False)
        return path
