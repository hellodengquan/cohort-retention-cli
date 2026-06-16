from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd


@dataclass
class CohortResult:
    cohort_date: date
    cohort_size: int
    day_0: int
    day_n: Dict[int, int] = field(default_factory=dict)
    day_n_rate: Dict[int, float] = field(default_factory=dict)


@dataclass
class RetentionMatrix:
    cohort_dates: List[date]
    days: List[int]
    retention_counts: pd.DataFrame
    retention_rates: pd.DataFrame
    cohort_sizes: Dict[date, int]

    def to_dict(self) -> Dict[str, Any]:
        counts_df = self.retention_counts.copy()
        counts_df.index = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in counts_df.index]
        counts_df.columns = [f"day_{d}" for d in self.days]

        rates_df = self.retention_rates.copy()
        rates_df.index = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in rates_df.index]
        rates_df.columns = [f"day_{d}" for d in self.days]

        return {
            "cohort_dates": [d.isoformat() for d in self.cohort_dates],
            "days": self.days,
            "retention_counts": counts_df.to_dict(),
            "retention_rates": rates_df.to_dict(),
            "cohort_sizes": {k.isoformat(): v for k, v in self.cohort_sizes.items()},
        }


@dataclass
class FunnelStep:
    name: str
    user_count: int
    conversion_rate: float
    drop_off_rate: float


@dataclass
class FunnelResult:
    steps: List[FunnelStep]
    total_users: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_users": self.total_users,
            "steps": [
                {
                    "name": s.name,
                    "user_count": s.user_count,
                    "conversion_rate": s.conversion_rate,
                    "drop_off_rate": s.drop_off_rate,
                }
                for s in self.steps
            ],
        }


@dataclass
class CohortAnalysisResult:
    matrix: RetentionMatrix
    funnel: Optional[FunnelResult] = None
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": self.matrix.to_dict(),
            "funnel": self.funnel.to_dict() if self.funnel else None,
            "config": self.config,
            "summary": self.summary,
        }
