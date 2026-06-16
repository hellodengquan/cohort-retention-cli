import pandas as pd
import numpy as np
from datetime import date
from typing import Optional, Dict, Any, List

from .config import CohortConfig, CohortKey, CohortGranularity
from .models import RetentionMatrix, FunnelResult, FunnelStep, CohortAnalysisResult
from .loaders import BaseDataLoader


class CohortAnalyzer:
    def __init__(self, loader: BaseDataLoader, config: CohortConfig):
        self.loader = loader
        self.config = config

    def _truncate_date(self, dt: pd.Series) -> pd.Series:
        if self.config.granularity == CohortGranularity.DAY:
            return dt.dt.floor("D")
        elif self.config.granularity == CohortGranularity.WEEK:
            return dt.dt.to_period("W").dt.to_timestamp()
        elif self.config.granularity == CohortGranularity.MONTH:
            return dt.dt.to_period("M").dt.to_timestamp()
        return dt

    def _cohort_to_date(self, cohort_period) -> date:
        if hasattr(cohort_period, "date"):
            return cohort_period.date()
        return pd.Timestamp(cohort_period).date()

    def build_cohort_base(self, events_df: pd.DataFrame) -> pd.DataFrame:
        user_col = self.config.user_id_col
        time_col = self.config.event_time_col

        first_events = self.loader.get_first_event_df(events_df)
        first_events["cohort_period"] = self._truncate_date(first_events["cohort_time"])

        base = events_df.copy()
        base[time_col] = pd.to_datetime(base[time_col])

        if self.config.event_type_col and self.config.event_type:
            base = base[base[self.config.event_type_col] == self.config.event_type]

        base = base.merge(
            first_events[[user_col, "cohort_period"]],
            on=user_col,
            how="inner",
        )

        base["event_period"] = self._truncate_date(base[time_col])
        base["days_since_cohort"] = (base["event_period"] - base["cohort_period"]).dt.days

        return base

    def compute_retention_matrix(self, events_df: pd.DataFrame) -> RetentionMatrix:
        user_col = self.config.user_id_col
        base = self.build_cohort_base(events_df)

        cohort_groups = base.groupby("cohort_period")
        cohort_sizes = cohort_groups[user_col].nunique()
        cohort_sizes = cohort_sizes[cohort_sizes >= self.config.min_users]

        max_days = self.config.retention_days
        retention_counts = pd.DataFrame(
            index=sorted(cohort_sizes.index),
            columns=range(0, max_days + 1),
            dtype=float,
        )
        retention_rates = pd.DataFrame(
            index=sorted(cohort_sizes.index),
            columns=range(0, max_days + 1),
            dtype=float,
        )

        for cohort_period in sorted(cohort_sizes.index):
            cohort_data = base[base["cohort_period"] == cohort_period]
            cohort_size = cohort_sizes[cohort_period]

            for day in range(0, max_days + 1):
                day_users = cohort_data[
                    (cohort_data["days_since_cohort"] >= day)
                    & (cohort_data["days_since_cohort"] < day + 1)
                    ][user_col].nunique()
                retention_counts.loc[cohort_period, day] = day_users
                retention_rates.loc[cohort_period, day] = day_users / cohort_size if cohort_size > 0 else 0.0

        cohort_dates = [self._cohort_to_date(d) for d in retention_counts.index]
        days_list = list(range(max_days + 1))

        cohort_sizes_dict = {
            self._cohort_to_date(k): int(v) for k, v in cohort_sizes.items()
        }

        return RetentionMatrix(
            cohort_dates=cohort_dates,
            days=days_list,
            retention_counts=retention_counts,
            retention_rates=retention_rates,
            cohort_sizes=cohort_sizes_dict,
        )

    def compute_funnel(self, events_df: pd.DataFrame) -> FunnelResult:
        user_col = self.config.user_id_col
        time_col = self.config.event_time_col
        max_days = self.config.retention_days

        base = self.build_cohort_base(events_df)

        total_users = base[user_col].nunique()

        steps = []
        prev_count = total_users

        day_milestones = [0, 1, 3, 7, 14, 30]
        day_milestones = [d for d in day_milestones if d <= max_days]

        for day in day_milestones:
            users_at_day = base[base["days_since_cohort"] >= day][user_col].nunique()
            conv_rate = users_at_day / total_users if total_users > 0 else 0.0
            drop_off = 1.0 - (users_at_day / prev_count) if prev_count > 0 else 0.0

            steps.append(FunnelStep(
                name=f"第 {day} 天留存",
                user_count=users_at_day,
                conversion_rate=conv_rate,
                drop_off_rate=drop_off,
            ))
            prev_count = users_at_day

        return FunnelResult(steps=steps, total_users=total_users)

    def run_analysis(self) -> CohortAnalysisResult:
        events_df = self.loader.load_events()

        matrix = self.compute_retention_matrix(events_df)
        funnel = self.compute_funnel(events_df)

        config_dict = {
            "cohort_key": self.config.cohort_key.value,
            "granularity": self.config.granularity.value,
            "retention_days": self.config.retention_days,
            "event_type": self.config.event_type.value,
            "min_users": self.config.min_users,
        }

        avg_retention = {}
        for day in matrix.days:
            rates = matrix.retention_rates[day].dropna()
            if len(rates) > 0:
                avg_retention[f"day_{day}"] = float(rates.mean())

        summary = {
            "total_cohorts": len(matrix.cohort_dates),
            "total_users": funnel.total_users,
            "average_retention": avg_retention,
        }

        return CohortAnalysisResult(
            matrix=matrix,
            funnel=funnel,
            config=config_dict,
            summary=summary,
        )
