import pandas as pd
import numpy as np
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

from .config import (
    CohortConfig,
    CohortKey,
    CohortGranularity,
    EventType,
    OutlierMethod,
    RetentionType,
)
from .models import RetentionMatrix, FunnelResult, FunnelStep, CohortAnalysisResult
from .loaders import BaseDataLoader
from .dsl import CohortDSLEngine


class OutlierRemover:
    def __init__(self, config: "OutlierConfig"):
        self.config = config

    def _detect_iqr(self, values: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - self.config.threshold * iqr
        upper = q3 + self.config.threshold * iqr
        return (values < lower) | (values > upper)

    def _detect_zscore(self, values: np.ndarray) -> np.ndarray:
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values, dtype=bool)
        zscores = np.abs((values - mean) / std)
        return zscores > self.config.threshold

    def _detect_percentile(self, values: np.ndarray) -> np.ndarray:
        lower, upper = np.percentile(values, [self.config.percentile_low, self.config.percentile_high])
        return (values < lower) | (values > upper)

    def _detect_mad(self, values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.zeros_like(values, dtype=bool)
        modified_zscores = 0.6745 * np.abs(values - median) / mad
        return modified_zscores > self.config.mad_threshold

    def _detect_isolation_forest(self, values: np.ndarray) -> np.ndarray:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "Isolation Forest 需要 scikit-learn，请运行: pip install scikit-learn"
            )

        X = values.reshape(-1, 1)
        iso = IsolationForest(
            contamination=self.config.contamination,
            random_state=42,
        )
        preds = iso.fit_predict(X)
        return preds == -1

    def _detect_dbscan(self, values: np.ndarray) -> np.ndarray:
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError(
                "DBSCAN 需要 scikit-learn，请运行: pip install scikit-learn"
            )

        X = values.reshape(-1, 1)
        dbscan = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
        )
        labels = dbscan.fit_predict(X)
        return labels == -1

    def remove_outliers(
        self,
        df: pd.DataFrame,
        value_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.config.enabled:
            return df, {"removed": 0, "method": None}

        target_col = value_col or self.config.target_column
        if not target_col:
            raise ValueError("异常值检测需要指定 value_col 或 target_column")

        values = df[target_col].values

        if self.config.method == OutlierMethod.IQR:
            mask = self._detect_iqr(values)
        elif self.config.method == OutlierMethod.ZSCORE:
            mask = self._detect_zscore(values)
        elif self.config.method == OutlierMethod.PERCENTILE:
            mask = self._detect_percentile(values)
        elif self.config.method == OutlierMethod.MAD:
            mask = self._detect_mad(values)
        elif self.config.method == OutlierMethod.ISOLATION_FOREST:
            mask = self._detect_isolation_forest(values)
        elif self.config.method == OutlierMethod.DBSCAN:
            mask = self._detect_dbscan(values)
        else:
            raise ValueError(f"未知的异常值检测方法: {self.config.method}")

        removed_count = int(mask.sum())
        cleaned_df = df[~mask].copy()

        stats = {
            "removed": removed_count,
            "method": self.config.method.value,
            "threshold": self.config.threshold,
            "original_count": len(df),
            "remaining_count": len(cleaned_df),
            "target_column": target_col,
        }

        if self.config.method == OutlierMethod.MAD:
            stats["mad_threshold"] = self.config.mad_threshold
        elif self.config.method == OutlierMethod.PERCENTILE:
            stats["percentile_low"] = self.config.percentile_low
            stats["percentile_high"] = self.config.percentile_high
        elif self.config.method == OutlierMethod.ISOLATION_FOREST:
            stats["contamination"] = self.config.contamination
        elif self.config.method == OutlierMethod.DBSCAN:
            stats["eps"] = self.config.eps
            stats["min_samples"] = self.config.min_samples

        return cleaned_df, stats


class CohortAnalyzer:
    def __init__(self, loader: BaseDataLoader, config: CohortConfig):
        self.loader = loader
        self.config = config
        self.dsl_engine = CohortDSLEngine(config.dsl) if config.dsl else None
        self.outlier_remover = OutlierRemover(config.outlier) if config.outlier else None
        self._outlier_stats: Optional[Dict[str, Any]] = None

    def _truncate_date(self, dt: pd.Series) -> pd.Series:
        if self.config.granularity == CohortGranularity.DAY:
            return dt.dt.floor("D")
        elif self.config.granularity == CohortGranularity.WEEK:
            return dt.dt.to_period("W").dt.to_timestamp()
        elif self.config.granularity == CohortGranularity.MONTH:
            return dt.dt.to_period("M").dt.to_timestamp()
        elif self.config.granularity == CohortGranularity.QUARTER:
            return dt.dt.to_period("Q").dt.to_timestamp()
        return dt

    def _cohort_to_date(self, cohort_period) -> date:
        if hasattr(cohort_period, "date"):
            return cohort_period.date()
        return pd.Timestamp(cohort_period).date()

    def _get_cohort_time_col(self) -> str:
        if self.config.cohort_key == CohortKey.FIRST_ORDER:
            return "cohort_time"
        elif self.config.cohort_key == CohortKey.REGISTER:
            return "register_time"
        else:
            return "cohort_time"

    def _apply_composite_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.event_type != EventType.COMPOSITE:
            return events_df

        if not self.config.composite_events:
            raise ValueError("复合事件类型需要定义 composite_events")

        df = events_df.copy()
        event_type_col = self.config.event_type_col or "event_type"

        if event_type_col not in df.columns:
            df[event_type_col] = self.config.event_type.value

        sorted_events = sorted(
            self.config.composite_events,
            key=lambda x: x.priority,
            reverse=True,
        )

        remaining_indices = set(df.index)
        all_comp_rows = []

        for comp_event in sorted_events:
            if not remaining_indices:
                break

            remaining_df = df.loc[list(remaining_indices)].copy()

            try:
                from .dsl import DSLParser
                mask = DSLParser.evaluate(comp_event.expression, {}, remaining_df)
                if isinstance(mask, (pd.Series, np.ndarray)) and mask.any():
                    matched_indices = remaining_df[mask].index
                    comp_rows = remaining_df.loc[matched_indices].copy()
                    comp_rows[event_type_col] = comp_event.name
                    all_comp_rows.append(comp_rows)

                    if comp_event.stop_on_match:
                        remaining_indices -= set(matched_indices)
            except Exception as e:
                raise ValueError(f"复合事件 {comp_event.name} 表达式计算失败: {e}")

        if all_comp_rows:
            result = pd.concat(all_comp_rows, ignore_index=True)
            result = result.drop_duplicates(
                subset=[self.config.user_id_col, self.config.event_time_col, event_type_col]
            )
            return result
        else:
            return df.iloc[0:0]

    def _apply_dsl_filter(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if not self.dsl_engine:
            return events_df
        return self.dsl_engine.apply_filter(events_df)

    def _get_first_event_dsl(self, events_df: pd.DataFrame) -> pd.DataFrame:
        user_col = self.config.user_id_col
        time_col = self.config.event_time_col

        if self.dsl_engine and self.config.cohort_key == CohortKey.CUSTOM:
            return self.dsl_engine.compute_cohort_time(events_df, user_col, time_col)
        else:
            first_events = self.loader.get_first_event_df(events_df)
            return first_events

    def _remove_outliers(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if not self.outlier_remover or not self.config.outlier.enabled:
            return events_df

        user_col = self.config.user_id_col

        if self.config.outlier.remove_users:
            user_event_counts = events_df.groupby(user_col).size().reset_index(name="event_count")
            user_event_counts, stats = self.outlier_remover.remove_outliers(user_event_counts, "event_count")
            valid_users = user_event_counts[user_col].unique()
            events_df = events_df[events_df[user_col].isin(valid_users)].copy()
            self._outlier_stats = stats

        return events_df

    def _remove_outlier_cohorts(self, cohort_sizes: pd.Series) -> pd.Series:
        if not self.outlier_remover or not self.config.outlier.enabled:
            return cohort_sizes

        if self.config.outlier.remove_cohorts:
            sizes_df = cohort_sizes.reset_index()
            sizes_df.columns = ["cohort_period", "size"]
            cleaned_df, stats = self.outlier_remover.remove_outliers(sizes_df, "size")
            if self._outlier_stats:
                self._outlier_stats["cohorts_removed"] = stats.get("removed", 0)
            else:
                self._outlier_stats = stats
            result = cleaned_df.set_index("cohort_period")["size"]
            return result

        return cohort_sizes

    def _compute_rolling_retention(self, base: pd.DataFrame, cohort_sizes: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        from .config import RollingAlignment

        user_col = self.config.user_id_col
        window_size = self.config.rolling.window_size
        step = self.config.rolling.step
        min_periods = self.config.rolling.min_periods
        max_days = self.config.retention_days
        alignment = self.config.rolling.alignment
        include_partial = self.config.rolling.include_partial

        rolling_days = list(range(0, max_days + 1, step))

        retention_counts = pd.DataFrame(
            index=sorted(cohort_sizes.index),
            columns=rolling_days,
            dtype=float,
        )
        retention_rates = pd.DataFrame(
            index=sorted(cohort_sizes.index),
            columns=rolling_days,
            dtype=float,
        )

        half_window = window_size // 2

        for cohort_period in sorted(cohort_sizes.index):
            cohort_data = base[base["cohort_period"] == cohort_period]
            cohort_size = cohort_sizes[cohort_period]

            for day in rolling_days:
                if alignment == RollingAlignment.LEFT:
                    window_start = day
                    window_end = day + window_size
                elif alignment == RollingAlignment.CENTER:
                    window_start = day - half_window
                    window_end = day + window_size - half_window
                elif alignment == RollingAlignment.RIGHT:
                    window_start = day - window_size + 1
                    window_end = day + 1
                else:
                    window_start = day
                    window_end = day + window_size

                actual_start = max(0, window_start)
                actual_end = min(max_days + 1, window_end)
                actual_window_size = actual_end - actual_start

                if not include_partial and actual_window_size < window_size:
                    window_users = np.nan
                else:
                    window_users = cohort_data[
                        (cohort_data["days_since_cohort"] >= actual_start)
                        & (cohort_data["days_since_cohort"] < actual_end)
                    ][user_col].nunique()

                    if actual_window_size < min_periods:
                        window_users = np.nan

                retention_counts.loc[cohort_period, day] = window_users
                retention_rates.loc[cohort_period, day] = window_users / cohort_size if cohort_size > 0 and pd.notna(window_users) else np.nan

        return retention_counts, retention_rates, rolling_days

    def build_cohort_base(self, events_df: pd.DataFrame) -> pd.DataFrame:
        user_col = self.config.user_id_col
        time_col = self.config.event_time_col

        df = self._apply_dsl_filter(events_df)
        df = self._remove_outliers(df)

        first_events = self._get_first_event_dsl(df)
        first_events["cohort_period"] = self._truncate_date(first_events["cohort_time"])

        df = self._apply_composite_events(df)

        if self.config.event_type_col and self.config.event_type:
            if self.config.event_type == EventType.COMPOSITE:
                composite_names = [ce.name for ce in (self.config.composite_events or [])]
                if composite_names:
                    df = df[df[self.config.event_type_col].isin(composite_names)]
            else:
                df = df[df[self.config.event_type_col] == self.config.event_type.value]

        base = df.copy()
        base[time_col] = pd.to_datetime(base[time_col])

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
        cohort_sizes = self._remove_outlier_cohorts(cohort_sizes)

        max_days = self.config.retention_days

        if self.config.retention_type == RetentionType.ROLLING and self.config.rolling.enabled:
            retention_counts, retention_rates, days_list = self._compute_rolling_retention(base, cohort_sizes)
        else:
            days_list = list(range(max_days + 1))
            retention_counts = pd.DataFrame(
                index=sorted(cohort_sizes.index),
                columns=days_list,
                dtype=float,
            )
            retention_rates = pd.DataFrame(
                index=sorted(cohort_sizes.index),
                columns=days_list,
                dtype=float,
            )

            for cohort_period in sorted(cohort_sizes.index):
                cohort_data = base[base["cohort_period"] == cohort_period]
                cohort_size = cohort_sizes[cohort_period]

                for day in days_list:
                    day_users = cohort_data[
                        (cohort_data["days_since_cohort"] >= day)
                        & (cohort_data["days_since_cohort"] < day + 1)
                        ][user_col].nunique()
                    retention_counts.loc[cohort_period, day] = day_users
                    retention_rates.loc[cohort_period, day] = day_users / cohort_size if cohort_size > 0 else 0.0

        cohort_dates = [self._cohort_to_date(d) for d in retention_counts.index]

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
        max_days = self.config.retention_days

        base = self.build_cohort_base(events_df)

        total_users = base[user_col].nunique()

        steps = []
        prev_count = total_users

        day_milestones = [0, 1, 3, 7, 14, 30, 60, 90]
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
            "retention_type": self.config.retention_type.value,
            "event_type": self.config.event_type.value,
            "min_users": self.config.min_users,
        }

        if self.config.retention_type == RetentionType.ROLLING:
            config_dict["rolling_window"] = {
                "window_size": self.config.rolling.window_size,
                "step": self.config.rolling.step,
                "min_periods": self.config.rolling.min_periods,
            }

        if self.config.outlier.enabled:
            config_dict["outlier"] = {
                "method": self.config.outlier.method.value,
                "threshold": self.config.outlier.threshold,
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

        if self._outlier_stats:
            summary["outlier_stats"] = self._outlier_stats

        return CohortAnalysisResult(
            matrix=matrix,
            funnel=funnel,
            config=config_dict,
            summary=summary,
        )
