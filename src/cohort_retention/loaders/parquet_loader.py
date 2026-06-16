from typing import Optional, Dict, Any
import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig, ParquetConfig


class ParquetDataLoader(BaseDataLoader):
    def __init__(
        self,
        source_config: DataSourceConfig,
        cohort_config: CohortConfig,
        parquet_config: Optional[ParquetConfig] = None,
    ):
        super().__init__(source_config, cohort_config)
        self.parquet_config = parquet_config or ParquetConfig()

    def _prepare_for_spark(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if self.parquet_config.coerce_timestamps == "ms":
                    df[col] = df[col].astype("datetime64[ms]")
                elif self.parquet_config.coerce_timestamps == "us":
                    df[col] = df[col].astype("datetime64[us]")

            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = df[col].astype(str)
                except Exception:
                    pass

        return df

    def _get_write_kwargs(self, **overrides) -> Dict[str, Any]:
        kwargs = {
            "compression": self.parquet_config.compression,
            "use_deprecated_int96_timestamps": self.parquet_config.use_deprecated_int96_timestamps,
            "coerce_timestamps": self.parquet_config.coerce_timestamps,
        }
        if self.parquet_config.row_group_size:
            kwargs["row_group_size"] = self.parquet_config.row_group_size
        if self.parquet_config.partition_cols:
            kwargs["partition_cols"] = self.parquet_config.partition_cols
        kwargs.update(overrides)
        return kwargs

    def load_events(self, columns: Optional[list] = None, filters: Optional[list] = None) -> pd.DataFrame:
        df = pd.read_parquet(
            self.source_config.path,
            columns=columns,
            filters=filters,
        )
        if self.cohort_config.event_time_col in df.columns:
            df[self.cohort_config.event_time_col] = pd.to_datetime(
                df[self.cohort_config.event_time_col]
            )
        return df

    def write_events(
        self,
        df: pd.DataFrame,
        spark_compatible: Optional[bool] = None,
        **kwargs
    ) -> None:
        use_spark = spark_compatible if spark_compatible is not None else self.parquet_config.spark_compatible
        if use_spark:
            df = self._prepare_for_spark(df)
        write_kwargs = self._get_write_kwargs(**kwargs)
        df.to_parquet(self.source_config.path, **write_kwargs)

    def write_partitioned(
        self,
        df: pd.DataFrame,
        partition_cols: Optional[list] = None,
        spark_compatible: Optional[bool] = None,
        **kwargs
    ) -> None:
        cols = partition_cols or self.parquet_config.partition_cols
        if not cols:
            raise ValueError("分区写入需要指定 partition_cols")

        use_spark = spark_compatible if spark_compatible is not None else self.parquet_config.spark_compatible
        if use_spark:
            df = self._prepare_for_spark(df)

        write_kwargs = self._get_write_kwargs(partition_cols=cols, **kwargs)
        df.to_parquet(self.source_config.path, **write_kwargs)
