from typing import Optional, Dict, Any, List
import os
import time
import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig, ParquetConfig, ParquetCodecMetrics


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

    def benchmark_codecs(
        self,
        df: pd.DataFrame,
        codecs: Optional[List[str]] = None,
        spark_compatible: bool = True,
        sample_rows: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[ParquetCodecMetrics]:
        import tempfile

        if codecs is None:
            codecs = ["snappy", "gzip", "zstd", "lz4", "brotli", "none"]

        test_df = df
        if sample_rows and sample_rows < len(df):
            test_df = df.sample(n=min(sample_rows, len(df)), random_state=42).copy()

        if spark_compatible:
            test_df = self._prepare_for_spark(test_df)

        metrics_list: List[ParquetCodecMetrics] = []
        valid_codecs = {"snappy", "gzip", "brotli", "lz4", "zstd", "none"}

        original_csv_bytes = len(test_df.to_csv(index=False).encode())

        with tempfile.TemporaryDirectory() as tmpdir:
            for codec in codecs:
                if codec not in valid_codecs:
                    continue
                m = ParquetCodecMetrics(compression=codec)
                tmp_path = os.path.join(tmpdir, f"test_{codec}.parquet")
                try:
                    kwargs = self._get_write_kwargs(compression=None if codec == "none" else codec)
                    kwargs.pop("partition_cols", None)
                    if kwargs.get("coerce_timestamps") is None:
                        pass
                    t0 = time.time()
                    test_df.to_parquet(tmp_path, **{k: v for k, v in kwargs.items() if v is not None})
                    m.write_time_ms = (time.time() - t0) * 1000
                    compressed_bytes = os.path.getsize(tmp_path)
                    m.compressed_bytes = compressed_bytes
                    m.original_bytes = int(original_csv_bytes)
                    if compressed_bytes > 0:
                        m.compression_ratio = original_csv_bytes / compressed_bytes
                    t1 = time.time()
                    _ = pd.read_parquet(tmp_path)
                    m.read_time_ms = (time.time() - t1) * 1000
                    m.row_count = len(test_df)
                    m.spark_compatible = spark_compatible
                    try:
                        pf = _get_parquet_file(tmp_path)
                        if pf and hasattr(pf, 'metadata'):
                            m.row_group_count = getattr(pf.metadata, 'num_row_groups', 1)
                        else:
                            m.row_group_count = max(1, len(test_df) // max(1, self.parquet_config.row_group_size or len(test_df)))
                    except Exception:
                        m.row_group_count = max(1, len(test_df) // max(1, self.parquet_config.row_group_size or len(test_df)))
                except ImportError as e:
                    m.note = f"压缩异常: {e}"
                    continue
                except Exception as e:
                    m.note = f"不支持: {e}"
                    continue
                metrics_list.append(m)

        metrics_list.sort(key=lambda x: x.compressed_bytes)
        self.parquet_config.codec_metrics = metrics_list
        return metrics_list


def _get_parquet_file(path: str):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path)
    except ImportError:
        try:
            import fastparquet
            return None
        except ImportError:
            return None
        return None
