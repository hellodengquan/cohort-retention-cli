from .base import BaseDataLoader
from .csv_loader import CsvDataLoader
from .sqlite_loader import SqliteDataLoader
from .parquet_loader import ParquetDataLoader
from .postgresql_loader import PostgreSqlDataLoader
from ..config import DataSourceConfig, CohortConfig


def create_loader(source_config: DataSourceConfig, cohort_config: CohortConfig) -> BaseDataLoader:
    errors = source_config.validate() + cohort_config.validate()
    if errors:
        raise ValueError(f"配置错误: {'; '.join(errors)}")

    if source_config.source_type == "csv":
        return CsvDataLoader(source_config, cohort_config)
    elif source_config.source_type == "sqlite":
        return SqliteDataLoader(source_config, cohort_config)
    elif source_config.source_type == "parquet":
        return ParquetDataLoader(source_config, cohort_config)
    elif source_config.source_type == "postgresql":
        return PostgreSqlDataLoader(source_config, cohort_config)
    else:
        raise ValueError(f"不支持的数据源类型: {source_config.source_type}")


__all__ = [
    "BaseDataLoader",
    "CsvDataLoader",
    "SqliteDataLoader",
    "ParquetDataLoader",
    "PostgreSqlDataLoader",
    "create_loader",
]
