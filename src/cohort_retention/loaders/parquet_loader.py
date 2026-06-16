import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig


class ParquetDataLoader(BaseDataLoader):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        super().__init__(source_config, cohort_config)

    def load_events(self) -> pd.DataFrame:
        df = pd.read_parquet(self.source_config.path)
        df[self.cohort_config.event_time_col] = pd.to_datetime(
            df[self.cohort_config.event_time_col]
        )
        return df

    def write_events(self, df: pd.DataFrame, **kwargs) -> None:
        df.to_parquet(self.source_config.path, **kwargs)
