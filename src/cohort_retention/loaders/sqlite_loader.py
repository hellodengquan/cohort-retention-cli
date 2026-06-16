import sqlite3
import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig


class SqliteDataLoader(BaseDataLoader):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        super().__init__(source_config, cohort_config)

    def load_events(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.source_config.path)
        try:
            query = f"SELECT * FROM {self.source_config.table}"
            df = pd.read_sql(query, conn)
            df[self.cohort_config.event_time_col] = pd.to_datetime(
                df[self.cohort_config.event_time_col]
            )
            return df
        finally:
            conn.close()
