import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig


class CsvDataLoader(BaseDataLoader):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        super().__init__(source_config, cohort_config)

    def load_events(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.source_config.path,
            encoding=self.source_config.encoding,
            parse_dates=[self.cohort_config.event_time_col],
        )
        return df
