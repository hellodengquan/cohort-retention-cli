from abc import ABC, abstractmethod
import pandas as pd
from ..config import DataSourceConfig, CohortConfig


class BaseDataLoader(ABC):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        self.source_config = source_config
        self.cohort_config = cohort_config

    @abstractmethod
    def load_events(self) -> pd.DataFrame:
        pass

    def load_users(self) -> pd.DataFrame:
        return self.load_events()

    def get_first_event_df(self, events_df: pd.DataFrame) -> pd.DataFrame:
        user_col = self.cohort_config.user_id_col
        time_col = self.cohort_config.event_time_col
        event_type_col = self.cohort_config.event_type_col
        event_type = self.cohort_config.event_type

        df = events_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        if event_type_col and event_type:
            df = df[df[event_type_col] == event_type]

        first_events = df.sort_values(time_col).groupby(user_col).first().reset_index()
        return first_events[[user_col, time_col]].rename(columns={time_col: "cohort_time"})
