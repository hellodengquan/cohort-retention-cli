from typing import Optional
import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig


def _get_psycopg():
    try:
        import psycopg2
        return psycopg2
    except ImportError:
        raise ImportError(
            "PostgreSQL 支持需要 psycopg2，请运行: pip install psycopg2-binary"
        )


def _get_sqlalchemy():
    try:
        from sqlalchemy import create_engine
        return create_engine
    except ImportError:
        raise ImportError(
            "PostgreSQL 写入支持需要 SQLAlchemy，请运行: pip install sqlalchemy"
        )


class PostgreSqlDataLoader(BaseDataLoader):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        super().__init__(source_config, cohort_config)
        self._psycopg2 = _get_psycopg()
        self._create_engine = _get_sqlalchemy()

    def _build_connection_string(self, for_sqlalchemy: bool = False) -> str:
        cfg = self.source_config
        user = cfg.username or ""
        password = cfg.password or ""
        auth = f"{user}:{password}@" if user else ""
        host = cfg.host or "localhost"
        port = cfg.port or 5432
        db = cfg.database or ""

        if for_sqlalchemy:
            return f"postgresql+psycopg2://{auth}{host}:{port}/{db}"
        return f"host={host} port={port} dbname={db} user={user} password={password}"

    def _get_connection(self):
        conn_str = self._build_connection_string(for_sqlalchemy=False)
        return self._psycopg2.connect(conn_str)

    def _get_table_name(self) -> str:
        cfg = self.source_config
        if cfg.schema:
            return f"{cfg.schema}.{cfg.table}"
        return cfg.table or "events"

    def load_events(self) -> pd.DataFrame:
        table = self._get_table_name()
        query = f"SELECT * FROM {table}"

        conn = self._get_connection()
        try:
            df = pd.read_sql(query, conn)
            df[self.cohort_config.event_time_col] = pd.to_datetime(
                df[self.cohort_config.event_time_col]
            )
            return df
        finally:
            conn.close()

    def write_events(self, df: pd.DataFrame, if_exists: str = "replace", **kwargs) -> None:
        engine = self._create_engine(self._build_connection_string(for_sqlalchemy=True))
        table = self.source_config.table or "events"
        schema = self.source_config.schema
        df.to_sql(
            table,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            **kwargs
        )

    def execute_query(self, query: str) -> pd.DataFrame:
        conn = self._get_connection()
        try:
            return pd.read_sql(query, conn)
        finally:
            conn.close()

    def execute_non_query(self, sql: str) -> int:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
