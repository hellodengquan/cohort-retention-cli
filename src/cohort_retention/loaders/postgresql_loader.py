from typing import Optional, List, Any, Iterator, ContextManager
from contextlib import contextmanager
import pandas as pd
from .base import BaseDataLoader
from ..config import DataSourceConfig, CohortConfig


def _get_psycopg():
    try:
        import psycopg2
        from psycopg2.extensions import isolation_level
        return psycopg2, isolation_level
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


ISOLATION_LEVELS = {
    "auto_commit": 0,
    "read_uncommitted": 1,
    "read_committed": 2,
    "repeatable_read": 3,
    "serializable": 4,
}


class PostgreSqlDataLoader(BaseDataLoader):
    def __init__(self, source_config: DataSourceConfig, cohort_config: CohortConfig):
        super().__init__(source_config, cohort_config)
        self._psycopg2, self._isolation_levels = _get_psycopg()
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

    def _get_connection(self, isolation_level: Optional[str] = None):
        conn_str = self._build_connection_string(for_sqlalchemy=False)
        conn = self._psycopg2.connect(conn_str)
        if isolation_level and isolation_level in ISOLATION_LEVELS:
            conn.set_isolation_level(ISOLATION_LEVELS[isolation_level])
        return conn

    def _get_table_name(self) -> str:
        cfg = self.source_config
        if cfg.schema:
            return f"{cfg.schema}.{cfg.table}"
        return cfg.table or "events"

    @contextmanager
    def transaction(
        self,
        isolation_level: str = "read_committed",
        autocommit: bool = False,
    ) -> Iterator[Any]:
        conn = self._get_connection(isolation_level=isolation_level)
        conn.autocommit = autocommit
        try:
            yield conn
            if not autocommit:
                conn.commit()
        except Exception:
            if not autocommit:
                conn.rollback()
            raise
        finally:
            conn.close()

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

    def write_events(
        self,
        df: pd.DataFrame,
        if_exists: str = "replace",
        use_transaction: bool = True,
        chunk_size: Optional[int] = None,
        isolation_level: str = "read_committed",
        **kwargs
    ) -> None:
        engine = self._create_engine(self._build_connection_string(for_sqlalchemy=True))
        table = self.source_config.table or "events"
        schema = self.source_config.schema

        if use_transaction:
            with self.transaction(isolation_level=isolation_level):
                df.to_sql(
                    table,
                    engine,
                    schema=schema,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunk_size,
                    **kwargs
                )
        else:
            df.to_sql(
                table,
                engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                chunksize=chunk_size,
                **kwargs
            )

    def batch_write(
        self,
        dfs: List[pd.DataFrame],
        if_exists: str = "append",
        chunk_size: Optional[int] = 10000,
        isolation_level: str = "read_committed",
    ) -> List[int]:
        row_counts = []
        with self.transaction(isolation_level=isolation_level):
            for i, df in enumerate(dfs):
                rows_written = 0
                for chunk_start in range(0, len(df), chunk_size or len(df)):
                    chunk = df.iloc[chunk_start:chunk_start + (chunk_size or len(df))]
                    if i == 0 and chunk_start == 0:
                        self.write_events(
                            chunk,
                            if_exists=if_exists,
                            use_transaction=False,
                            chunk_size=chunk_size,
                        )
                    else:
                        self.write_events(
                            chunk,
                            if_exists="append",
                            use_transaction=False,
                            chunk_size=chunk_size,
                        )
                    rows_written += len(chunk)
                row_counts.append(rows_written)
        return row_counts

    def execute_query(self, query: str, use_transaction: bool = True) -> pd.DataFrame:
        if use_transaction:
            with self.transaction() as conn:
                return pd.read_sql(query, conn)
        else:
            conn = self._get_connection()
            try:
                return pd.read_sql(query, conn)
            finally:
                conn.close()

    def execute_non_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        use_transaction: bool = True,
    ) -> int:
        if use_transaction:
            with self.transaction() as conn:
                cur = conn.cursor()
                cur.execute(sql, params or ())
                return cur.rowcount
        else:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(sql, params or ())
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def execute_many(
        self,
        sql: str,
        params_list: List[tuple],
        chunk_size: int = 1000,
        isolation_level: str = "read_committed",
    ) -> int:
        total_rows = 0
        with self.transaction(isolation_level=isolation_level) as conn:
            cur = conn.cursor()
            for i in range(0, len(params_list), chunk_size):
                chunk = params_list[i:i + chunk_size]
                cur.executemany(sql, chunk)
                total_rows += len(chunk)
        return total_rows

    def copy_from_csv(
        self,
        csv_path: str,
        table: Optional[str] = None,
        columns: Optional[List[str]] = None,
        sep: str = ",",
        null: str = "",
        header: bool = True,
        isolation_level: str = "read_committed",
    ) -> int:
        import csv

        table_name = table or self._get_table_name()
        with self.transaction(isolation_level=isolation_level) as conn:
            cur = conn.cursor()
            with open(csv_path, 'r') as f:
                if header:
                    next(f)
                cur.copy_from(
                    f,
                    table_name,
                    sep=sep,
                    null=null,
                    columns=columns,
                )
                return cur.rowcount
