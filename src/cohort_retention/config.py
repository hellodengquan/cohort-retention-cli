from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum


class CohortKey(str, Enum):
    FIRST_ORDER = "first_order"
    REGISTER = "register"
    CUSTOM = "custom"


class CohortGranularity(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"


class EventType(str, Enum):
    ORDER = "order"
    LOGIN = "login"
    ACTIVE = "active"
    PAY = "pay"
    COMPOSITE = "composite"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"


class RetentionType(str, Enum):
    STANDARD = "standard"
    ROLLING = "rolling"


@dataclass
class OutlierConfig:
    enabled: bool = False
    method: OutlierMethod = OutlierMethod.IQR
    threshold: float = 1.5
    remove_cohorts: bool = True
    remove_users: bool = False

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if self.threshold <= 0:
                errors.append("outlier threshold 必须大于 0")
        return errors


@dataclass
class RollingWindowConfig:
    enabled: bool = False
    window_size: int = 7
    step: int = 1
    min_periods: int = 1

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if self.window_size <= 0:
                errors.append("rolling window_size 必须大于 0")
            if self.step <= 0:
                errors.append("rolling step 必须大于 0")
            if self.min_periods < 1:
                errors.append("rolling min_periods 必须至少为 1")
        return errors


@dataclass
class CompositeEventConfig:
    name: str
    expression: str
    description: Optional[str] = None


@dataclass
class CohortDSLConfig:
    cohort_expression: Optional[str] = None
    filter_expression: Optional[str] = None
    segment_expression: Optional[str] = None

    def validate(self) -> List[str]:
        errors = []
        return errors


@dataclass
class CohortConfig:
    cohort_key: CohortKey = CohortKey.FIRST_ORDER
    granularity: CohortGranularity = CohortGranularity.DAY
    retention_days: int = 30
    retention_type: RetentionType = RetentionType.STANDARD
    event_type: EventType = EventType.ORDER
    composite_events: List[CompositeEventConfig] = field(default_factory=list)
    user_id_col: str = "user_id"
    event_time_col: str = "event_time"
    event_type_col: Optional[str] = None
    amount_col: Optional[str] = None
    filter_segment: Optional[str] = None
    min_users: int = 1
    dsl: CohortDSLConfig = field(default_factory=CohortDSLConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    rolling: RollingWindowConfig = field(default_factory=RollingWindowConfig)

    def validate(self) -> List[str]:
        errors = []
        if self.retention_days <= 0:
            errors.append("retention_days 必须大于 0")
        if self.min_users < 1:
            errors.append("min_users 必须至少为 1")
        if self.event_type == EventType.COMPOSITE and not self.composite_events:
            errors.append("使用复合事件类型时必须定义 composite_events")
        errors.extend(self.dsl.validate())
        errors.extend(self.outlier.validate())
        errors.extend(self.rolling.validate())
        return errors


@dataclass
class DataSourceConfig:
    source_type: str
    path: str
    table: Optional[str] = None
    encoding: str = "utf-8"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    def validate(self) -> List[str]:
        errors = []
        valid_types = {"csv", "sqlite", "parquet", "postgresql"}
        if self.source_type not in valid_types:
            errors.append(f"不支持的数据源类型: {self.source_type}，支持: {valid_types}")
        if not self.path and self.source_type not in ("postgresql",):
            errors.append("数据源路径不能为空")
        if self.source_type in ("sqlite", "postgresql") and not self.table:
            errors.append(f"{self.source_type} 数据源必须指定 table")
        if self.source_type == "postgresql":
            if not self.host:
                errors.append("PostgreSQL 必须指定 host")
            if not self.database:
                errors.append("PostgreSQL 必须指定 database")
        return errors


@dataclass
class ReportConfig:
    output_dir: str = "./reports"
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    matrix_format: str = "csv"

    def validate(self) -> List[str]:
        errors = []
        valid_formats = {"markdown", "json", "html"}
        for fmt in self.formats:
            if fmt not in valid_formats:
                errors.append(f"不支持的报告格式: {fmt}，支持: {valid_formats}")
        valid_matrix = {"csv", "json", "parquet"}
        if self.matrix_format not in valid_matrix:
            errors.append(f"不支持的矩阵格式: {self.matrix_format}，支持: {valid_matrix}")
        return errors
