from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class CohortKey(str, Enum):
    FIRST_ORDER = "first_order"
    REGISTER = "register"


class CohortGranularity(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class EventType(str, Enum):
    ORDER = "order"
    LOGIN = "login"
    ACTIVE = "active"
    PAY = "pay"


@dataclass
class CohortConfig:
    cohort_key: CohortKey = CohortKey.FIRST_ORDER
    granularity: CohortGranularity = CohortGranularity.DAY
    retention_days: int = 30
    event_type: EventType = EventType.ORDER
    user_id_col: str = "user_id"
    event_time_col: str = "event_time"
    event_type_col: Optional[str] = None
    amount_col: Optional[str] = None
    filter_segment: Optional[str] = None
    min_users: int = 1

    def validate(self) -> List[str]:
        errors = []
        if self.retention_days <= 0:
            errors.append("retention_days 必须大于 0")
        if self.min_users < 1:
            errors.append("min_users 必须至少为 1")
        return errors


@dataclass
class DataSourceConfig:
    source_type: str
    path: str
    table: Optional[str] = None
    encoding: str = "utf-8"

    def validate(self) -> List[str]:
        errors = []
        if self.source_type not in ("csv", "sqlite"):
            errors.append(f"不支持的数据源类型: {self.source_type}")
        if not self.path:
            errors.append("数据源路径不能为空")
        if self.source_type == "sqlite" and not self.table:
            errors.append("SQLite 数据源必须指定 table")
        return errors


@dataclass
class ReportConfig:
    output_dir: str = "./reports"
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    matrix_format: str = "csv"

    def validate(self) -> List[str]:
        errors = []
        valid_formats = {"markdown", "json"}
        for fmt in self.formats:
            if fmt not in valid_formats:
                errors.append(f"不支持的报告格式: {fmt}")
        if self.matrix_format not in ("csv", "json"):
            errors.append(f"不支持的矩阵格式: {self.matrix_format}")
        return errors
