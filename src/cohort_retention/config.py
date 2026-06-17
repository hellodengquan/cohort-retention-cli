from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Tuple
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
    MAD = "mad"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    AUTO = "auto"


class DSLErrorCode(str, Enum):
    COLUMN_NOT_FOUND = "E001"
    TYPE_MISMATCH = "E002"
    DIVISION_BY_ZERO = "E003"
    NULL_VALUE = "E004"
    FUNCTION_ERROR = "E005"
    ARGUMENT_MISSING = "E006"
    PARSE_ERROR = "E007"
    UNKNOWN = "E999"


class RetentionType(str, Enum):
    STANDARD = "standard"
    ROLLING = "rolling"


class RollingAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class TimeZoneHandling(str, Enum):
    NAIVE = "naive"
    UTC = "utc"
    LOCAL = "local"
    PRESERVE = "preserve"


@dataclass
class TimezoneDBConfig:
    enabled: bool = False
    validate_tz_names: bool = True
    lookup_historical_offsets: bool = True
    ambiguous_time_strategy: str = "shift_forward"
    nonexistent_time_strategy: str = "shift_forward"
    cache_transitions: bool = True


@dataclass
class OutlierConfig:
    enabled: bool = False
    method: OutlierMethod = OutlierMethod.IQR
    threshold: float = 1.5
    remove_cohorts: bool = True
    remove_users: bool = False
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    mad_threshold: float = 3.0
    contamination: float = 0.05
    min_samples: int = 5
    eps: float = 0.5
    target_column: Optional[str] = None
    auto_min_rows: int = 1000
    auto_max_rows: int = 100000

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if self.threshold <= 0:
                errors.append("outlier threshold 必须大于 0")
            if self.percentile_low <= 0 or self.percentile_low >= 100:
                errors.append("outlier percentile_low 必须在 (0, 100) 之间")
            if self.percentile_high <= 0 or self.percentile_high >= 100:
                errors.append("outlier percentile_high 必须在 (0, 100) 之间")
            if self.percentile_low >= self.percentile_high:
                errors.append("outlier percentile_low 必须小于 percentile_high")
            if self.mad_threshold <= 0:
                errors.append("outlier mad_threshold 必须大于 0")
            if self.contamination <= 0 or self.contamination >= 1:
                errors.append("outlier contamination 必须在 (0, 1) 之间")
            if self.min_samples < 2:
                errors.append("outlier min_samples 必须至少为 2")
            if self.eps <= 0:
                errors.append("outlier eps 必须大于 0")
            if self.auto_min_rows < 1:
                errors.append("outlier auto_min_rows 必须大于 0")
            if self.auto_max_rows <= self.auto_min_rows:
                errors.append("outlier auto_max_rows 必须大于 auto_min_rows")
        return errors


@dataclass
class RollingWindowConfig:
    enabled: bool = False
    window_size: int = 7
    step: int = 1
    min_periods: int = 1
    alignment: RollingAlignment = RollingAlignment.LEFT
    include_partial: bool = False
    timezone_handling: TimeZoneHandling = TimeZoneHandling.NAIVE
    target_timezone: Optional[str] = None
    handle_dst: bool = True
    timezone_db: TimezoneDBConfig = field(default_factory=TimezoneDBConfig)

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if self.window_size <= 0:
                errors.append("rolling window_size 必须大于 0")
            if self.step <= 0:
                errors.append("rolling step 必须大于 0")
            if self.min_periods < 1:
                errors.append("rolling min_periods 必须至少为 1")
            if self.min_periods > self.window_size:
                errors.append("rolling min_periods 不能大于 window_size")
            if self.timezone_handling == TimeZoneHandling.LOCAL and not self.target_timezone:
                errors.append("使用 local 时区模式时必须指定 target_timezone")
        return errors


@dataclass
class CompositeEventConfig:
    name: str
    expression: str
    description: Optional[str] = None
    priority: int = 0
    stop_on_match: bool = False


@dataclass
class CompositeEventSubGraph:
    nodes: List[str]
    edges: List[Tuple[str, str]]
    is_cycle: bool = False
    description: str = ""

    def to_dot(self) -> str:
        lines = ["digraph composite_events {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=filled];')
        for node in self.nodes:
            color = "#ff6b6b" if self.is_cycle else "#74b9ff"
            lines.append(f'  "{node}" [fillcolor="{color}"];')
        for src, dst in self.edges:
            style = " [color=red, penwidth=2]" if self.is_cycle else ""
            lines.append(f'  "{src}" -> "{dst}"{style};')
        lines.append("}")
        return "\n".join(lines)


def _tarjan_scc(adj: Dict[str, List[str]]) -> List[List[str]]:
    index_counter = [0]
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[List[str]] = []

    def _strongconnect(v: str):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in index:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for v in adj:
        if v not in index:
            _strongconnect(v)

    return result


@dataclass
class ConflictInfo:
    message: str
    severity: str = "error"
    param_paths: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None
    config_section: Optional[str] = None

    def __str__(self) -> str:
        location = ""
        if self.config_section:
            location = f"[{self.config_section}] "
        paths = ""
        if self.param_paths:
            paths = f" (参数: {', '.join(self.param_paths)})"
        sugg = f" → 建议: {self.suggestion}" if self.suggestion else ""
        return f"{location}{self.message}{paths}{sugg}"


@dataclass
class CohortDSLConfig:
    cohort_expression: Optional[str] = None
    filter_expression: Optional[str] = None
    segment_expression: Optional[str] = None
    error_recovery: bool = True
    coalesce_on_error: bool = True
    default_numeric: float = 0.0
    default_string: str = ""
    default_datetime: Optional[str] = None
    error_code_enabled: bool = True
    fail_on_critical: bool = True
    retry_on_recoverable: bool = False
    retry_max_attempts: int = 1
    strict_mode: bool = False

    def validate(self) -> List[str]:
        errors = []
        if self.retry_max_attempts < 1:
            errors.append("dsl retry_max_attempts 必须至少为 1")
        return errors


@dataclass
class SEOSettings:
    title: str = "留存队列分析报告"
    description: str = "基于 cohort-retention-cli 生成的用户留存分析报告，包含留存矩阵、漏斗分析、滚动留存等关键指标"
    keywords: List[str] = field(default_factory=lambda: ["留存分析", "用户留存", "cohort analysis", "用户行为分析", "数据分析"])
    author: str = "cohort-retention-cli"
    og_type: str = "article"
    og_image: Optional[str] = None
    enable_structured_data: bool = True
    language: str = "zh-CN"
    canonical_url: Optional[str] = None
    robots: str = "noindex,nofollow"


@dataclass
class PostgresTransactionConfig:
    isolation_level: str = "read_committed"
    autocommit: bool = False
    enable_savepoints: bool = True
    rollback_on_error: bool = True
    nested_transaction_limit: int = 5
    statement_timeout_ms: Optional[int] = None
    lock_timeout_ms: Optional[int] = None
    retry_on_deadlock: bool = True
    deadlock_retries: int = 3
    retry_delay_ms: int = 100


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
    detect_priority_cycle: bool = True
    resolve_priority_tie_by_name: bool = True
    transaction: PostgresTransactionConfig = field(default_factory=PostgresTransactionConfig)

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
        errors.extend(self._validate_conflicts())
        if self.detect_priority_cycle:
            errors.extend(self._validate_composite_priorities())
        return errors

    def validate_with_warnings(self) -> Tuple[List[str], List[str]]:
        hard_errors = []
        for e in self.dsl.validate():
            hard_errors.append(e)
        for e in self.outlier.validate():
            hard_errors.append(e)
        for e in self.rolling.validate():
            hard_errors.append(e)

        conflict_infos = self._collect_all_conflicts()
        hard_errors.extend(str(c) for c in conflict_infos if c.severity == "error")
        warnings_list = [str(c) for c in conflict_infos if c.severity == "warning"]

        if self.detect_priority_cycle:
            for e in self._validate_composite_priorities():
                hard_errors.append(e)

        return hard_errors, warnings_list

    def _collect_all_conflicts(self) -> List[ConflictInfo]:
        conflicts: List[ConflictInfo] = []
        if self.retention_type == RetentionType.ROLLING and not self.rolling.enabled:
            conflicts.append(ConflictInfo(
                message="retention_type=rolling 需要启用 rolling_enabled",
                param_paths=["cohort.retention_type", "rolling.enabled"],
                suggestion="设置 --rolling-enabled 或改 --retention-type standard",
                config_section="cohort",
            ))
        if self.outlier.enabled and self.outlier.method == OutlierMethod.AUTO:
            if self.outlier.target_column is None:
                conflicts.append(ConflictInfo(
                    message="outlier method=auto 需要指定 target_column",
                    param_paths=["outlier.method", "outlier.target_column"],
                    suggestion="设置 --outlier-target-col <列名>",
                    config_section="outlier",
                ))
        if self.rolling.enabled and self.rolling.step > self.rolling.window_size:
            conflicts.append(ConflictInfo(
                message="rolling.step 不能大于 rolling.window_size",
                param_paths=["rolling.step", "rolling.window_size"],
                suggestion=f"当前 step={self.rolling.step}, window={self.rolling.window_size}，请调整",
                config_section="rolling",
            ))
        if self.cohort_key == CohortKey.CUSTOM and not self.dsl.cohort_expression:
            conflicts.append(ConflictInfo(
                message="cohort_key=custom 需要设置 dsl.cohort_expression",
                param_paths=["cohort.cohort_key", "dsl.cohort_expression"],
                suggestion="设置 --dsl-cohort <表达式>",
                config_section="dsl",
            ))
        if self.dsl.retry_on_recoverable and not self.dsl.error_recovery:
            conflicts.append(ConflictInfo(
                message="retry_on_recoverable=True 需要 error_recovery=True",
                param_paths=["dsl.retry_on_recoverable", "dsl.error_recovery"],
                suggestion="启用 --dsl-error-recovery",
                config_section="dsl",
            ))
        if self.event_type == EventType.COMPOSITE:
            names = [e.name for e in self.composite_events]
            if len(names) != len(set(names)):
                conflicts.append(ConflictInfo(
                    message="复合事件名称不能重复",
                    param_paths=["cohort.composite_events"],
                    config_section="cohort",
                ))

        warnings: List[ConflictInfo] = []
        if self.rolling.enabled and self.rolling.window_size > self.retention_days:
            warnings.append(ConflictInfo(
                message=f"滚动窗口({self.rolling.window_size}天) 大于留存天数({self.retention_days}天)",
                param_paths=["rolling.window_size", "cohort.retention_days"],
                suggestion="缩小窗口或延长留存期",
                severity="warning",
                config_section="rolling",
            ))
        if self.event_type == EventType.COMPOSITE and self.event_type_col is not None:
            warnings.append(ConflictInfo(
                message="使用复合事件时 event_type_col 将被忽略",
                param_paths=["cohort.event_type_col", "cohort.event_type"],
                severity="warning",
                config_section="cohort",
            ))
        if self.outlier.method == OutlierMethod.DBSCAN and self.outlier.remove_users:
            warnings.append(ConflictInfo(
                message="DBSCAN + remove_users 可能过度剔除",
                param_paths=["outlier.method", "outlier.remove_users"],
                severity="warning",
                config_section="outlier",
            ))
        if self.dsl.strict_mode and self.dsl.error_recovery:
            warnings.append(ConflictInfo(
                message="strict_mode 将覆盖 error_recovery=True",
                param_paths=["dsl.strict_mode", "dsl.error_recovery"],
                severity="warning",
                config_section="dsl",
            ))
        if self.dsl.coalesce_on_error and not self.dsl.error_recovery:
            warnings.append(ConflictInfo(
                message="coalesce_on_error=True 但 error_recovery=False",
                param_paths=["dsl.coalesce_on_error", "dsl.error_recovery"],
                severity="warning",
                config_section="dsl",
            ))
        if self.outlier.enabled and (self.outlier.remove_users or self.outlier.remove_cohorts):
            if self.min_users < 5:
                warnings.append(ConflictInfo(
                    message="启用异常值剔除时建议 min_users >= 5",
                    param_paths=["outlier.enabled", "cohort.min_users"],
                    severity="warning",
                    config_section="cohort",
                ))
        if self.granularity in (CohortGranularity.MONTH, CohortGranularity.QUARTER) and self.retention_days < 90:
            warnings.append(ConflictInfo(
                message=f"粒度={self.granularity.value} 建议 retention_days >= 90",
                param_paths=["cohort.granularity", "cohort.retention_days"],
                severity="warning",
                config_section="cohort",
            ))
        if self.rolling.alignment == RollingAlignment.CENTER and self.rolling.window_size % 2 == 0:
            warnings.append(ConflictInfo(
                message="CENTER 对齐建议使用奇数窗口大小",
                param_paths=["rolling.alignment", "rolling.window_size"],
                severity="warning",
                config_section="rolling",
            ))
        if self.outlier.method in (OutlierMethod.ISOLATION_FOREST, OutlierMethod.DBSCAN):
            if self.outlier.auto_min_rows < 500:
                warnings.append(ConflictInfo(
                    message="机器学习类异常检测建议样本量 >= 500",
                    param_paths=["outlier.method", "outlier.auto_min_rows"],
                    severity="warning",
                    config_section="outlier",
                ))

        return conflicts + warnings

    def _validate_composite_priorities(self) -> List[str]:
        errors = []
        if not self.composite_events:
            return errors

        name_to_idx = {e.name: i for i, e in enumerate(self.composite_events)}

        for i, ev in enumerate(self.composite_events):
            for j, other in enumerate(self.composite_events):
                if i >= j:
                    continue
                if ev.priority == other.priority and ev.stop_on_match and other.stop_on_match:
                    if self.resolve_priority_tie_by_name:
                        continue
                    errors.append(
                        f"复合事件优先级冲突: '{ev.name}' 和 '{other.name}' "
                        f"优先级相同 ({ev.priority}) 且都启用 stop_on_match，"
                        f"按名称排序: '{min(ev.name, other.name)}' 优先"
                    )

        stop_events = [e for e in self.composite_events if e.stop_on_match]
        if len(stop_events) >= 2:
            sorted_by_prio = sorted(stop_events, key=lambda e: (-e.priority, e.name))
            adj: Dict[str, List[str]] = {e.name: [] for e in stop_events}
            for a_idx in range(len(sorted_by_prio)):
                for b_idx in range(a_idx + 1, len(sorted_by_prio)):
                    a = sorted_by_prio[a_idx]
                    b = sorted_by_prio[b_idx]
                    if a.priority >= b.priority:
                        adj[a.name].append(b.name)

            sccs = _tarjan_scc(adj)
            cycle_sccs = [c for c in sccs if len(c) > 1]
            for scc in cycle_sccs:
                cycle_edges = []
                for node in scc:
                    for nb in adj.get(node, []):
                        if nb in scc:
                            cycle_edges.append((node, nb))
                subgraph = CompositeEventSubGraph(
                    nodes=list(scc),
                    edges=cycle_edges,
                    is_cycle=True,
                    description=f"优先级环: {' -> '.join(scc)} -> {scc[0]}"
                )
                errors.append(
                    f"检测到 stop_on_match 优先级环: {' -> '.join(scc)} -> {scc[0]}，"
                    f"子图 DOT:\n{subgraph.to_dot()}，"
                    f"请调整优先级或部分事件禁用 stop_on_match"
                )

            non_cycle_sccs = [c for c in sccs if len(c) == 1]
            for scc in non_cycle_sccs:
                node = scc[0]
                out_edges = [(node, nb) for nb in adj.get(node, [])]
                if not out_edges:
                    continue
                subgraph = CompositeEventSubGraph(
                    nodes=[node] + [nb for _, nb in out_edges],
                    edges=out_edges,
                    is_cycle=False,
                    description=f"事件 '{node}' 依赖: {[nb for _, nb in out_edges]}"
                )

        return errors

    def _validate_conflicts(self) -> List[str]:
        all_infos = self._collect_all_conflicts()
        return [str(c) for c in all_infos if c.severity == "error"]


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
    transaction_config: PostgresTransactionConfig = field(default_factory=PostgresTransactionConfig)

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
            valid_iso = {"read_uncommitted", "read_committed", "repeatable_read", "serializable"}
            if self.transaction_config.isolation_level not in valid_iso:
                errors.append(f"不支持的隔离级别: {self.transaction_config.isolation_level}，支持: {valid_iso}")
            if self.transaction_config.deadlock_retries < 0:
                errors.append("deadlock_retries 不能为负数")
        return errors


@dataclass
class ParquetCodecMetrics:
    compression: str = ""
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    write_time_ms: float = 0.0
    read_time_ms: float = 0.0
    row_count: int = 0
    row_group_count: int = 0
    spark_compatible: bool = True
    note: Optional[str] = None
    write_throughput_rows_per_sec: float = 0.0
    read_throughput_rows_per_sec: float = 0.0
    overall_score: float = 0.0
    best_for: Optional[str] = None

    def compute_derived_metrics(self) -> None:
        if self.write_time_ms > 0 and self.row_count > 0:
            self.write_throughput_rows_per_sec = self.row_count / (self.write_time_ms / 1000.0)
        if self.read_time_ms > 0 and self.row_count > 0:
            self.read_throughput_rows_per_sec = self.row_count / (self.read_time_ms / 1000.0)
        speed_score = 0.0
        if self.write_time_ms > 0 and self.read_time_ms > 0:
            total_ms = self.write_time_ms + self.read_time_ms
            speed_score = 1.0 / (total_ms / 1000.0)
        compression_score = self.compression_ratio if self.compression_ratio > 0 else 0
        self.overall_score = round(0.4 * speed_score + 0.4 * compression_score + 0.2 * (1.0 if self.spark_compatible else 0.0), 4)
        if self.overall_score > 0:
            if self.compression_ratio >= 3.0 and self.read_time_ms <= 50:
                self.best_for = "归档存储"
            elif self.write_throughput_rows_per_sec > 10000:
                self.best_for = "实时写入"
            elif self.read_throughput_rows_per_sec > 10000:
                self.best_for = "实时查询"
            elif self.spark_compatible:
                self.best_for = "Spark 兼容"
            else:
                self.best_for = "通用"

    def to_dict(self) -> dict:
        self.compute_derived_metrics()
        return {
            "compression": self.compression,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": round(self.compression_ratio, 4),
            "write_time_ms": round(self.write_time_ms, 2),
            "read_time_ms": round(self.read_time_ms, 2),
            "row_count": self.row_count,
            "row_group_count": self.row_group_count,
            "spark_compatible": self.spark_compatible,
            "note": self.note,
            "write_throughput_rows_per_sec": round(self.write_throughput_rows_per_sec, 2),
            "read_throughput_rows_per_sec": round(self.read_throughput_rows_per_sec, 2),
            "overall_score": self.overall_score,
            "best_for": self.best_for,
        }


@dataclass
class ParquetConfig:
    spark_compatible: bool = True
    compression: str = "snappy"
    row_group_size: Optional[int] = None
    partition_cols: List[str] = field(default_factory=list)
    use_deprecated_int96_timestamps: bool = False
    coerce_timestamps: Optional[str] = "us"
    benchmark_codecs: bool = False
    codec_metrics: List[ParquetCodecMetrics] = field(default_factory=list)

    def validate(self) -> List[str]:
        errors = []
        valid_compressions = {"snappy", "gzip", "brotli", "lz4", "zstd", "none"}
        if self.compression not in valid_compressions:
            errors.append(f"不支持的 Parquet 压缩格式: {self.compression}，支持: {valid_compressions}")
        valid_coerce = {"us", "ms", None}
        if self.coerce_timestamps not in valid_coerce:
            errors.append(f"不支持的时间戳精度: {self.coerce_timestamps}，支持: {valid_coerce}")
        return errors


@dataclass
class ReportConfig:
    output_dir: str = "./reports"
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    matrix_format: str = "csv"
    ssr_enabled: bool = True
    ssr_render_charts: bool = True
    ssr_embed_data: bool = True
    ssr_minify: bool = False
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    seo: SEOSettings = field(default_factory=SEOSettings)
    detect_conflicts: bool = True
    enable_conflict_suggestions: bool = True

    def validate(self) -> List[str]:
        errors = []
        valid_formats = {"markdown", "json", "html"}
        for fmt in self.formats:
            if fmt not in valid_formats:
                errors.append(f"不支持的报告格式: {fmt}，支持: {valid_formats}")
        valid_matrix = {"csv", "json", "parquet"}
        if self.matrix_format not in valid_matrix:
            errors.append(f"不支持的矩阵格式: {self.matrix_format}，支持: {valid_matrix}")
        errors.extend(self.parquet.validate())
        if self.detect_conflicts:
            errors.extend(self._validate_report_conflicts())
        return errors

    def _validate_report_conflicts(self) -> List[str]:
        conflicts: List[ConflictInfo] = []

        if "html" not in self.formats and self.ssr_enabled:
            conflicts.append(ConflictInfo(
                message="ssr_enabled=True 但未选择 html 格式",
                param_paths=["report.formats", "report.ssr_enabled"],
                suggestion="添加 --format html",
                severity="warning",
                config_section="report",
            ))
        if "html" not in self.formats and self.seo.title != "留存队列分析报告":
            conflicts.append(ConflictInfo(
                message="自定义 SEO 设置需要 html 格式才会生效",
                param_paths=["report.formats", "seo.title"],
                severity="warning",
                config_section="report",
            ))
        if self.parquet.benchmark_codecs and self.matrix_format != "parquet":
            conflicts.append(ConflictInfo(
                message="benchmark_codecs 需要 matrix_format=parquet",
                param_paths=["parquet.benchmark_codecs", "report.matrix_format"],
                suggestion="设置 --matrix-format parquet",
                severity="warning",
                config_section="parquet",
            ))
        if self.ssr_minify and not self.ssr_enabled:
            conflicts.append(ConflictInfo(
                message="ssr_minify=True 需要 ssr_enabled=True",
                param_paths=["report.ssr_minify", "report.ssr_enabled"],
                suggestion="启用 --ssr-enabled",
                config_section="report",
            ))
        if self.ssr_render_charts and not self.ssr_enabled:
            conflicts.append(ConflictInfo(
                message="ssr_render_charts=True 但 ssr_enabled=False",
                param_paths=["report.ssr_render_charts", "report.ssr_enabled"],
                severity="warning",
                config_section="report",
            ))
        if self.ssr_embed_data and not self.ssr_enabled:
            conflicts.append(ConflictInfo(
                message="ssr_embed_data=True 但 ssr_enabled=False",
                param_paths=["report.ssr_embed_data", "report.ssr_enabled"],
                severity="warning",
                config_section="report",
            ))
        if self.enable_conflict_suggestions and not self.detect_conflicts:
            conflicts.append(ConflictInfo(
                message="enable_conflict_suggestions=True 但 detect_conflicts=False",
                param_paths=["report.enable_conflict_suggestions", "report.detect_conflicts"],
                severity="warning",
                config_section="report",
            ))
        if self.parquet.spark_compatible and self.parquet.compression == "brotli":
            conflicts.append(ConflictInfo(
                message="Spark 2.x 对 brotli 压缩支持有限",
                param_paths=["parquet.spark_compatible", "parquet.compression"],
                suggestion="建议用 snappy/zstd",
                severity="warning",
                config_section="parquet",
            ))
        if self.parquet.coerce_timestamps is None and self.parquet.spark_compatible:
            conflicts.append(ConflictInfo(
                message="Spark 兼容模式建议设置 coerce_timestamps='us' 或 'ms'",
                param_paths=["parquet.coerce_timestamps", "parquet.spark_compatible"],
                severity="warning",
                config_section="parquet",
            ))

        return [str(c) for c in conflicts if c.severity == "error"]
