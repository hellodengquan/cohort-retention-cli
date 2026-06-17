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

            color: Dict[str, int] = {e.name: 0 for e in stop_events}
            cycle_found = None

            def _dfs(node: str, path: List[str]) -> Optional[List[str]]:
                color[node] = 1
                for nb in adj.get(node, []):
                    if color[nb] == 1:
                        idx = path.index(nb) if nb in path else -1
                        if idx >= 0:
                            return path[idx:] + [nb]
                        return [nb, node]
                    elif color[nb] == 0:
                        res = _dfs(nb, path + [nb])
                        if res:
                            return res
                color[node] = 2
                return None

            for ev in stop_events:
                if color[ev.name] == 0:
                    cycle = _dfs(ev.name, [ev.name])
                    if cycle:
                        cycle_found = cycle
                        break

            if cycle_found:
                errors.append(
                    f"检测到 stop_on_match 优先级环: {' -> '.join(cycle_found)}，"
                    f"请调整优先级或部分事件禁用 stop_on_match"
                )

        return errors

    def _validate_conflicts(self) -> List[str]:
        conflicts = []
        warnings = []

        if self.retention_type == RetentionType.ROLLING and not self.rolling.enabled:
            conflicts.append("retention_type=rolling 需要启用 rolling_enabled")
        if self.outlier.enabled and self.outlier.method == OutlierMethod.AUTO:
            if self.outlier.target_column is None:
                conflicts.append("outlier method=auto 需要指定 target_column")
        if self.rolling.enabled and self.rolling.window_size > self.retention_days:
            warnings.append(
                f"滚动窗口({self.rolling.window_size}天) 大于留存天数({self.retention_days}天)，"
                f"建议缩小窗口或延长留存期"
            )
        if self.event_type == EventType.COMPOSITE and self.event_type_col is not None:
            warnings.append("使用复合事件时 event_type_col 将被忽略，改用表达式定义")
        if self.outlier.method == OutlierMethod.DBSCAN and self.outlier.remove_users:
            warnings.append("DBSCAN + remove_users 可能过度剔除，建议谨慎使用")
        if self.dsl.strict_mode and self.dsl.error_recovery:
            warnings.append("strict_mode 将覆盖 error_recovery=True 的行为，不做错误恢复")
        if self.dsl.retry_on_recoverable and not self.dsl.error_recovery:
            conflicts.append("retry_on_recoverable=True 需要 error_recovery=True")
        if self.outlier.enabled and (self.outlier.remove_users or self.outlier.remove_cohorts):
            if self.min_users < 5:
                warnings.append("启用异常值剔除时建议 min_users >= 5 以避免队列过度减少")
        if self.cohort_key == CohortKey.CUSTOM and not self.dsl.cohort_expression:
            conflicts.append("cohort_key=custom 需要设置 dsl.cohort_expression")
        if self.granularity in (CohortGranularity.MONTH, CohortGranularity.QUARTER) and self.retention_days < 90:
            warnings.append(f"粒度={self.granularity.value} 建议 retention_days >= 90 以获得有意义的结果")
        if self.rolling.enabled and self.rolling.step > self.rolling.window_size:
            conflicts.append("rolling.step 不能大于 rolling.window_size")
        if self.outlier.method == OutlierMethod.ISOLATION_FOREST and self.outlier.min_samples >= 2:
            pass
        if self.event_type == EventType.COMPOSITE:
            names = [e.name for e in self.composite_events]
            if len(names) != len(set(names)):
                conflicts.append("复合事件名称不能重复")
            for e in self.composite_events:
                if e.stop_on_match and self.min_users < 2:
                    warnings.append(f"复合事件 '{e.name}' 启用 stop_on_match 建议 min_users >= 2")
        if self.dsl.coalesce_on_error and not self.dsl.error_recovery:
            warnings.append("coalesce_on_error=True 但 error_recovery=False，coalesce 将不生效")
        if self.rolling.alignment == RollingAlignment.CENTER and self.rolling.window_size % 2 == 0:
            warnings.append("CENTER 对齐建议使用奇数窗口大小以获得对称窗口")
        if self.outlier.method in (OutlierMethod.ISOLATION_FOREST, OutlierMethod.DBSCAN):
            if self.outlier.auto_min_rows < 500:
                warnings.append("机器学习类异常检测建议样本量 >= 500")

        return conflicts


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

    def to_dict(self) -> dict:
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
        conflicts = []
        warnings = []
        valid_formats = {"markdown", "json", "html"}
        if "html" not in self.formats and self.ssr_enabled:
            warnings.append("ssr_enabled=True 但未选择 html 格式，SSR 将不生效")
        if "html" not in self.formats and self.seo.title != "留存队列分析报告":
            warnings.append("自定义 SEO 设置需要 html 格式才会生效")
        if self.matrix_format == "parquet":
            pass
        if self.parquet.benchmark_codecs and self.matrix_format != "parquet":
            warnings.append("parquet benchmark_codecs 需要 matrix_format=parquet 才会运行基准测试")
        if self.ssr_minify and not self.ssr_enabled:
            conflicts.append("ssr_minify=True 需要 ssr_enabled=True")
        if self.ssr_render_charts and not self.ssr_enabled:
            warnings.append("ssr_render_charts=True 但 ssr_enabled=False，图表渲染将被跳过")
        if self.ssr_embed_data and not self.ssr_enabled:
            warnings.append("ssr_embed_data=True 但 ssr_enabled=False，数据嵌入将被跳过")
        if self.enable_conflict_suggestions and not self.detect_conflicts:
            warnings.append("enable_conflict_suggestions=True 但 detect_conflicts=False，无法生成建议")
        if self.parquet.spark_compatible and self.parquet.compression == "brotli":
            warnings.append("Spark 2.x 对 brotli 压缩支持有限，建议用 snappy/zstd")
        if self.parquet.coerce_timestamps is None and self.parquet.spark_compatible:
            warnings.append("Spark 兼容模式建议设置 coerce_timestamps='us' 或 'ms'")
        return conflicts + warnings
