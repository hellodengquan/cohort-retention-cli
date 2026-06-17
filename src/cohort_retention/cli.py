import click
import sys
import os
import json
from typing import List, Tuple

from .config import (
    CohortConfig,
    DataSourceConfig,
    ReportConfig,
    CohortKey,
    CohortGranularity,
    EventType,
    RetentionType,
    OutlierMethod,
    CohortDSLConfig,
    OutlierConfig,
    RollingWindowConfig,
    CompositeEventConfig,
    RollingAlignment,
    TimeZoneHandling,
    SEOSettings,
    PostgresTransactionConfig,
    ParquetConfig,
    ParquetCodecMetrics,
)
from .loaders import create_loader
from .analyzer import CohortAnalyzer
from .reporters import MarkdownReporter, JsonReporter, HtmlReporter, MatrixExporter
from .dsl import CohortDSLEngine, DSLError


def _parse_composite_events(events_str: Tuple[str, ...]) -> List[CompositeEventConfig]:
    events = []
    for ev_str in events_str:
        parts = ev_str.split(":")
        if len(parts) < 2:
            raise click.BadParameter(f"复合事件格式错误: {ev_str}，应为 'name:expression[:priority[:stop]]'")
        name = parts[0].strip()
        expr = parts[1].strip()
        priority = 0
        stop_on_match = False
        if len(parts) >= 3:
            try:
                priority = int(parts[2].strip())
            except ValueError:
                raise click.BadParameter(f"优先级必须是整数: {parts[2]}")
        if len(parts) >= 4:
            stop_on_match = parts[3].strip().lower() in ("true", "1", "yes", "stop")
        events.append(CompositeEventConfig(
            name=name,
            expression=expr,
            priority=priority,
            stop_on_match=stop_on_match,
        ))
    return events


def _build_cohort_config(
    cohort_key: str,
    granularity: str,
    retention_days: int,
    retention_type: str,
    event_type: str,
    composite_events: Tuple[str, ...],
    user_id_col: str,
    event_time_col: str,
    event_type_col: str,
    min_users: int,
    dsl_cohort: str,
    dsl_filter: str,
    dsl_segment: str,
    dsl_error_recovery: bool,
    dsl_default_numeric: float,
    dsl_default_string: str,
    dsl_coalesce_on_error: bool,
    dsl_strict_mode: bool,
    dsl_fail_on_critical: bool,
    outlier_enabled: bool,
    outlier_method: str,
    outlier_threshold: float,
    outlier_percentile_low: float,
    outlier_percentile_high: float,
    outlier_mad_threshold: float,
    outlier_contamination: float,
    outlier_eps: float,
    outlier_min_samples: int,
    outlier_remove_users: bool,
    outlier_remove_cohorts: bool,
    outlier_target_col: str,
    outlier_auto_min_rows: int,
    outlier_auto_max_rows: int,
    rolling_enabled: bool,
    rolling_window: int,
    rolling_step: int,
    rolling_min_periods: int,
    rolling_alignment: str,
    rolling_include_partial: bool,
    rolling_timezone_handling: str,
    rolling_target_timezone: str,
    rolling_handle_dst: bool,
) -> CohortConfig:
    dsl_config = CohortDSLConfig(
        cohort_expression=dsl_cohort if dsl_cohort else None,
        filter_expression=dsl_filter if dsl_filter else None,
        segment_expression=dsl_segment if dsl_segment else None,
        error_recovery=dsl_error_recovery,
        default_numeric=dsl_default_numeric,
        default_string=dsl_default_string,
        coalesce_on_error=dsl_coalesce_on_error,
        strict_mode=dsl_strict_mode,
        fail_on_critical=dsl_fail_on_critical,
    )

    outlier_config = OutlierConfig(
        enabled=outlier_enabled,
        method=OutlierMethod(outlier_method),
        threshold=outlier_threshold,
        percentile_low=outlier_percentile_low,
        percentile_high=outlier_percentile_high,
        mad_threshold=outlier_mad_threshold,
        contamination=outlier_contamination,
        eps=outlier_eps,
        min_samples=outlier_min_samples,
        remove_users=outlier_remove_users,
        remove_cohorts=outlier_remove_cohorts,
        target_column=outlier_target_col if outlier_target_col else None,
        auto_min_rows=outlier_auto_min_rows,
        auto_max_rows=outlier_auto_max_rows,
    )

    rolling_config = RollingWindowConfig(
        enabled=rolling_enabled,
        window_size=rolling_window,
        step=rolling_step,
        min_periods=rolling_min_periods,
        alignment=RollingAlignment(rolling_alignment),
        include_partial=rolling_include_partial,
        timezone_handling=TimeZoneHandling(rolling_timezone_handling),
        target_timezone=rolling_target_timezone if rolling_target_timezone else None,
        handle_dst=rolling_handle_dst,
    )

    composite_configs = _parse_composite_events(composite_events) if composite_events else []

    return CohortConfig(
        cohort_key=CohortKey(cohort_key),
        granularity=CohortGranularity(granularity),
        retention_days=retention_days,
        retention_type=RetentionType(retention_type),
        event_type=EventType(event_type),
        composite_events=composite_configs,
        user_id_col=user_id_col,
        event_time_col=event_time_col,
        event_type_col=event_type_col if event_type_col else None,
        min_users=min_users,
        dsl=dsl_config,
        outlier=outlier_config,
        rolling=rolling_config,
    )


def _build_source_config(
    source_type: str,
    path: str,
    table: str,
    encoding: str,
    host: str,
    port: int,
    username: str,
    password: str,
    database: str,
    schema: str,
    pg_isolation_level: str,
    pg_autocommit: bool,
    pg_enable_savepoints: bool,
    pg_nested_transaction_limit: int,
    pg_retry_on_deadlock: bool,
    pg_deadlock_retries: int,
    pg_statement_timeout_ms: int,
    pg_lock_timeout_ms: int,
) -> DataSourceConfig:
    transaction_config = PostgresTransactionConfig(
        isolation_level=pg_isolation_level,
        autocommit=pg_autocommit,
        enable_savepoints=pg_enable_savepoints,
        nested_transaction_limit=pg_nested_transaction_limit,
        retry_on_deadlock=pg_retry_on_deadlock,
        deadlock_retries=pg_deadlock_retries,
        statement_timeout_ms=pg_statement_timeout_ms if pg_statement_timeout_ms > 0 else None,
        lock_timeout_ms=pg_lock_timeout_ms if pg_lock_timeout_ms > 0 else None,
    )
    return DataSourceConfig(
        source_type=source_type,
        path=path if path else "",
        table=table if table else None,
        encoding=encoding,
        host=host if host else None,
        port=port if port else None,
        username=username if username else None,
        password=password if password else None,
        database=database if database else None,
        schema=schema if schema else None,
        transaction_config=transaction_config,
    )


def _build_report_config(
    output_dir: str,
    formats: tuple,
    matrix_format: str,
    ssr_enabled: bool = True,
    ssr_render_charts: bool = True,
    ssr_embed_data: bool = True,
    ssr_minify: bool = False,
    parquet_spark_compatible: bool = True,
    parquet_compression: str = "snappy",
    parquet_benchmark_codecs: bool = False,
    parquet_coerce_timestamps: str = "us",
    seo_title: str = "留存队列分析报告",
    seo_description: str = "基于 cohort-retention-cli 生成的用户留存分析报告，包含留存矩阵、漏斗分析、滚动留存等关键指标",
    seo_keywords: str = "",
    seo_author: str = "cohort-retention-cli",
    seo_language: str = "zh-CN",
    seo_og_type: str = "article",
    seo_robots: str = "noindex,nofollow",
    seo_enable_structured_data: bool = True,
    detect_conflicts: bool = True,
    enable_conflict_suggestions: bool = True,
) -> ReportConfig:
    kw_list = [k.strip() for k in seo_keywords.split(",") if k.strip()]
    if not kw_list:
        kw_list = ["留存分析", "用户留存", "cohort analysis", "用户行为分析", "数据分析"]

    seo = SEOSettings(
        title=seo_title,
        description=seo_description,
        keywords=kw_list,
        author=seo_author,
        language=seo_language,
        og_type=seo_og_type,
        robots=seo_robots,
        enable_structured_data=seo_enable_structured_data,
    )

    coerce_val = None if parquet_coerce_timestamps in ("none", "None", "") else parquet_coerce_timestamps

    parquet_cfg = ParquetConfig(
        spark_compatible=parquet_spark_compatible,
        compression=parquet_compression,
        benchmark_codecs=parquet_benchmark_codecs,
        coerce_timestamps=coerce_val,
    )

    return ReportConfig(
        output_dir=output_dir,
        formats=list(formats),
        matrix_format=matrix_format,
        ssr_enabled=ssr_enabled,
        ssr_render_charts=ssr_render_charts,
        ssr_embed_data=ssr_embed_data,
        ssr_minify=ssr_minify,
        parquet=parquet_cfg,
        seo=seo,
        detect_conflicts=detect_conflicts,
        enable_conflict_suggestions=enable_conflict_suggestions,
    )


@click.group()
@click.version_option(version="0.2.0", prog_name="cohort")
def cli():
    """留存队列分析 CLI - 按用户首单或注册时间分组，统计后续 N 天回流比例与漏斗"""
    pass


@cli.command()
@click.option("--source-type", type=click.Choice(["csv", "sqlite", "parquet", "postgresql"]), required=True, help="数据源类型")
@click.option("--path", default=None, help="数据源文件路径")
@click.option("--table", default=None, help="表名（SQLite/PostgreSQL 需要）")
@click.option("--encoding", default="utf-8", help="CSV 文件编码")
@click.option("--host", default=None, help="PostgreSQL 主机")
@click.option("--port", default=5432, type=int, help="PostgreSQL 端口")
@click.option("--username", default=None, help="PostgreSQL 用户名")
@click.option("--password", default=None, help="PostgreSQL 密码")
@click.option("--database", default=None, help="PostgreSQL 数据库名")
@click.option("--schema", default=None, help="PostgreSQL schema")
@click.option("--pg-isolation-level", type=click.Choice(["read_uncommitted", "read_committed", "repeatable_read", "serializable"]), default="read_committed", help="PostgreSQL 隔离级别")
@click.option("--pg-autocommit/--no-pg-autocommit", default=False, help="PostgreSQL 自动提交")
@click.option("--pg-enable-savepoints/--no-pg-enable-savepoints", default=True, help="PostgreSQL 启用 savepoint")
@click.option("--pg-nested-transaction-limit", default=5, type=int, help="嵌套事务层数限制")
@click.option("--pg-retry-on-deadlock/--no-pg-retry-on-deadlock", default=True, help="死锁时自动重试")
@click.option("--pg-deadlock-retries", default=3, type=int, help="死锁最大重试次数")
@click.option("--pg-statement-timeout-ms", default=0, type=int, help="语句超时(ms)，0表示不限制")
@click.option("--pg-lock-timeout-ms", default=0, type=int, help="锁等待超时(ms)，0表示不限制")
@click.option("--cohort-key", type=click.Choice(["first_order", "register", "custom"]), default="first_order", help="分群依据")
@click.option("--granularity", type=click.Choice(["day", "week", "month", "quarter"]), default="day", help="时间粒度")
@click.option("--retention-days", default=30, type=int, help="统计留存的天数")
@click.option("--retention-type", type=click.Choice(["standard", "rolling"]), default="standard", help="留存类型")
@click.option("--event-type", type=click.Choice(["order", "login", "active", "pay", "composite"]), default="order", help="回流事件类型")
@click.option("--composite-event", "composite_events", multiple=True, help="复合事件定义: name:expression（可多次指定）")
@click.option("--user-id-col", default="user_id", help="用户ID列名")
@click.option("--event-time-col", default="event_time", help="事件时间列名")
@click.option("--event-type-col", default=None, help="事件类型列名")
@click.option("--min-users", default=1, type=int, help="队列最小用户数")
@click.option("--dsl-cohort", default=None, help="DSL 分群时间表达式")
@click.option("--dsl-filter", default=None, help="DSL 过滤表达式")
@click.option("--dsl-segment", default=None, help="DSL 分段表达式")
@click.option("--dsl-error-recovery/--no-dsl-error-recovery", default=True, help="DSL 错误恢复")
@click.option("--dsl-default-numeric", default=0.0, type=float, help="DSL 数值默认值")
@click.option("--dsl-default-string", default="", help="DSL 字符串默认值")
@click.option("--dsl-coalesce-on-error/--no-dsl-coalesce-on-error", default=True, help="DSL 错误时 coalesce 为默认值")
@click.option("--dsl-strict-mode/--no-dsl-strict-mode", default=False, help="DSL 严格模式（不做错误恢复）")
@click.option("--dsl-fail-on-critical/--no-dsl-fail-on-critical", default=True, help="关键错误时终止")
@click.option("--outlier-enabled/--outlier-disabled", default=False, help="是否启用异常值剔除")
@click.option("--outlier-method", type=click.Choice(["iqr", "zscore", "percentile", "mad", "isolation_forest", "dbscan", "auto"]), default="iqr", help="异常值检测方法")
@click.option("--outlier-threshold", default=1.5, type=float, help="异常值检测阈值")
@click.option("--outlier-percentile-low", default=1.0, type=float, help="百分位数法下限")
@click.option("--outlier-percentile-high", default=99.0, type=float, help="百分位数法上限")
@click.option("--outlier-mad-threshold", default=3.0, type=float, help="MAD 方法阈值")
@click.option("--outlier-contamination", default=0.05, type=float, help="Isolation Forest 污染率")
@click.option("--outlier-eps", default=0.5, type=float, help="DBSCAN eps 参数")
@click.option("--outlier-min-samples", default=5, type=int, help="DBSCAN min_samples 参数")
@click.option("--outlier-remove-users/--no-outlier-remove-users", default=False, help="是否移除异常用户")
@click.option("--outlier-remove-cohorts/--no-outlier-remove-cohorts", default=True, help="是否移除异常队列")
@click.option("--outlier-target-col", default=None, help="异常值检测目标列")
@click.option("--outlier-auto-min-rows", default=1000, type=int, help="AUTO 方法最小样本量")
@click.option("--outlier-auto-max-rows", default=100000, type=int, help="AUTO 方法最大样本量阈值")
@click.option("--rolling-enabled/--rolling-disabled", default=False, help="是否启用滚动留存窗")
@click.option("--rolling-window", default=7, type=int, help="滚动窗口大小（天）")
@click.option("--rolling-step", default=1, type=int, help="滚动步长（天）")
@click.option("--rolling-min-periods", default=1, type=int, help="滚动最小观测期")
@click.option("--rolling-alignment", type=click.Choice(["left", "center", "right"]), default="left", help="滚动窗口对齐方式")
@click.option("--rolling-include-partial/--no-rolling-include-partial", default=False, help="是否包含不完整窗口")
@click.option("--rolling-timezone-handling", type=click.Choice(["naive", "utc", "local", "preserve"]), default="naive", help="时区处理模式")
@click.option("--rolling-target-timezone", default=None, help="目标时区（如 Asia/Shanghai）")
@click.option("--rolling-handle-dst/--no-rolling-handle-dst", default=True, help="处理夏令时偏移")
@click.option("--ssr-enabled/--no-ssr-enabled", default=True, help="HTML 报告 SSR 启用")
@click.option("--ssr-render-charts/--no-ssr-render-charts", default=True, help="SSR 渲染图表")
@click.option("--ssr-embed-data/--no-ssr-embed-data", default=True, help="SSR 嵌入数据")
@click.option("--ssr-minify/--no-ssr-minify", default=False, help="SSR 压缩 HTML")
@click.option("--parquet-spark-compatible/--no-parquet-spark-compatible", default=True, help="Parquet Spark 兼容")
@click.option("--parquet-compression", type=click.Choice(["snappy", "gzip", "brotli", "lz4", "zstd", "none"]), default="snappy", help="Parquet 压缩格式")
@click.option("--parquet-benchmark-codecs/--no-parquet-benchmark-codecs", default=False, help="运行 Parquet 压缩基准测试")
@click.option("--parquet-coerce-timestamps", type=click.Choice(["us", "ms", "none"]), default="us", help="时间戳强制精度")
@click.option("--seo-title", default="留存队列分析报告", help="HTML 报告 SEO 标题")
@click.option("--seo-description", default="基于 cohort-retention-cli 生成的用户留存分析报告，包含留存矩阵、漏斗分析、滚动留存等关键指标", help="SEO 描述")
@click.option("--seo-keywords", default="", help="SEO 关键词(逗号分隔）")
@click.option("--seo-author", default="cohort-retention-cli", help="SEO 作者")
@click.option("--seo-language", default="zh-CN", help="页面语言")
@click.option("--seo-og-type", default="article", help="Open Graph 类型")
@click.option("--seo-robots", default="noindex,nofollow", help="robots meta")
@click.option("--seo-enable-structured-data/--no-seo-enable-structured-data", default=True, help="嵌入 JSON-LD 结构化数据")
@click.option("--detect-conflicts/--no-detect-conflicts", default=True, help="检测参数冲突")
@click.option("--enable-conflict-suggestions/--no-enable-conflict-suggestions", default=True, help="生成冲突解决建议")
@click.option("--output-dir", default="./reports", help="报告输出目录")
@click.option("--format", "formats", type=click.Choice(["markdown", "json", "html"]), multiple=True, default=["markdown", "json"], help="报告格式")
@click.option("--matrix-format", type=click.Choice(["csv", "parquet"]), default="csv", help="矩阵导出格式")
@click.option("--export-matrix/--no-export-matrix", default=True, help="是否导出留存矩阵")
def analyze(
    source_type,
    path,
    table,
    encoding,
    host,
    port,
    username,
    password,
    database,
    schema,
    pg_isolation_level,
    pg_autocommit,
    pg_enable_savepoints,
    pg_nested_transaction_limit,
    pg_retry_on_deadlock,
    pg_deadlock_retries,
    pg_statement_timeout_ms,
    pg_lock_timeout_ms,
    cohort_key,
    granularity,
    retention_days,
    retention_type,
    event_type,
    composite_events,
    user_id_col,
    event_time_col,
    event_type_col,
    min_users,
    dsl_cohort,
    dsl_filter,
    dsl_segment,
    dsl_error_recovery,
    dsl_default_numeric,
    dsl_default_string,
    dsl_coalesce_on_error,
    dsl_strict_mode,
    dsl_fail_on_critical,
    outlier_enabled,
    outlier_method,
    outlier_threshold,
    outlier_percentile_low,
    outlier_percentile_high,
    outlier_mad_threshold,
    outlier_contamination,
    outlier_eps,
    outlier_min_samples,
    outlier_remove_users,
    outlier_remove_cohorts,
    outlier_target_col,
    outlier_auto_min_rows,
    outlier_auto_max_rows,
    rolling_enabled,
    rolling_window,
    rolling_step,
    rolling_min_periods,
    rolling_alignment,
    rolling_include_partial,
    rolling_timezone_handling,
    rolling_target_timezone,
    rolling_handle_dst,
    ssr_enabled,
    ssr_render_charts,
    ssr_embed_data,
    ssr_minify,
    parquet_spark_compatible,
    parquet_compression,
    parquet_benchmark_codecs,
    parquet_coerce_timestamps,
    seo_title,
    seo_description,
    seo_keywords,
    seo_author,
    seo_language,
    seo_og_type,
    seo_robots,
    seo_enable_structured_data,
    detect_conflicts,
    enable_conflict_suggestions,
    output_dir,
    formats,
    matrix_format,
    export_matrix,
):
    """运行留存队列分析并生成报告（30+ 完整参数）"""

    try:
        cohort_config = _build_cohort_config(
            cohort_key=cohort_key,
            granularity=granularity,
            retention_days=retention_days,
            retention_type=retention_type,
            event_type=event_type,
            composite_events=composite_events,
            user_id_col=user_id_col,
            event_time_col=event_time_col,
            event_type_col=event_type_col,
            min_users=min_users,
            dsl_cohort=dsl_cohort,
            dsl_filter=dsl_filter,
            dsl_segment=dsl_segment,
            dsl_error_recovery=dsl_error_recovery,
            dsl_default_numeric=dsl_default_numeric,
            dsl_default_string=dsl_default_string,
            dsl_coalesce_on_error=dsl_coalesce_on_error,
            dsl_strict_mode=dsl_strict_mode,
            dsl_fail_on_critical=dsl_fail_on_critical,
            outlier_enabled=outlier_enabled,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            outlier_percentile_low=outlier_percentile_low,
            outlier_percentile_high=outlier_percentile_high,
            outlier_mad_threshold=outlier_mad_threshold,
            outlier_contamination=outlier_contamination,
            outlier_eps=outlier_eps,
            outlier_min_samples=outlier_min_samples,
            outlier_remove_users=outlier_remove_users,
            outlier_remove_cohorts=outlier_remove_cohorts,
            outlier_target_col=outlier_target_col,
            outlier_auto_min_rows=outlier_auto_min_rows,
            outlier_auto_max_rows=outlier_auto_max_rows,
            rolling_enabled=rolling_enabled,
            rolling_window=rolling_window,
            rolling_step=rolling_step,
            rolling_min_periods=rolling_min_periods,
            rolling_alignment=rolling_alignment,
            rolling_include_partial=rolling_include_partial,
            rolling_timezone_handling=rolling_timezone_handling,
            rolling_target_timezone=rolling_target_timezone,
            rolling_handle_dst=rolling_handle_dst,
        )

        source_config = _build_source_config(
            source_type=source_type,
            path=path,
            table=table,
            encoding=encoding,
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            schema=schema,
            pg_isolation_level=pg_isolation_level,
            pg_autocommit=pg_autocommit,
            pg_enable_savepoints=pg_enable_savepoints,
            pg_nested_transaction_limit=pg_nested_transaction_limit,
            pg_retry_on_deadlock=pg_retry_on_deadlock,
            pg_deadlock_retries=pg_deadlock_retries,
            pg_statement_timeout_ms=pg_statement_timeout_ms,
            pg_lock_timeout_ms=pg_lock_timeout_ms,
        )

        report_config = _build_report_config(
            output_dir=output_dir,
            formats=formats,
            matrix_format=matrix_format,
            ssr_enabled=ssr_enabled,
            ssr_render_charts=ssr_render_charts,
            ssr_embed_data=ssr_embed_data,
            ssr_minify=ssr_minify,
            parquet_spark_compatible=parquet_spark_compatible,
            parquet_compression=parquet_compression,
            parquet_benchmark_codecs=parquet_benchmark_codecs,
            parquet_coerce_timestamps=parquet_coerce_timestamps,
            seo_title=seo_title,
            seo_description=seo_description,
            seo_keywords=seo_keywords,
            seo_author=seo_author,
            seo_language=seo_language,
            seo_og_type=seo_og_type,
            seo_robots=seo_robots,
            seo_enable_structured_data=seo_enable_structured_data,
            detect_conflicts=detect_conflicts,
            enable_conflict_suggestions=enable_conflict_suggestions,
        )

        errors = source_config.validate() + cohort_config.validate() + report_config.validate()
        if errors:
            for err in errors:
                click.echo(f"[错误] {err}", err=True)
            sys.exit(1)

        click.echo("加载数据中...")
        loader = create_loader(source_config, cohort_config)

        click.echo("执行队列分析中...")
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        click.echo("")
        click.echo(f"分析完成:")
        click.echo(f"  - 队列数: {result.summary.get('total_cohorts', 0)}")
        click.echo(f"  - 总用户数: {result.summary.get('total_users', 0):,}")

        outlier_stats = result.summary.get("outlier_stats")
        if outlier_stats:
            method_info = outlier_stats.get('method', '-')
            auto_info = ""
            if outlier_stats.get('method_resolved_from_auto') and outlier_stats.get('auto_reason'):
                auto_info = f" [AUTO: {outlier_stats['auto_reason']}]"
            click.echo(f"  - 异常移除: {outlier_stats.get('removed', 0)} 用户, {outlier_stats.get('cohorts_removed', 0)} 队列, 方法={method_info}{auto_info}")

        timezone_info = result.summary.get("timezone_info")
        if timezone_info:
            click.echo(f"  - 时区处理: {timezone_info}")

        analysis_warnings = result.summary.get("analysis_warnings")
        if analysis_warnings:
            click.echo("")
            click.echo("⚠️  分析警告:")
            for w in analysis_warnings[:5]:
                click.echo(f"  - {w}")
            if len(analysis_warnings) > 5:
                click.echo(f"  ... 还有 {len(analysis_warnings) - 5} 条警告")

        codec_metrics = None
        if report_config.parquet.benchmark_codecs:
            click.echo("")
            click.echo("运行 Parquet 压缩算法基准测试...")
            try:
                from .loaders.parquet_loader import ParquetDataLoader
                events_df = loader.load_events()
                codec_metrics = None
                if isinstance(loader, ParquetDataLoader):
                    codec_metrics = loader.benchmark_codecs(
                        events_df,
                        spark_compatible=report_config.parquet.spark_compatible,
                    )
                else:
                    tmp_source = DataSourceConfig(
                        source_type="parquet",
                        path="/tmp/_cohort_codec_bench.parquet",
                    )
                    bench_loader = ParquetDataLoader(tmp_source, cohort_config, report_config.parquet)
                    codec_metrics = bench_loader.benchmark_codecs(
                        events_df,
                        spark_compatible=report_config.parquet.spark_compatible,
                    )
                if codec_metrics:
                    click.echo("")
                    click.echo("📊 Parquet 压缩基准测试:")
                    click.echo(f"  {'压缩算法':<12} {'原始KB':>10} {'压缩后KB':>10} {'压缩比':>8} {'写入ms':>10} {'读取ms':>10} {'备注':>10}")
                    for m in codec_metrics:
                        orig_kb = m.original_bytes / 1024
                        comp_kb = m.compressed_bytes / 1024
                        note = m.note or ("✓ Spark" if m.spark_compatible else "")
                        click.echo(
                            f"  {m.compression:<12} {orig_kb:>10.1f} {comp_kb:>10.1f} "
                            f"{m.compression_ratio:>7.2f}x {m.write_time_ms:>10.1f} "
                            f"{m.read_time_ms:>10.1f} {note:>10}"
                        )
            except Exception as e:
                click.echo(f"  ⚠️  基准测试失败: {e}")

        generated_files = []

        if "markdown" in formats:
            md_reporter = MarkdownReporter(report_config)
            md_path = md_reporter.save(result)
            generated_files.append(md_path)
            click.echo(f"  - Markdown 报告: {md_path}")

        if "json" in formats:
            json_reporter = JsonReporter(report_config)
            json_path = json_reporter.save(result)
            generated_files.append(json_path)
            click.echo(f"  - JSON 报告: {json_path}")

        if "html" in formats:
            html_reporter = HtmlReporter(report_config)
            html_path = html_reporter.save(result)
            generated_files.append(html_path)
            click.echo(f"  - HTML 报告: {html_path}")

        if export_matrix:
            exporter = MatrixExporter(report_config)
            if matrix_format == "csv":
                counts_path = exporter.export_counts_csv(result)
                rates_path = exporter.export_rates_csv(result)
            elif matrix_format == "parquet":
                counts_path = exporter.export_counts_parquet(result)
                rates_path = exporter.export_rates_parquet(result)
            generated_files.extend([counts_path, rates_path])
            click.echo(f"  - 留存人数矩阵: {counts_path}")
            click.echo(f"  - 留存率矩阵: {rates_path}")

        click.echo("")
        click.echo("全部文件已生成 ✅")

    except Exception as e:
        click.echo(f"[错误] 分析失败: {str(e)}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("output_path", default="./sample_data.csv")
@click.option("--users", default=200, type=int, help="用户数量")
@click.option("--days", default=60, type=int, help="数据跨度天数")
@click.option("--format", "output_format", type=click.Choice(["csv", "parquet"]), default="csv", help="输出格式")
def gen_sample(output_path, users, days, output_format):
    """生成示例数据用于测试"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    click.echo(f"生成示例数据到 {output_path} ...")

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    user_ids = [f"user_{i:04d}" for i in range(1, users + 1)]

    records = []
    for user_id in user_ids:
        register_offset = np.random.randint(0, days)
        register_date = start_date + timedelta(days=register_offset)

        records.append({
            "user_id": user_id,
            "event_time": register_date,
            "event_type": "register",
            "amount": 0,
        })

        num_orders = np.random.poisson(lam=3)
        for _ in range(num_orders):
            order_offset = np.random.randint(0, days - register_offset)
            order_date = register_date + timedelta(days=order_offset, hours=np.random.randint(0, 24))
            if order_date <= end_date:
                records.append({
                    "user_id": user_id,
                    "event_time": order_date,
                    "event_type": "order",
                    "amount": round(np.random.uniform(10, 500), 2),
                })

        if np.random.random() < 0.7:
            login_offset = np.random.randint(0, days - register_offset)
            login_date = register_date + timedelta(days=login_offset, hours=np.random.randint(0, 24))
            if login_date <= end_date:
                records.append({
                    "user_id": user_id,
                    "event_time": login_date,
                    "event_type": "login",
                    "amount": 0,
                })

    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    if output_format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif output_format == "parquet":
        df.to_parquet(output_path, index=False)

    click.echo(f"✅ 已生成 {len(df)} 条记录，共 {users} 个用户")
    click.echo(f"   时间范围: {start_date.date()} ~ {end_date.date()}")
    click.echo(f"   格式: {output_format}")


@cli.command(name="init-db")
@click.argument("db_path")
@click.option("--csv-path", required=True, help="源 CSV 数据路径")
@click.option("--table", default="events", help="表名")
def init_db(db_path, csv_path, table):
    """从 CSV 创建 SQLite 数据库"""
    import sqlite3
    import pandas as pd

    click.echo(f"从 {csv_path} 创建 SQLite 数据库 {db_path} ...")

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        click.echo(f"✅ 已导入 {len(df)} 条记录到表 {table}")
    finally:
        conn.close()


@cli.command(name="write-pg")
@click.option("--source-path", required=True, help="源数据文件路径（CSV/Parquet）")
@click.option("--source-format", type=click.Choice(["csv", "parquet"]), default="csv", help="源数据格式")
@click.option("--host", default="localhost", help="PostgreSQL 主机")
@click.option("--port", default=5432, type=int, help="PostgreSQL 端口")
@click.option("--username", required=True, help="PostgreSQL 用户名")
@click.option("--password", default=None, help="PostgreSQL 密码")
@click.option("--database", required=True, help="PostgreSQL 数据库名")
@click.option("--schema", default=None, help="PostgreSQL schema")
@click.option("--table", default="events", help="目标表名")
@click.option("--if-exists", type=click.Choice(["fail", "replace", "append"]), default="replace", help="表已存在时的处理方式")
def write_pg(source_path, source_format, host, port, username, password, database, schema, table, if_exists):
    """将数据写入 PostgreSQL 数据库"""
    import pandas as pd

    click.echo(f"从 {source_path} 写入 PostgreSQL {host}:{port}/{database} ...")

    try:
        if source_format == "csv":
            df = pd.read_csv(source_path)
        elif source_format == "parquet":
            df = pd.read_parquet(source_path)

        source_config = DataSourceConfig(
            source_type="postgresql",
            path="",
            table=table,
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            schema=schema,
        )

        cohort_config = CohortConfig()

        from .loaders.postgresql_loader import PostgreSqlDataLoader
        loader = PostgreSqlDataLoader(source_config, cohort_config)
        loader.write_events(df, if_exists=if_exists)

        click.echo(f"✅ 已写入 {len(df)} 条记录到 {schema + '.' if schema else ''}{table}")

    except Exception as e:
        click.echo(f"[错误] 写入失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command(name="validate-dsl")
@click.option("--dsl-cohort", default=None, help="DSL 分群时间表达式")
@click.option("--dsl-filter", default=None, help="DSL 过滤表达式")
@click.option("--dsl-segment", default=None, help="DSL 分段表达式")
def validate_dsl(dsl_cohort, dsl_filter, dsl_segment):
    """验证 DSL 表达式语法"""
    dsl_config = CohortDSLConfig(
        cohort_expression=dsl_cohort,
        filter_expression=dsl_filter,
        segment_expression=dsl_segment,
    )

    engine = CohortDSLEngine(dsl_config)
    valid, errors = engine.validate_expressions()

    if valid:
        click.echo("✅ 所有 DSL 表达式语法正确")
        if dsl_cohort:
            click.echo(f"  - 分群表达式: {dsl_cohort}")
        if dsl_filter:
            click.echo(f"  - 过滤表达式: {dsl_filter}")
        if dsl_segment:
            click.echo(f"  - 分段表达式: {dsl_segment}")
    else:
        click.echo("❌ DSL 表达式存在错误:")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)


@cli.command(name="list-functions")
def list_functions():
    """列出 DSL 可用的函数"""
    from .dsl import DSLParser

    click.echo("DSL 可用函数:")
    click.echo("")
    click.echo("比较运算符: ==, !=, <, <=, >, >=")
    click.echo("逻辑运算符: and, or, not")
    click.echo("算术运算符: +, -, *, /, //, %, **")
    click.echo("")
    click.echo("函数列表:")
    for func_name in sorted(DSLParser._FUNCTIONS.keys()):
        click.echo(f"  - {func_name}")


if __name__ == "__main__":
    cli()
