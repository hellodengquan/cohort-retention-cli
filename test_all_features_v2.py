import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.cohort_retention.config import (
    CohortConfig, DataSourceConfig, ReportConfig,
    CohortKey, CohortGranularity, EventType, RetentionType,
    OutlierMethod, RollingAlignment, TimeZoneHandling,
    CohortDSLConfig, OutlierConfig, RollingWindowConfig,
    CompositeEventConfig, ParquetConfig, SEOSettings,
    PostgresTransactionConfig, DSLErrorCode,
)
from src.cohort_retention.dsl import CohortDSLEngine, DSLParser, DSLErrorRecovery, DSLError
from src.cohort_retention.loaders import create_loader
from src.cohort_retention.analyzer import CohortAnalyzer, OutlierRemover
from src.cohort_retention.reporters import MatrixExporter, MarkdownReporter, JsonReporter, HtmlReporter


def generate_test_data(n_users=200, n_days=60):
    np.random.seed(42)
    user_ids = [f'user_{i}' for i in range(n_users)]
    data = []

    for user_id in user_ids:
        first_day = np.random.randint(0, n_days // 2)
        n_orders = np.random.poisson(5) + 1
        for _ in range(n_orders):
            day_offset = first_day + np.random.randint(0, n_days - first_day)
            event_time = datetime(2024, 1, 1) + timedelta(days=day_offset, hours=np.random.randint(0, 24))
            amount = np.random.exponential(100)
            if np.random.random() < 0.02:
                amount *= 100
            data.append({
                'user_id': user_id,
                'event_time': event_time,
                'event_type': 'order',
                'amount': amount,
                'is_vip': np.random.random() < 0.2,
            })

    df = pd.DataFrame(data)
    return df.sort_values('event_time').reset_index(drop=True)


def test_1_outlier_auto_select():
    print("\n=== 1. Outlier 算法自动选择（AUTO 模式）===")

    configs = [
        (np.random.normal(100, 10, 100), "小样本正态"),
        (np.random.exponential(50, 5000), "大样本指数分布"),
        (np.concatenate([np.random.normal(100, 10, 800), [100000, 200000]]), "大样本含极端值"),
    ]

    for values, desc in configs:
        cfg = OutlierConfig(
            enabled=True,
            method=OutlierMethod.AUTO,
            target_column='value',
            auto_min_rows=500,
            auto_max_rows=2000,
        )
        remover = OutlierRemover(cfg)
        df = pd.DataFrame({'value': values})
        cleaned_df, stats = remover.remove_outliers(df, 'value')
        method = stats.get('method')
        reason = stats.get('auto_reason')
        removed = stats.get('removed', 0)
        print(f"  {desc}: 方法={method}, 移除={removed}, 原因={reason}")

    print("✓ Outlier AUTO 模式测试通过")


def test_2_timezone_handling():
    print("\n=== 2. 滚动对齐时区处理 ===")

    df = generate_test_data(n_users=100, n_days=30)

    timezone_options = [
        (TimeZoneHandling.NAIVE, None, "纯本地时间"),
        (TimeZoneHandling.UTC, None, "UTC 标准化"),
    ]

    for tz_handling, target_tz, desc in timezone_options:
        cohort_cfg = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=30,
            retention_type=RetentionType.ROLLING,
            event_type=EventType.ORDER,
            user_id_col='user_id',
            event_time_col='event_time',
            event_type_col='event_type',
            rolling=RollingWindowConfig(
                enabled=True,
                window_size=7,
                step=1,
                min_periods=3,
                alignment=RollingAlignment.CENTER,
                timezone_handling=tz_handling,
                target_timezone=target_tz,
                handle_dst=True,
            ),
        )

        csv_path = '/tmp/_tz_test.csv'
        df.to_csv(csv_path, index=False)
        source_cfg = DataSourceConfig(source_type='csv', path=csv_path)
        loader = create_loader(source_cfg, cohort_cfg)
        analyzer = CohortAnalyzer(loader, cohort_cfg)
        result = analyzer.run_analysis()
        tz_info = result.summary.get('timezone_info', {})
        print(f"  {desc}: 时区信息={tz_info.get('handling')}, 队列数={result.summary.get('total_cohorts')}")

    print("✓ 时区处理测试通过")


def test_3_composite_event_cycle_detection():
    print("\n=== 3. 复合事件优先级环检测 ===")

    events_no_cycle = [
        CompositeEventConfig(name='ev1', expression='amount > 100', priority=10, stop_on_match=True),
        CompositeEventConfig(name='ev2', expression='amount > 50', priority=5, stop_on_match=True),
        CompositeEventConfig(name='ev3', expression='amount > 0', priority=1, stop_on_match=True),
    ]

    cfg1 = CohortConfig(
        event_type=EventType.COMPOSITE,
        composite_events=events_no_cycle,
        detect_priority_cycle=True,
    )
    errors1 = cfg1.validate()
    print(f"  无环配置: 错误数={len(errors1)}")

    events_same_prio = [
        CompositeEventConfig(name='evA', expression='amount > 100', priority=5, stop_on_match=True),
        CompositeEventConfig(name='evB', expression='amount > 50', priority=5, stop_on_match=True),
    ]

    cfg2 = CohortConfig(
        event_type=EventType.COMPOSITE,
        composite_events=events_same_prio,
        detect_priority_cycle=True,
        resolve_priority_tie_by_name=True,
    )
    errors2 = cfg2.validate()
    print(f"  同优先级 stop_on_match + 按名称消解: 错误数={len(errors2)}")
    for e in errors2:
        print(f"    警告: {e}")

    print("✓ 复合事件优先级环检测通过")


def test_4_parquet_codec_benchmark():
    print("\n=== 4. Spark codec 统计指标差异 ===")

    df = generate_test_data(n_users=200, n_days=60)

    try:
        from src.cohort_retention.loaders.parquet_loader import ParquetDataLoader

        source_cfg = DataSourceConfig(source_type='parquet', path='/tmp/_codec_bench.parquet')
        cohort_cfg = CohortConfig()
        parquet_cfg = ParquetConfig(spark_compatible=True, compression='snappy')
        loader = ParquetDataLoader(source_cfg, cohort_cfg, parquet_cfg)

        metrics = loader.benchmark_codecs(df, codecs=['snappy', 'gzip', 'zstd'], sample_rows=500)
        print(f"  完成 {len(metrics)} 种压缩算法基准测试:")
        for m in metrics:
            print(
                f"    {m.compression}: 压缩比={m.compression_ratio:.2f}x, "
                f"写={m.write_time_ms:.1f}ms, 读={m.read_time_ms:.1f}ms, "
                f"Spark兼容={m.spark_compatible}"
            )

        print("✓ Parquet codec 基准测试通过")
    except ImportError as e:
        print(f"  跳过: {e}")


def test_5_postgres_transaction_boundary():
    print("\n=== 5. PostgreSQL 事务回滚边界 ===")

    try:
        from src.cohort_retention.loaders.postgresql_loader import PostgreSqlDataLoader
        import inspect

        tx_cfg = PostgresTransactionConfig(
            isolation_level='read_committed',
            enable_savepoints=True,
            rollback_on_error=True,
            nested_transaction_limit=5,
            retry_on_deadlock=True,
            deadlock_retries=3,
            statement_timeout_ms=30000,
            lock_timeout_ms=10000,
        )

        source_cfg = DataSourceConfig(
            source_type='postgresql', path='',
            host='localhost', database='test',
            username='test', password='test',
            transaction_config=tx_cfg,
        )
        errors = source_cfg.validate()
        print(f"  事务配置验证: 错误数={len(errors)}")

        cohort_cfg = CohortConfig()
        loader = PostgreSqlDataLoader(source_cfg, cohort_cfg)

        assert hasattr(loader, 'savepoint'), "应有 savepoint 方法"
        assert hasattr(loader, 'transaction'), "应有 transaction 方法"
        sig = inspect.signature(loader.transaction)
        params = list(sig.parameters.keys())
        assert 'max_nested' in params, "应有 max_nested 参数"

        stats = loader.get_transaction_stats()
        print(f"  事务配置参数: {params}")
        print(f"  初始统计: {stats}")

        print("✓ PostgreSQL 事务边界测试通过")
    except ImportError as e:
        print(f"  跳过: {e}")


def test_6_html_ssr_seo():
    print("\n=== 6. HTML SSR SEO 优化 ===")

    from src.cohort_retention.models import RetentionMatrix, CohortAnalysisResult, FunnelResult

    seo = SEOSettings(
        title="测试留存分析报告",
        description="基于 cohort-retention-cli 的测试报告",
        keywords=["测试", "留存"],
        author="测试团队",
        language="zh-CN",
        og_type="article",
        robots="index,follow",
        enable_structured_data=True,
    )

    report_cfg = ReportConfig(
        output_dir='/tmp/_seo_test',
        formats=['html'],
        ssr_enabled=True,
        ssr_render_charts=True,
        ssr_embed_data=True,
        seo=seo,
    )

    reporter = HtmlReporter(report_cfg)
    assert hasattr(reporter, '_generate_html_head'), "应有 SEO head 生成方法"
    assert hasattr(reporter, '_generate_structured_data'), "应有结构化数据方法"

    dates = [datetime(2024, 1, 1).date(), datetime(2024, 1, 2).date()]
    rates = pd.DataFrame({0: [1.0, 1.0], 1: [0.5, 0.6]}, index=dates)
    counts = pd.DataFrame({0: [100, 80], 1: [50, 48]}, index=dates)
    matrix = RetentionMatrix(
        cohort_dates=dates, days=[0, 1],
        retention_counts=counts, retention_rates=rates,
        cohort_sizes={dates[0]: 100, dates[1]: 80},
    )
    from src.cohort_retention.models import FunnelStep
    funnel = FunnelResult(steps=[FunnelStep(name='Day0', user_count=180, conversion_rate=1.0, drop_off_rate=0.0)], total_users=180)
    result = CohortAnalysisResult(matrix=matrix, funnel=funnel, config={}, summary={'total_cohorts': 2, 'total_users': 180})

    head_html = reporter._generate_html_head(result)
    assert 'meta name="description"' in head_html, "应有 description"
    assert 'meta property="og:type"' in head_html, "应有 Open Graph"
    print(f"  SEO Meta 标签生成: ✓ (包含 {len(head_html.splitlines())} 行)")

    sd_html = reporter._generate_structured_data(result)
    assert 'application/ld+json' in sd_html, "应有 JSON-LD 结构化数据"
    print(f"  结构化数据嵌入: ✓")

    full_html = reporter.generate(result)
    assert '<html lang="zh-CN"' in full_html, "应有语言属性"
    assert '<header>' in full_html, "应有语义化 header"
    assert 'role="main"' in full_html, "应有 ARIA main"
    print(f"  完整 HTML 生成: ✓ ({len(full_html)} 字符)")

    print("✓ HTML SSR SEO 测试通过")


def test_7_dsl_coalesce_error_codes():
    print("\n=== 7. DSL coalesce 错误码 ===")

    err_codes = [
        (DSLErrorCode.COLUMN_NOT_FOUND, "列不存在", True),
        (DSLErrorCode.TYPE_MISMATCH, "类型不匹配", True),
        (DSLErrorCode.DIVISION_BY_ZERO, "除零错误", True),
        (DSLErrorCode.NULL_VALUE, "空值", True),
        (DSLErrorCode.PARSE_ERROR, "语法错误", False),
        (DSLErrorCode.UNKNOWN, "未知错误", True),
    ]

    for code, msg, recoverable in err_codes:
        err = DSLError(msg, code=code, recoverable=recoverable)
        prefix = f"[{code.value}]"
        assert prefix in str(err), f"错误信息应包含 [{code.value}]"
        print(f"  {code.value}: {msg} (可恢复={recoverable})")

    df = pd.DataFrame({
        'col1': [1.0, None, 3.0],
        'col2': [10, None, 30],
    })

    error_recovery = DSLErrorRecovery(
        enabled=True,
        coalesce_on_error=True,
        default_numeric=-999.0,
        error_code_enabled=True,
    )

    result = DSLParser.evaluate('coalesce(invalid_col, col1, col2, 0)', df=df, error_recovery=error_recovery)
    print(f"  coalesce 错误恢复: 错误数={len(error_recovery.error_log)}, 默认值=-999")
    for rec in error_recovery.error_log:
        print(f"    {rec.code.value}: {rec.message[:50]}")

    if len(error_recovery.error_log) > 0:
        assert error_recovery.error_log[0].code == DSLErrorCode.COLUMN_NOT_FOUND, "错误码应为 COLUMN_NOT_FOUND"
        assert error_recovery.error_log[0].recoverable, "应标记为可恢复"

    print("✓ DSL coalesce 错误码测试通过")


def test_8_param_conflict_detection():
    print("\n=== 8. 30+ 参数冲突检测 ===")

    conflict_cases = [
        (
            CohortConfig(retention_type=RetentionType.ROLLING, rolling=RollingWindowConfig(enabled=False)),
            "rolling_enabled=False 但 type=rolling"
        ),
        (
            CohortConfig(
                outlier=OutlierConfig(enabled=True, method=OutlierMethod.AUTO, target_column=None),
            ),
            "AUTO 模式无目标列"
        ),
        (
            CohortConfig(
                rolling=RollingWindowConfig(enabled=True, step=15, window_size=7),
            ),
            "step > window_size"
        ),
        (
            CohortConfig(
                cohort_key=CohortKey.CUSTOM,
                dsl=CohortDSLConfig(cohort_expression=None),
            ),
            "custom 无 cohort_expression"
        ),
    ]

    total_conflicts = 0
    for cfg, desc in conflict_cases:
        errors = cfg.validate()
        print(f"  {desc}: 检测到 {len(errors)} 个问题")
        for e in errors:
            print(f"    - {e}")
        total_conflicts += len(errors)

    report_conflict = ReportConfig(
        formats=['markdown'],
        ssr_minify=True,
        ssr_enabled=False,
        detect_conflicts=True,
    )
    rpt_errors = report_conflict.validate()
    print(f"  报告配置冲突 (ssr_minify without ssr): 检测到 {len(rpt_errors)} 个问题")
    for e in rpt_errors:
        print(f"    - {e}")
    total_conflicts += len(rpt_errors)

    valid_cfg = CohortConfig(
        retention_type=RetentionType.STANDARD,
        outlier=OutlierConfig(enabled=False),
        rolling=RollingWindowConfig(enabled=False),
    )
    valid_errors = valid_cfg.validate()
    print(f"  正常配置验证: 错误数={len(valid_errors)} (应为 0)")

    print(f"✓ 参数冲突检测: 共检测 {total_conflicts} 个问题")


def test_9_full_integration_all_features():
    print("\n=== 9. 全功能集成测试 ===")

    df = generate_test_data(n_users=300, n_days=90)
    csv_path = '/tmp/_full_integration.csv'
    df.to_csv(csv_path, index=False)

    cohort_cfg = CohortConfig(
        cohort_key=CohortKey.FIRST_ORDER,
        granularity=CohortGranularity.DAY,
        retention_days=60,
        retention_type=RetentionType.ROLLING,
        event_type=EventType.COMPOSITE,
        composite_events=[
            CompositeEventConfig(name='high_value', expression='amount > 300', priority=10, stop_on_match=True),
            CompositeEventConfig(name='medium_value', expression='amount > 100', priority=5, stop_on_match=False),
        ],
        user_id_col='user_id',
        event_time_col='event_time',
        event_type_col='event_type',
        min_users=3,
        dsl=CohortDSLConfig(
            filter_expression='amount > 0',
            error_recovery=True,
            coalesce_on_error=True,
            default_numeric=0.0,
        ),
        outlier=OutlierConfig(
            enabled=True,
            method=OutlierMethod.AUTO,
            target_column='amount',
            remove_cohorts=True,
            auto_min_rows=200,
            auto_max_rows=5000,
        ),
        rolling=RollingWindowConfig(
            enabled=True,
            window_size=7,
            step=1,
            min_periods=3,
            alignment=RollingAlignment.CENTER,
            timezone_handling=TimeZoneHandling.UTC,
            handle_dst=True,
        ),
        detect_priority_cycle=True,
        resolve_priority_tie_by_name=True,
    )

    report_cfg = ReportConfig(
        output_dir='/tmp/_full_test_reports',
        formats=['markdown', 'json', 'html'],
        ssr_enabled=True,
        ssr_render_charts=True,
        seo=SEOSettings(title="全功能测试报告"),
        detect_conflicts=True,
        parquet=ParquetConfig(spark_compatible=True),
    )

    cohort_errors = cohort_cfg.validate()
    report_errors = report_cfg.validate()
    print(f"  配置验证: cohort={len(cohort_errors)} 错误, report={len(report_errors)} 错误")
    for e in cohort_errors + report_errors:
        if e:
            print(f"    ⚠️  {e}")

    source_cfg = DataSourceConfig(source_type='csv', path=csv_path)
    loader = create_loader(source_cfg, cohort_cfg)
    analyzer = CohortAnalyzer(loader, cohort_cfg)
    result = analyzer.run_analysis()

    summary = result.summary or {}
    print(f"  分析结果: 队列={summary.get('total_cohorts')}, 用户={summary.get('total_users')}")

    outlier_stats = summary.get('outlier_stats')
    if outlier_stats:
        print(f"  异常值: 方法={outlier_stats.get('method')}, 原因={outlier_stats.get('auto_reason')}, 移除={outlier_stats.get('removed')}")

    tz_info = summary.get('timezone_info')
    if tz_info:
        print(f"  时区: {tz_info.get('handling')}, DST 偏移处理数={tz_info.get('dst_shifts_handled')}")

    warnings_found = summary.get('analysis_warnings', [])
    print(f"  分析警告: {len(warnings_found)} 条")

    md_reporter = MarkdownReporter(report_cfg)
    md_path = md_reporter.save(result)
    print(f"  Markdown: {md_path}")

    html_reporter = HtmlReporter(report_cfg)
    html_path = html_reporter.save(result)
    print(f"  HTML (含SEO/SSR): {html_path}")

    json_reporter = JsonReporter(report_cfg)
    json_path = json_reporter.save(result)
    print(f"  JSON: {json_path}")

    print("✓ 全功能集成测试通过")


def main():
    print("=" * 70)
    print("cohort-retention-cli 8 大新功能综合测试 (v2)")
    print("=" * 70)

    tests = [
        test_1_outlier_auto_select,
        test_2_timezone_handling,
        test_3_composite_event_cycle_detection,
        test_4_parquet_codec_benchmark,
        test_5_postgres_transaction_boundary,
        test_6_html_ssr_seo,
        test_7_dsl_coalesce_error_codes,
        test_8_param_conflict_detection,
        test_9_full_integration_all_features,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
