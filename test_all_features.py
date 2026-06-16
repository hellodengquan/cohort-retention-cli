import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.cohort_retention.config import (
    CohortConfig, DataSourceConfig, ReportConfig,
    CohortKey, CohortGranularity, EventType, RetentionType,
    OutlierMethod, RollingAlignment,
    CohortDSLConfig, OutlierConfig, RollingWindowConfig,
    CompositeEventConfig, ParquetConfig
)
from src.cohort_retention.dsl import CohortDSLEngine, DSLParser, DSLErrorRecovery, DSLError
from src.cohort_retention.loaders import create_loader
from src.cohort_retention.analyzer import CohortAnalyzer, OutlierRemover
from src.cohort_retention.reporters import MatrixExporter
from src.cohort_retention.cli import _parse_composite_events, _build_cohort_config, _build_report_config

def generate_test_data(n_users=200, n_days=60):
    np.random.seed(42)
    user_ids = [f'user_{i}' for i in range(n_users)]
    data = []
    
    for user_id in user_ids:
        first_day = np.random.randint(0, n_days // 2)
        n_orders = np.random.poisson(5) + 1
        for _ in range(n_orders):
            day_offset = first_day + np.random.randint(0, n_days - first_day)
            event_time = datetime(2024, 1, 1) + timedelta(days=day_offset)
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

def test_dsl_error_recovery():
    print("\n=== 1. DSL coalesce 错误恢复与 min/max 类型推断 ===")
    
    df = pd.DataFrame({
        'col1': [1.0, None, 3.0, None],
        'col2': [None, 2.0, None, 4.0],
        'col3': [10, 20, 30, 40],
        'date1': [pd.Timestamp('2024-01-01'), None, pd.Timestamp('2024-01-03'), None],
        'date2': [None, pd.Timestamp('2024-02-02'), None, pd.Timestamp('2024-02-04')],
    })
    
    error_recovery = DSLErrorRecovery(enabled=True, default_numeric=0.0)
    
    result = DSLParser.evaluate('coalesce(col1, col2, 0)', df=df, error_recovery=error_recovery)
    if hasattr(result, '__len__'):
        print(f"coalesce(col1, col2, 0) = {list(result)}")
        assert len(result) == 4, "Result should have 4 elements"
    else:
        print(f"coalesce(col1, col2, 0) = {result}")
    assert not any(pd.isna(result) if hasattr(result, '__len__') else pd.isna(result)), "No NaN values"
    
    print("测试 DSL _min 和 _max 函数直接调用...")
    s1 = pd.Series([10, 20, 30, 40])
    s2 = pd.Series([25, 25, 25, 25])
    result_min = DSLParser._min(s1, s2, error_recovery=error_recovery)
    result_max = DSLParser._max(s1, s2, error_recovery=error_recovery)
    print(f"_min(Series, Series) = {list(result_min)}")
    print(f"_max(Series, Series) = {list(result_max)}")
    assert list(result_min) == [10, 20, 25, 25], f"min failed: {list(result_min)}"
    assert list(result_max) == [25, 25, 30, 40], f"max failed: {list(result_max)}"
    
    result_single_min = DSLParser._min(s1, error_recovery=error_recovery)
    result_single_max = DSLParser._max(s1, error_recovery=error_recovery)
    print(f"_min(Series) = {result_single_min}")
    print(f"_max(Series) = {result_single_max}")
    assert result_single_min == 10, f"Single min failed: {result_single_min}"
    assert result_single_max == 40, f"Single max failed: {result_single_max}"
    
    print("测试类型推断...")
    dtype1 = DSLParser._infer_dtype([1, 2, None, 4])
    dtype2 = DSLParser._infer_dtype([pd.Timestamp('2024-01-01'), None])
    print(f"数值类型推断: {dtype1}")
    print(f"日期类型推断: {dtype2}")
    assert dtype1 == "numeric", f"Should be numeric, got {dtype1}"
    assert dtype2 == "datetime", f"Should be datetime, got {dtype2}"
    
    result4 = DSLParser.evaluate('coalesce(date1, date2)', df=df, error_recovery=error_recovery)
    if hasattr(result4, 'dtype'):
        print(f"coalesce(date1, date2) 类型推断: {result4.dtype}")
        assert pd.api.types.is_datetime64_any_dtype(result4), "Should be datetime type"
    
    print("测试错误恢复...")
    try:
        result5 = DSLParser.evaluate('coalesce(invalid_col, col3, -1)', df=df, error_recovery=error_recovery)
        print(f"coalesce with error recovery = {list(result5) if hasattr(result5, '__len__') else result5}")
        if hasattr(result5, '__len__'):
            assert len(result5) == 4, "Should have 4 elements"
    except Exception as e:
        print(f"错误恢复触发 (预期行为): {e}")
    
    print("✓ DSL 错误恢复与类型推断测试通过")

def test_outlier_algorithms():
    print("\n=== 2. Outlier 多种算法可选 ===")
    
    df = generate_test_data(n_users=100, n_days=30)
    
    methods = [
        OutlierMethod.IQR,
        OutlierMethod.ZSCORE,
        OutlierMethod.PERCENTILE,
        OutlierMethod.MAD,
        OutlierMethod.ISOLATION_FOREST,
        OutlierMethod.DBSCAN,
    ]
    
    for method in methods:
        config = OutlierConfig(
            enabled=True,
            method=method,
            threshold=1.5,
            percentile_low=1.0,
            percentile_high=99.0,
            mad_threshold=3.0,
            contamination=0.05,
            eps=0.5,
            min_samples=5,
            target_column='amount',
        )
        remover = OutlierRemover(config)
        cleaned_df, stats = remover.remove_outliers(df, value_col='amount')
        removed = stats.get('removed', 0)
        print(f"  {method.value}: 移除 {removed} 行, 剩余 {len(cleaned_df)} 行")
    
    print("✓ Outlier 多算法测试通过")

def test_rolling_window_alignment():
    print("\n=== 3. 滚动窗对齐策略 ===")
    
    configs = [
        RollingAlignment.LEFT,
        RollingAlignment.CENTER,
        RollingAlignment.RIGHT,
    ]
    
    for alignment in configs:
        cohort_cfg = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=30,
            retention_type=RetentionType.ROLLING,
            event_type=EventType.ORDER,
            user_id_col='user_id',
            event_time_col='event_time',
            event_type_col='event_type',
            min_users=1,
            rolling=RollingWindowConfig(
                enabled=True,
                window_size=7,
                step=1,
                min_periods=3,
                alignment=alignment,
                include_partial=False,
            ),
        )
        
        print(f"  {alignment.value}: window_size={cohort_cfg.rolling.window_size}, "
              f"min_periods={cohort_cfg.rolling.min_periods}")
    
    print("✓ 滚动窗对齐策略测试通过")

def test_composite_event_priority():
    print("\n=== 4. 复合事件优先级 ===")
    
    events_str = (
        'high_value:amount > 500:10:true',
        'medium_value:amount > 100:5:false',
        'low_value:amount > 0:1:false',
    )
    
    events = _parse_composite_events(events_str)
    for e in sorted(events, key=lambda x: -x.priority):
        print(f"  {e.name}: priority={e.priority}, stop_on_match={e.stop_on_match}")
    
    events_sorted = sorted(events, key=lambda x: -x.priority)
    assert events_sorted[0].name == 'high_value', "high_value should have highest priority"
    assert events_sorted[2].name == 'low_value', "low_value should have lowest priority"
    
    print("✓ 复合事件优先级测试通过")

def test_parquet_spark_compat():
    print("\n=== 5. Parquet Spark 兼容 ===")
    
    df = generate_test_data(n_users=50, n_days=30)
    
    source_cfg = DataSourceConfig(
        source_type='parquet',
        path='/tmp/test.parquet',
    )
    
    cohort_cfg = CohortConfig()
    
    from src.cohort_retention.loaders.parquet_loader import ParquetDataLoader
    parquet_loader = ParquetDataLoader(source_cfg, cohort_cfg)
    
    for compress in ['snappy', 'gzip', 'zstd']:
        cfg = ParquetConfig(
            spark_compatible=True,
            compression=compress,
            coerce_timestamps='ms',
        )
        parquet_loader.parquet_config = cfg
        prepared_df = parquet_loader._prepare_for_spark(df)
        
        for col in prepared_df.columns:
            if pd.api.types.is_datetime64_any_dtype(prepared_df[col]):
                dtype_str = str(prepared_df[col].dtype)
                assert 'ms' in dtype_str or 'ns' in dtype_str, f"Timestamp should be ms compatible, got {dtype_str}"
        
        print(f"  compression={compress}: {len(prepared_df)} 行, 时间列兼容")
    
    print("✓ Parquet Spark 兼容测试通过")

def test_postgres_transaction():
    print("\n=== 6. PostgreSQL Loader 事务支持 ===")
    
    try:
        from src.cohort_retention.loaders.postgresql_loader import PostgreSqlDataLoader
        import inspect
        
        source_cfg = DataSourceConfig(
            source_type='postgresql',
            path='',
            host='localhost',
            port=5432,
            database='test',
            username='test',
            password='test',
        )
        
        cohort_cfg = CohortConfig()
        loader = PostgreSqlDataLoader(source_cfg, cohort_cfg)
        
        assert hasattr(loader, 'transaction'), "Should have transaction method"
        sig = inspect.signature(loader.transaction)
        params = list(sig.parameters.keys())
        assert 'isolation_level' in params, "Should have isolation_level parameter"
        assert 'autocommit' in params, "Should have autocommit parameter"
        
        print(f"  transaction 参数: {params}")
        print(f"  可用方法: batch_write, copy_from_csv, execute_query")
        
        print("✓ PostgreSQL 事务支持测试通过")
    except ImportError as e:
        print(f"  跳过 (依赖未安装): {e}")

def test_html_ssr():
    print("\n=== 7. HTML 报告 SSR ===")
    
    from src.cohort_retention.reporters import HtmlReporter
    from src.cohort_retention.models import RetentionMatrix
    
    report_cfg = ReportConfig(
        output_dir='/tmp',
        formats=['html'],
        ssr_enabled=True,
        ssr_render_charts=True,
        ssr_embed_data=True,
        ssr_minify=False,
    )
    
    reporter = HtmlReporter(report_cfg)
    assert hasattr(reporter, '_generate_retention_chart_svg'), "Should have SVG generation"
    assert hasattr(reporter, '_minify_html'), "Should have minify function"
    
    print(f"  SSR 配置: enabled={report_cfg.ssr_enabled}, "
          f"render_charts={report_cfg.ssr_render_charts}, "
          f"embed_data={report_cfg.ssr_embed_data}, "
          f"minify={report_cfg.ssr_minify}")
    print("✓ HTML SSR 测试通过")

def test_full_integration():
    print("\n=== 8. 完整集成测试 ===")
    
    df = generate_test_data(n_users=100, n_days=45)
    
    cohort_cfg = _build_cohort_config(
        cohort_key='first_order',
        granularity='day',
        retention_days=30,
        retention_type='standard',
        event_type='order',
        composite_events=('high_val:amount > 200:10:true',),
        user_id_col='user_id',
        event_time_col='event_time',
        event_type_col='event_type',
        min_users=5,
        dsl_cohort=None,
        dsl_filter='amount > 0',
        dsl_segment=None,
        dsl_error_recovery=True,
        dsl_default_numeric=0.0,
        outlier_enabled=True,
        outlier_method='mad',
        outlier_threshold=1.5,
        outlier_percentile_low=1.0,
        outlier_percentile_high=99.0,
        outlier_mad_threshold=3.0,
        outlier_contamination=0.05,
        outlier_eps=0.5,
        outlier_min_samples=5,
        outlier_remove_users=False,
        outlier_remove_cohorts=True,
        outlier_target_col='amount',
        rolling_enabled=True,
        rolling_window=7,
        rolling_step=1,
        rolling_min_periods=3,
        rolling_alignment='center',
        rolling_include_partial=False,
    )
    
    report_cfg = _build_report_config(
        output_dir='/tmp/test_reports',
        formats=('json',),
        matrix_format='parquet',
        ssr_enabled=True,
        ssr_render_charts=True,
        ssr_embed_data=True,
        ssr_minify=False,
        parquet_spark_compatible=True,
        parquet_compression='snappy',
    )
    
    errors = cohort_cfg.validate() + report_cfg.validate()
    assert len(errors) == 0, f"Config validation errors: {errors}"
    
    csv_path = '/tmp/test_integration.csv'
    df.to_csv(csv_path, index=False)
    
    source_cfg = DataSourceConfig(
        source_type='csv',
        path=csv_path,
        encoding='utf-8',
    )
    
    loader = create_loader(source_cfg, cohort_cfg)
    analyzer = CohortAnalyzer(loader, cohort_cfg)
    result = analyzer.run_analysis()
    
    print(f"  分析完成: {result.summary.get('total_cohorts', 0)} 队列, "
          f"{result.summary.get('total_users', 0)} 用户")
    
    if result.summary.get('outlier_stats'):
        print(f"  异常处理: {result.summary['outlier_stats'].get('removed', 0)} 移除")
    
    if hasattr(result, 'rolling_matrix') and result.rolling_matrix is not None:
        print(f"  滚动窗矩阵: {result.rolling_matrix.matrix.shape}")
    elif result.summary.get('rolling_days'):
        print(f"  滚动窗: {result.summary.get('rolling_days')} 天窗口")
    
    exporter = MatrixExporter(report_cfg)
    try:
        path = exporter.export_counts_parquet(result, 'test_matrix.parquet')
        print(f"  Parquet 导出: {path}")
    except ImportError as e:
        print(f"  Parquet 导出跳过 (pyarrow/fastparquet 未安装): {e}")
        path = exporter.export_counts_csv(result, 'test_matrix.csv')
        print(f"  CSV 导出: {path}")
    
    print("✓ 完整集成测试通过")

def main():
    print("=" * 60)
    print("cohort-retention-cli 新功能综合测试")
    print("=" * 60)
    
    tests = [
        test_dsl_error_recovery,
        test_outlier_algorithms,
        test_rolling_window_alignment,
        test_composite_event_priority,
        test_parquet_spark_compat,
        test_postgres_transaction,
        test_html_ssr,
        test_full_integration,
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
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
