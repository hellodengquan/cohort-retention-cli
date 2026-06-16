import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from cohort_retention.config import (
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
)
from cohort_retention.loaders import create_loader, CsvDataLoader, ParquetDataLoader
from cohort_retention.analyzer import CohortAnalyzer
from cohort_retention.reporters import MarkdownReporter, JsonReporter, HtmlReporter, MatrixExporter
from cohort_retention.dsl import DSLParser, CohortDSLEngine, DSLError


def generate_test_data(users=300, days=120, seed=42):
    np.random.seed(seed)
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
            "channel": np.random.choice(["app", "web", "wechat"], p=[0.4, 0.35, 0.25]),
            "region": np.random.choice(["north", "south", "east", "west"]),
        })

        num_orders = np.random.poisson(lam=4)
        for _ in range(num_orders):
            order_offset = np.random.randint(0, days - register_offset)
            order_date = register_date + timedelta(days=order_offset, hours=np.random.randint(0, 24))
            if order_date <= end_date:
                records.append({
                    "user_id": user_id,
                    "event_time": order_date,
                    "event_type": "order",
                    "amount": round(np.random.uniform(10, 500), 2),
                    "channel": np.random.choice(["app", "web", "wechat"], p=[0.5, 0.3, 0.2]),
                    "region": np.random.choice(["north", "south", "east", "west"]),
                })

        num_logins = np.random.poisson(lam=8)
        for _ in range(num_logins):
            login_offset = np.random.randint(0, days - register_offset)
            login_date = register_date + timedelta(days=login_offset, hours=np.random.randint(0, 24))
            if login_date <= end_date:
                records.append({
                    "user_id": user_id,
                    "event_time": login_date,
                    "event_type": "login",
                    "amount": 0,
                    "channel": np.random.choice(["app", "web", "wechat"], p=[0.6, 0.3, 0.1]),
                    "region": np.random.choice(["north", "south", "east", "west"]),
                })

    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def run_test(name, test_func):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    try:
        test_func()
        print(f"✅ {name} - 通过")
        return True
    except Exception as e:
        print(f"❌ {name} - 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quarter_granularity():
    df = generate_test_data(users=300, days=365, seed=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.QUARTER,
            retention_days=90,
            event_type=EventType.ORDER,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            min_users=5,
        )

        source_config = DataSourceConfig(
            source_type="csv",
            path=csv_path,
        )

        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        assert len(result.matrix.cohort_dates) > 0, "应至少有一个队列"
        print(f"  季粒度队列数: {len(result.matrix.cohort_dates)}")
        print(f"  队列日期: {[str(d) for d in result.matrix.cohort_dates[:3]]}")
        assert all(isinstance(d, object) for d in result.matrix.cohort_dates)


def test_rolling_retention():
    df = generate_test_data(users=200, days=90, seed=200)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=30,
            retention_type=RetentionType.ROLLING,
            event_type=EventType.ORDER,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            rolling=RollingWindowConfig(
                enabled=True,
                window_size=7,
                step=3,
                min_periods=3,
            ),
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        print(f"  滚动留存天数: {result.matrix.days}")
        assert 0 in result.matrix.days, "应包含第0天"
        assert 3 in result.matrix.days, "应包含第3天（步长3）"
        assert result.config["retention_type"] == "rolling"
        assert "rolling_window" in result.config


def test_outlier_detection():
    df = generate_test_data(users=300, days=60, seed=300)

    outlier_user = "user_9999"
    for i in range(50):
        df = pd.concat([df, pd.DataFrame([{
            "user_id": outlier_user,
            "event_time": datetime.now() - timedelta(days=i),
            "event_type": "order",
            "amount": 100,
            "channel": "app",
            "region": "north",
        }])], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=14,
            event_type=EventType.ORDER,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            outlier=OutlierConfig(
                enabled=True,
                method=OutlierMethod.IQR,
                threshold=1.5,
                remove_users=True,
                remove_cohorts=True,
            ),
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        outlier_stats = result.summary.get("outlier_stats", {})
        print(f"  异常值统计: {outlier_stats}")
        assert outlier_stats.get("removed", 0) > 0, "应检测到并移除异常用户"
        assert outlier_stats.get("method") == "iqr"


def test_composite_events():
    df = generate_test_data(users=200, days=60, seed=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=14,
            event_type=EventType.COMPOSITE,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            composite_events=[
                CompositeEventConfig(
                    name="active_user",
                    expression="(event_type == 'order') or (event_type == 'login')",
                    description="活跃用户事件（订单或登录）"
                ),
            ],
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        print(f"  复合事件分析完成，队列数: {len(result.matrix.cohort_dates)}")
        assert len(result.matrix.cohort_dates) > 0
        assert result.config["event_type"] == "composite"


def test_dsl_expressions():
    print("  测试 DSL 表达式解析...")

    test_df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "event_time": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-05", "2024-01-20"]),
        "event_type": ["register", "order", "register", "order"],
        "amount": [0, 150, 0, 300],
        "channel": ["app", "app", "web", "web"],
    })

    result1 = DSLParser.evaluate("amount > 100", {}, test_df)
    assert len(result1) == 4
    assert result1.tolist() == [False, True, False, True]

    result2 = DSLParser.evaluate("channel == 'app' and event_type == 'order'", {}, test_df)
    assert result2.tolist() == [False, True, False, False]

    result3 = DSLParser.evaluate("coalesce(amount, 0)", {}, test_df)
    assert result3.tolist() == [0, 150, 0, 300]

    result4 = DSLParser.evaluate("year(event_time)", {}, test_df)
    assert result4.tolist() == [2024, 2024, 2024, 2024]

    print("  测试 DSL 分群计算...")
    dsl_config = CohortDSLConfig(
        cohort_expression="group['event_time'].min()",
        filter_expression="channel != 'web'",
    )
    engine = CohortDSLEngine(dsl_config)

    filtered_df = engine.apply_filter(test_df)
    assert len(filtered_df) == 2, f"过滤后应有2条记录，实际{len(filtered_df)}"
    assert all(filtered_df["channel"] == "app")

    cohort_times = engine.compute_cohort_time(test_df, "user_id", "event_time")
    assert len(cohort_times) == 2
    assert "cohort_time" in cohort_times.columns

    valid, errors = engine.validate_expressions()
    assert valid, f"DSL验证失败: {errors}"

    print("  测试 DSL 语法验证...")
    bad_dsl = CohortDSLConfig(cohort_expression="invalid syntax [")
    bad_engine = CohortDSLEngine(bad_dsl)
    valid2, errors2 = bad_engine.validate_expressions()
    assert not valid2, "错误的DSL应验证失败"
    assert len(errors2) > 0

    print("  所有 DSL 测试通过")


def test_dsl_custom_cohort():
    df = generate_test_data(users=200, days=60, seed=500)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.CUSTOM,
            granularity=CohortGranularity.DAY,
            retention_days=14,
            event_type=EventType.LOGIN,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            dsl=CohortDSLConfig(
                cohort_expression="group['event_time'].min()",
                filter_expression="channel == 'app'",
            ),
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        print(f"  DSL自定义分群完成，队列数: {len(result.matrix.cohort_dates)}")
        print(f"  总用户数: {result.summary.get('total_users', 0)}")
        assert len(result.matrix.cohort_dates) > 0
        assert result.config["cohort_key"] == "custom"


def test_parquet_io():
    df = generate_test_data(users=200, days=60, seed=600)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_data.parquet")
        df.to_parquet(parquet_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=14,
            event_type=EventType.ORDER,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
        )

        source_config = DataSourceConfig(
            source_type="parquet",
            path=parquet_path,
        )

        loader = create_loader(source_config, cohort_config)
        loaded_df = loader.load_events()
        assert len(loaded_df) == len(df), "Parquet读取记录数不匹配"
        print(f"  Parquet 读取成功: {len(loaded_df)} 条记录")

        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()
        assert len(result.matrix.cohort_dates) > 0

        report_config = ReportConfig(output_dir=tmpdir, matrix_format="parquet")
        exporter = MatrixExporter(report_config)
        counts_path = exporter.export_counts_parquet(result)
        rates_path = exporter.export_rates_parquet(result)

        assert os.path.exists(counts_path), "Parquet计数矩阵导出失败"
        assert os.path.exists(rates_path), "Parquet比率矩阵导出失败"

        counts_df = pd.read_parquet(counts_path)
        print(f"  Parquet 矩阵导出成功: {counts_df.shape}")

        write_path = os.path.join(tmpdir, "write_test.parquet")
        loader2 = ParquetDataLoader(
            DataSourceConfig(source_type="parquet", path=write_path),
            cohort_config,
        )
        loader2.write_events(df.head(100))
        assert os.path.exists(write_path), "Parquet写入失败"
        print("  Parquet 写入成功")


def test_html_report():
    df = generate_test_data(users=200, days=60, seed=700)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.FIRST_ORDER,
            granularity=CohortGranularity.DAY,
            retention_days=14,
            event_type=EventType.ORDER,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        report_config = ReportConfig(output_dir=tmpdir)
        html_reporter = HtmlReporter(report_config)
        html_path = html_reporter.save(result)

        assert os.path.exists(html_path), "HTML报告生成失败"

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "留存队列分析报告" in content
        assert "热力图" in content
        assert "funnel-bar" in content
        print(f"  HTML 报告生成成功: {os.path.getsize(html_path)} 字节")

        md_reporter = MarkdownReporter(report_config)
        md_path = md_reporter.save(result)
        assert os.path.exists(md_path)

        json_reporter = JsonReporter(report_config)
        json_path = json_reporter.save(result)
        assert os.path.exists(json_path)
        print("  所有报告格式生成成功")


def test_all_formats_and_export():
    df = generate_test_data(users=150, days=60, seed=800)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        df.to_csv(csv_path, index=False)

        cohort_config = CohortConfig(
            cohort_key=CohortKey.REGISTER,
            granularity=CohortGranularity.WEEK,
            retention_days=28,
            event_type=EventType.LOGIN,
            user_id_col="user_id",
            event_time_col="event_time",
            event_type_col="event_type",
            min_users=3,
        )

        source_config = DataSourceConfig(source_type="csv", path=csv_path)
        loader = create_loader(source_config, cohort_config)
        analyzer = CohortAnalyzer(loader, cohort_config)
        result = analyzer.run_analysis()

        report_config = ReportConfig(
            output_dir=tmpdir,
            formats=["markdown", "json", "html"],
            matrix_format="parquet",
        )

        for fmt in report_config.formats:
            if fmt == "markdown":
                r = MarkdownReporter(report_config)
            elif fmt == "json":
                r = JsonReporter(report_config)
            elif fmt == "html":
                r = HtmlReporter(report_config)
            path = r.save(result)
            assert os.path.exists(path)
            print(f"  {fmt.upper()} 报告: {os.path.basename(path)}")

        exporter = MatrixExporter(report_config)
        counts_path = exporter.export_counts_parquet(result)
        rates_path = exporter.export_rates_parquet(result)
        assert os.path.exists(counts_path)
        assert os.path.exists(rates_path)
        print(f"  Parquet 矩阵导出成功")

        counts_csv = exporter.export_counts_csv(result)
        rates_csv = exporter.export_rates_csv(result)
        assert os.path.exists(counts_csv)
        assert os.path.exists(rates_csv)
        print(f"  CSV 矩阵导出成功")


def test_cli_commands():
    print("  测试 CLI 命令...")

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = os.path.join(tmpdir, "sample.csv")

        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "cohort_retention.cli", "list-functions"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"list-functions 失败: {result.stderr}"
        assert "min" in result.stdout
        assert "max" in result.stdout
        print("    list-functions: OK")

        result2 = subprocess.run(
            [sys.executable, "-m", "cohort_retention.cli", "validate-dsl",
             "--dsl-filter", "amount > 100"],
            capture_output=True, text=True
        )
        assert result2.returncode == 0, f"validate-dsl 失败: {result2.stderr}"
        assert "✅" in result2.stdout
        print("    validate-dsl: OK")

        result3 = subprocess.run(
            [sys.executable, "-m", "cohort_retention.cli", "gen-sample",
             sample_path, "--users", "50", "--days", "30"],
            capture_output=True, text=True
        )
        assert result3.returncode == 0, f"gen-sample 失败: {result3.stderr}"
        assert os.path.exists(sample_path)
        print("    gen-sample: OK")

        result4 = subprocess.run(
            [sys.executable, "-m", "cohort_retention.cli", "analyze",
             "--source-type", "csv",
             "--path", sample_path,
             "--event-type-col", "event_type",
             "--event-type", "order",
             "--retention-days", "7",
             "--output-dir", os.path.join(tmpdir, "reports"),
             "--format", "markdown",
             "--no-export-matrix"],
            capture_output=True, text=True
        )
        assert result4.returncode == 0, f"analyze 失败: {result4.stderr}"
        print("    analyze: OK")

    print("  所有 CLI 命令测试通过")


if __name__ == "__main__":
    tests = [
        ("季粒度支持", test_quarter_granularity),
        ("滚动留存窗", test_rolling_retention),
        ("异常值剔除", test_outlier_detection),
        ("复合事件类型", test_composite_events),
        ("自定义分群 DSL", test_dsl_expressions),
        ("DSL 自定义分群分析", test_dsl_custom_cohort),
        ("Parquet 读写", test_parquet_io),
        ("HTML 报告生成", test_html_report),
        ("多格式报告与导出", test_all_formats_and_export),
        ("CLI 命令", test_cli_commands),
    ]

    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print("开始综合测试")
    print(f"{'='*60}")

    for name, func in tests:
        if run_test(name, func):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
