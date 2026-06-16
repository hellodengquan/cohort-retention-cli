import click
import sys
import os

from .config import (
    CohortConfig,
    DataSourceConfig,
    ReportConfig,
    CohortKey,
    CohortGranularity,
    EventType,
)
from .loaders import create_loader
from .analyzer import CohortAnalyzer
from .reporters import MarkdownReporter, JsonReporter, MatrixExporter


def _build_cohort_config(
    cohort_key: str,
    granularity: str,
    retention_days: int,
    event_type: str,
    user_id_col: str,
    event_time_col: str,
    event_type_col: str,
    min_users: int,
) -> CohortConfig:
    return CohortConfig(
        cohort_key=CohortKey(cohort_key),
        granularity=CohortGranularity(granularity),
        retention_days=retention_days,
        event_type=EventType(event_type),
        user_id_col=user_id_col,
        event_time_col=event_time_col,
        event_type_col=event_type_col if event_type_col else None,
        min_users=min_users,
    )


def _build_source_config(
    source_type: str,
    path: str,
    table: str,
    encoding: str,
) -> DataSourceConfig:
    return DataSourceConfig(
        source_type=source_type,
        path=path,
        table=table if table else None,
        encoding=encoding,
    )


def _build_report_config(
    output_dir: str,
    formats: tuple,
) -> ReportConfig:
    return ReportConfig(
        output_dir=output_dir,
        formats=list(formats),
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="cohort")
def cli():
    """留存队列分析 CLI - 按用户首单或注册时间分组，统计后续 N 天回流比例与漏斗"""
    pass


@cli.command()
@click.option("--source-type", type=click.Choice(["csv", "sqlite"]), required=True, help="数据源类型")
@click.option("--path", required=True, help="数据源文件路径")
@click.option("--table", default=None, help="SQLite 表名（仅 SQLite 需要）")
@click.option("--encoding", default="utf-8", help="CSV 文件编码")
@click.option("--cohort-key", type=click.Choice(["first_order", "register"]), default="first_order", help="分群依据")
@click.option("--granularity", type=click.Choice(["day", "week", "month"]), default="day", help="时间粒度")
@click.option("--retention-days", default=30, type=int, help="统计留存的天数")
@click.option("--event-type", type=click.Choice(["order", "login", "active", "pay"]), default="order", help="回流事件类型")
@click.option("--user-id-col", default="user_id", help="用户ID列名")
@click.option("--event-time-col", default="event_time", help="事件时间列名")
@click.option("--event-type-col", default=None, help="事件类型列名")
@click.option("--min-users", default=1, type=int, help="队列最小用户数")
@click.option("--output-dir", default="./reports", help="报告输出目录")
@click.option("--format", "formats", type=click.Choice(["markdown", "json"]), multiple=True, default=["markdown", "json"], help="报告格式")
@click.option("--export-matrix/--no-export-matrix", default=True, help="是否导出留存矩阵 CSV")
def analyze(
    source_type,
    path,
    table,
    encoding,
    cohort_key,
    granularity,
    retention_days,
    event_type,
    user_id_col,
    event_time_col,
    event_type_col,
    min_users,
    output_dir,
    formats,
    export_matrix,
):
    """运行留存队列分析并生成报告"""

    try:
        cohort_config = _build_cohort_config(
            cohort_key=cohort_key,
            granularity=granularity,
            retention_days=retention_days,
            event_type=event_type,
            user_id_col=user_id_col,
            event_time_col=event_time_col,
            event_type_col=event_type_col,
            min_users=min_users,
        )

        source_config = _build_source_config(
            source_type=source_type,
            path=path,
            table=table,
            encoding=encoding,
        )

        report_config = _build_report_config(
            output_dir=output_dir,
            formats=formats,
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

        if export_matrix:
            exporter = MatrixExporter(report_config)
            counts_path = exporter.export_counts_csv(result)
            rates_path = exporter.export_rates_csv(result)
            generated_files.extend([counts_path, rates_path])
            click.echo(f"  - 留存人数矩阵: {counts_path}")
            click.echo(f"  - 留存率矩阵: {rates_path}")

        click.echo("")
        click.echo("全部文件已生成 ✅")

    except Exception as e:
        click.echo(f"[错误] 分析失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("output_path", default="./sample_data.csv")
@click.option("--users", default=200, type=int, help="用户数量")
@click.option("--days", default=60, type=int, help="数据跨度天数")
def gen_sample(output_path, users, days):
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

    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    click.echo(f"✅ 已生成 {len(df)} 条记录，共 {users} 个用户")
    click.echo(f"   时间范围: {start_date.date()} ~ {end_date.date()}")


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


if __name__ == "__main__":
    cli()
