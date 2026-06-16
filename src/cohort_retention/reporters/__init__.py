import os
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd

from ..models import CohortAnalysisResult
from ..config import ReportConfig


class MarkdownReporter:
    def __init__(self, config: ReportConfig):
        self.config = config

    def _fmt_pct(self, value: float) -> str:
        return f"{value * 100:.2f}%"

    def _fmt_int(self, value) -> str:
        return f"{int(value):,}"

    def generate(self, result: CohortAnalysisResult) -> str:
        lines = []
        lines.append("# 留存队列分析报告")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("## 1. 分析配置")
        lines.append("")
        if result.config:
            lines.append("| 配置项 | 值 |")
            lines.append("|--------|----|")
            lines.append(f"| 分群依据 | {result.config.get('cohort_key', '-')} |")
            lines.append(f"| 时间粒度 | {result.config.get('granularity', '-')} |")
            lines.append(f"| 留存天数 | {result.config.get('retention_days', '-')} |")
            lines.append(f"| 事件类型 | {result.config.get('event_type', '-')} |")
            lines.append(f"| 最小用户数 | {result.config.get('min_users', '-')} |")
        lines.append("")

        lines.append("## 2. 总体概览")
        lines.append("")
        if result.summary:
            lines.append("- **队列总数**: {}".format(result.summary.get("total_cohorts", 0)))
            lines.append("- **总用户数**: {}".format(result.summary.get("total_users", 0)))
            avg_ret = result.summary.get("average_retention", {})
            if avg_ret:
                lines.append("- **平均留存率**:")
                sorted_days = sorted(
                    avg_ret.items(),
                    key=lambda x: int(x[0].replace("day_", ""))
                )
                for day_key, rate in sorted_days:
                    day_num = day_key.replace("day_", "")
                    lines.append(f"  - 第 {day_num} 天: {self._fmt_pct(rate)}")
        lines.append("")

        lines.append("## 3. 留存漏斗")
        lines.append("")
        if result.funnel:
            lines.append("| 阶段 | 用户数 | 总转化率 | 环节流失率 |")
            lines.append("|------|--------|----------|------------|")
            for step in result.funnel.steps:
                lines.append(
                    f"| {step.name} | {self._fmt_int(step.user_count)} | "
                    f"{self._fmt_pct(step.conversion_rate)} | "
                    f"{self._fmt_pct(step.drop_off_rate)} |"
                )
        lines.append("")

        lines.append("## 4. 留存矩阵（人数）")
        lines.append("")
        lines.append("行：分群日期；列：距分群天数")
        lines.append("")

        matrix = result.matrix
        counts_df = matrix.retention_counts.copy()
        counts_df.index = [d.strftime("%Y-%m-%d") for d in matrix.cohort_dates]
        counts_df.columns = [f"Day {d}" for d in matrix.days]
        counts_df = counts_df.fillna("-")

        lines.append("| 分群日期 \\ 天数 | " + " | ".join(counts_df.columns) + " |")
        lines.append("|" + "|".join(["---"] * (len(counts_df.columns) + 1)) + "|")
        for idx, row in counts_df.iterrows():
            formatted = [str(int(v)) if isinstance(v, (int, float)) and not pd.isna(v) and v != "-" else str(v) for v in row]
            lines.append(f"| {idx} | " + " | ".join(formatted) + " |")
        lines.append("")

        lines.append("## 5. 留存矩阵（比例）")
        lines.append("")
        lines.append("行：分群日期；列：距分群天数；数值为留存率")
        lines.append("")

        rates_df = matrix.retention_rates.copy()
        rates_df.index = [d.strftime("%Y-%m-%d") for d in matrix.cohort_dates]
        rates_df.columns = [f"Day {d}" for d in matrix.days]

        lines.append("| 分群日期 \\ 天数 | " + " | ".join(rates_df.columns) + " |")
        lines.append("|" + "|".join(["---"] * (len(rates_df.columns) + 1)) + "|")
        for idx, row in rates_df.iterrows():
            formatted = [self._fmt_pct(v) if pd.notna(v) else "-" for v in row]
            lines.append(f"| {idx} | " + " | ".join(formatted) + " |")
        lines.append("")

        lines.append("## 6. 各队列规模")
        lines.append("")
        lines.append("| 分群日期 | 用户数 |")
        lines.append("|----------|--------|")
        for d in matrix.cohort_dates:
            size = matrix.cohort_sizes.get(d, 0)
            lines.append(f"| {d.strftime('%Y-%m-%d')} | {self._fmt_int(size)} |")
        lines.append("")

        return "\n".join(lines)

    def save(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cohort_report_{timestamp}.md"

        filepath = os.path.join(self.config.output_dir, filename)
        content = self.generate(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath


class JsonReporter:
    def __init__(self, config: ReportConfig):
        self.config = config

    def generate(self, result: CohortAnalysisResult) -> str:
        data = result.to_dict()
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    def save(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cohort_report_{timestamp}.json"

        filepath = os.path.join(self.config.output_dir, filename)
        content = self.generate(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath


class MatrixExporter:
    def __init__(self, config: ReportConfig):
        self.config = config

    def export_counts_csv(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_counts_{timestamp}.csv"

        filepath = os.path.join(self.config.output_dir, filename)
        df = result.matrix.retention_counts.copy()
        df.index = [d.strftime("%Y-%m-%d") for d in result.matrix.cohort_dates]
        df.columns = [f"day_{d}" for d in result.matrix.days]
        df.to_csv(filepath, encoding="utf-8")
        return filepath

    def export_rates_csv(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_rates_{timestamp}.csv"

        filepath = os.path.join(self.config.output_dir, filename)
        df = result.matrix.retention_rates.copy()
        df.index = [d.strftime("%Y-%m-%d") for d in result.matrix.cohort_dates]
        df.columns = [f"day_{d}" for d in result.matrix.days]
        df.to_csv(filepath, encoding="utf-8")
        return filepath


__all__ = ["MarkdownReporter", "JsonReporter", "MatrixExporter"]
