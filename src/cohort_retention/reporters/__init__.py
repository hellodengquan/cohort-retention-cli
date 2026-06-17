import os
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from ..models import CohortAnalysisResult, RetentionMatrix
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
            for k, v in result.config.items():
                if isinstance(v, dict):
                    v_str = json.dumps(v, ensure_ascii=False)
                else:
                    v_str = str(v)
                lines.append(f"| {k} | {v_str} |")
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
            outlier_stats = result.summary.get("outlier_stats")
            if outlier_stats:
                lines.append("- **异常值统计**:")
                lines.append(f"  - 检测方法: {outlier_stats.get('method', '-')}")
                lines.append(f"  - 移除用户数: {outlier_stats.get('removed', 0)}")
                if "cohorts_removed" in outlier_stats:
                    lines.append(f"  - 移除队列数: {outlier_stats.get('cohorts_removed', 0)}")
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


class HtmlReporter:
    def __init__(self, config: ReportConfig):
        self.config = config

    def _fmt_pct(self, value: float) -> str:
        return f"{value * 100:.2f}%"

    def _fmt_int(self, value) -> str:
        return f"{int(value):,}"

    def _get_heatmap_color(self, value: float) -> str:
        if pd.isna(value):
            return "#f5f5f5"
        v = max(0.0, min(1.0, float(value)))
        r = int(255 * (1 - v))
        g = int(200 * v + 55)
        b = int(150 * v + 50)
        return f"rgb({r}, {g}, {b})"

    def _generate_retention_chart_svg(self, matrix: RetentionMatrix) -> str:
        if not self.config.ssr_render_charts:
            return ""

        dates = matrix.cohort_dates
        days = matrix.days
        rates = matrix.retention_rates

        width = 800
        height = 400
        padding = {"left": 60, "right": 20, "top": 40, "bottom": 60}
        chart_w = width - padding["left"] - padding["right"]
        chart_h = height - padding["top"] - padding["bottom"]

        svg_parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
        svg_parts.append(f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">各队列留存曲线</text>')

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

        max_day = max(days) if days else 1
        x_step = chart_w / max_day

        svg_parts.append(f'<line x1="{padding["left"]}" y1="{padding["top"]+chart_h}" x2="{padding["left"]+chart_w}" y2="{padding["top"]+chart_h}" stroke="#ccc" stroke-width="1"/>')
        svg_parts.append(f'<line x1="{padding["left"]}" y1="{padding["top"]}" x2="{padding["left"]}" y2="{padding["top"]+chart_h}" stroke="#ccc" stroke-width="1"/>')

        for i in range(0, 101, 20):
            y = padding["top"] + chart_h - (i / 100 * chart_h)
            svg_parts.append(f'<line x1="{padding["left"]}" y1="{y}" x2="{padding["left"]+chart_w}" y2="{y}" stroke="#eee" stroke-width="1"/>')
            svg_parts.append(f'<text x="{padding["left"]-5}" y="{y+4}" text-anchor="end" font-size="11" fill="#666">{i}%</text>')

        for d in [0, 7, 14, 30, 60, 90]:
            if d <= max_day:
                x = padding["left"] + (d / max_day * chart_w)
                svg_parts.append(f'<text x="{x}" y="{padding["top"]+chart_h+20}" text-anchor="middle" font-size="11" fill="#666">Day {d}</text>')

        for i, (date, row) in enumerate(rates.iterrows()):
            color = colors[i % len(colors)]
            points = []
            for day in days:
                if day in row.index and pd.notna(row[day]):
                    x = padding["left"] + (day / max_day * chart_w)
                    y = padding["top"] + chart_h - (row[day] * chart_h)
                    points.append(f"{x},{y}")

            if points:
                date_str = dates[i].strftime("%Y-%m-%d") if i < len(dates) else str(i)
                svg_parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
                for p in points:
                    x, y = p.split(",")
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')

                legend_y = padding["top"] + i * 20
                svg_parts.append(f'<rect x="{padding["left"]+10}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
                svg_parts.append(f'<text x="{padding["left"]+28}" y="{legend_y+10}" font-size="11" fill="#333">{date_str}</text>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _generate_funnel_svg(self, result: CohortAnalysisResult) -> str:
        if not self.config.ssr_render_charts or not result.funnel:
            return ""

        width = 600
        height = 300
        padding = {"left": 20, "right": 20, "top": 40, "bottom": 40}

        svg_parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
        svg_parts.append(f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">留存漏斗图</text>')

        colors = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#9b59b6", "#8e44ad", "#e67e22", "#d35400"]
        steps = result.funnel.steps
        max_count = max(s.user_count for s in steps) if steps else 1
        bar_h = (height - padding["top"] - padding["bottom"]) / len(steps)
        bar_w_max = width - padding["left"] - padding["right"]

        for i, step in enumerate(steps):
            color = colors[i % len(colors)]
            bar_w = (step.user_count / max_count) * bar_w_max
            x = padding["left"]
            y = padding["top"] + i * bar_h + 5

            svg_parts.append(f'<rect x="{x}" y="{y}" width="{max(bar_w, 10)}" height="{bar_h-10}" fill="{color}" rx="4"/>')
            svg_parts.append(f'<text x="{x+10}" y="{y+bar_h/2}" font-size="12" fill="white" font-weight="bold">{step.name}</text>')
            svg_parts.append(f'<text x="{x+bar_w+10}" y="{y+bar_h/2+4}" font-size="11" fill="#333">{self._fmt_int(step.user_count)} ({self._fmt_pct(step.conversion_rate)})</text>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _embed_data_json(self, result: CohortAnalysisResult) -> str:
        if not self.config.ssr_embed_data:
            return ""
        data = result.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        escaped_json = json_str.replace("</script>", "<\\/script>")
        return f'<script id="cohort-data" type="application/json">{escaped_json}</script>'

    def _minify_html(self, html: str) -> str:
        if not self.config.ssr_minify:
            return html
        import re
        html = re.sub(r">\s+<", "><", html)
        html = re.sub(r"\s+", " ", html)
        return html.strip()

    def _generate_html_head(self, result: CohortAnalysisResult) -> str:
        seo = self.config.seo
        parts = []
        parts.append(f'<meta charset="UTF-8">')
        parts.append(f'<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        parts.append(f'<meta name="description" content="{seo.description}">')
        parts.append(f'<meta name="keywords" content="{",".join(seo.keywords)}">')
        parts.append(f'<meta name="author" content="{seo.author}">')
        parts.append(f'<meta name="robots" content="{seo.robots}">')
        if seo.canonical_url:
            parts.append(f'<link rel="canonical" href="{seo.canonical_url}">')
        parts.append(f'<meta property="og:type" content="{seo.og_type}">')
        parts.append(f'<meta property="og:title" content="{seo.title}">')
        parts.append(f'<meta property="og:description" content="{seo.description}">')
        parts.append(f'<meta property="og:locale" content="{seo.language}">')
        if seo.og_image:
            parts.append(f'<meta property="og:image" content="{seo.og_image}">')
        parts.append(f'<title>{seo.title}</title>')
        return "\n".join(parts)

    def _generate_structured_data(self, result: CohortAnalysisResult) -> str:
        seo = self.config.seo
        if not seo.enable_structured_data:
            return ""

        summary = result.summary or {}
        sd = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": seo.title,
            "description": seo.description,
            "author": {"@type": "Organization", "name": seo.author},
            "inLanguage": seo.language,
            "datePublished": datetime.now().strftime("%Y-%m-%d"),
            "dateModified": datetime.now().strftime("%Y-%m-%d"),
            "keywords": seo.keywords,
        }
        total_users = summary.get("total_users", 0)
        total_cohorts = summary.get("total_cohorts", 0)
        if total_users or total_cohorts:
            sd["about"] = {
                "@type": "Dataset",
                "name": "留存分析数据",
                "description": f"包含 {total_cohorts} 个队列，共 {total_users} 个用户的留存分析数据",
                "variableMeasured": ["用户留存率", "队列规模", "漏斗转化率"],
            }
        import json as _json
        escaped = _json.dumps(sd, ensure_ascii=False).replace("</script>", "<\\/script>")
        return f'<script type="application/ld+json">{escaped}</script>'

    def generate(self, result: CohortAnalysisResult) -> str:
        matrix = result.matrix
        config = result.config or {}
        summary = result.summary or {}

        counts_df = matrix.retention_counts.copy()
        counts_df.index = [d.strftime("%Y-%m-%d") for d in matrix.cohort_dates]
        counts_df.columns = [f"Day {d}" for d in matrix.days]

        rates_df = matrix.retention_rates.copy()
        rates_df.index = [d.strftime("%Y-%m-%d") for d in matrix.cohort_dates]
        rates_df.columns = [f"Day {d}" for d in matrix.days]

        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append(f'<html lang="{self.config.seo.language}">')
        html_parts.append("<head>")
        html_parts.append(self._generate_html_head(result))
        html_parts.append(self._generate_structured_data(result))
        html_parts.append("<style>")
        html_parts.append("""
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; background: #fafafa; color: #333; }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #2c3e50; margin-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
            .metadata { color: #7f8c8d; margin-bottom: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            th, td { padding: 10px 12px; text-align: right; border: 1px solid #ecf0f1; }
            th { background: #3498db; color: white; font-weight: 600; }
            tr:nth-child(even) { background: #f8f9fa; }
            tr:hover { background: #e8f4f8; }
            td:first-child, th:first-child { text-align: left; font-weight: 600; background: #f8f9fa; position: sticky; left: 0; z-index: 10; }
            .heatmap-cell { font-weight: 500; }
            .funnel-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
            .funnel-bar { height: 40px; display: flex; align-items: center; padding: 0 15px; color: white; font-weight: 600; border-radius: 4px; transition: all 0.3s ease; }
            .funnel-bar:hover { transform: translateX(5px); }
            .funnel-meta { margin-left: auto; font-size: 0.9em; opacity: 0.9; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #3498db; }
            .stat-label { font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; }
            .stat-value { font-size: 1.8em; font-weight: 700; color: #2c3e50; }
            .config-table { max-width: 600px; }
            .table-wrapper { overflow-x: auto; }
        """)
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append('<div class="container" role="main">')
        html_parts.append(f"<header><h1>{self.config.seo.title}</h1></header>")
        html_parts.append(f'<p class="metadata">生成时间: <time datetime="{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</time></p>')
        html_parts.append(f'<meta itemprop="datePublished" content="{datetime.now().strftime("%Y-%m-%d")}">')

        html_parts.append("<h2>1. 分析配置</h2>")
        html_parts.append('<table class="config-table">')
        html_parts.append("<tr><th>配置项</th><th>值</th></tr>")
        for k, v in config.items():
            if isinstance(v, dict):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            html_parts.append(f"<tr><td>{k}</td><td>{v_str}</td></tr>")
        html_parts.append("</table>")

        html_parts.append("<h2>2. 总体概览</h2>")
        html_parts.append('<div class="stats-grid">')
        html_parts.append(f'<div class="stat-card"><div class="stat-label">队列总数</div><div class="stat-value">{summary.get("total_cohorts", 0)}</div></div>')
        html_parts.append(f'<div class="stat-card"><div class="stat-label">总用户数</div><div class="stat-value">{self._fmt_int(summary.get("total_users", 0))}</div></div>')
        avg_ret = summary.get("average_retention", {})
        if avg_ret:
            sorted_days = sorted(avg_ret.items(), key=lambda x: int(x[0].replace("day_", "")))
            key_days = [0, 1, 3, 7, 14, 30]
            for day_key, rate in sorted_days:
                day_num = int(day_key.replace("day_", ""))
                if day_num in key_days:
                    html_parts.append(f'<div class="stat-card"><div class="stat-label">第 {day_num} 天留存</div><div class="stat-value">{self._fmt_pct(rate)}</div></div>')
        html_parts.append("</div>")

        outlier_stats = summary.get("outlier_stats")
        if outlier_stats:
            html_parts.append("<h3>异常值统计</h3>")
            html_parts.append('<div class="stats-grid">')
            html_parts.append(f'<div class="stat-card"><div class="stat-label">检测方法</div><div class="stat-value" style="font-size: 1.2em;">{outlier_stats.get("method", "-")}</div></div>')
            html_parts.append(f'<div class="stat-card"><div class="stat-label">移除用户数</div><div class="stat-value">{outlier_stats.get("removed", 0)}</div></div>')
            if "cohorts_removed" in outlier_stats:
                html_parts.append(f'<div class="stat-card"><div class="stat-label">移除队列数</div><div class="stat-value">{outlier_stats.get("cohorts_removed", 0)}</div></div>')
            html_parts.append("</div>")

        html_parts.append("<h2>3. 留存漏斗</h2>")
        html_parts.append('<div class="funnel-container">')
        if result.funnel:
            colors = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#9b59b6", "#8e44ad", "#e67e22", "#d35400"]
            for i, step in enumerate(result.funnel.steps):
                color = colors[i % len(colors)]
                pct = step.conversion_rate * 100
                html_parts.append(f'<div class="funnel-bar" style="width: {max(pct, 5)}%; background: {color};">')
                html_parts.append(f"<span>{step.name}</span>")
                html_parts.append(f'<span class="funnel-meta">{self._fmt_int(step.user_count)} 用户 | {self._fmt_pct(step.conversion_rate)}</span>')
                html_parts.append("</div>")
        html_parts.append("</div>")

        html_parts.append("<h2>4. 留存矩阵（人数）</h2>")
        html_parts.append('<div class="table-wrapper">')
        html_parts.append("<table>")
        html_parts.append("<tr><th>分群日期 \\ 天数</th>")
        for col in counts_df.columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr>")
        for idx, row in counts_df.iterrows():
            html_parts.append(f"<tr><td>{idx}</td>")
            for v in row:
                display_v = str(int(v)) if isinstance(v, (int, float)) and not pd.isna(v) and v != "-" else "-"
                html_parts.append(f"<td>{display_v}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")
        html_parts.append("</div>")

        html_parts.append("<h2>5. 留存矩阵（热力图）</h2>")
        html_parts.append('<div class="table-wrapper">')
        html_parts.append("<table>")
        html_parts.append("<tr><th>分群日期 \\ 天数</th>")
        for col in rates_df.columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr>")
        for idx, row in rates_df.iterrows():
            html_parts.append(f"<tr><td>{idx}</td>")
            for v in row:
                color = self._get_heatmap_color(v)
                display_v = self._fmt_pct(v) if pd.notna(v) else "-"
                html_parts.append(f'<td class="heatmap-cell" style="background: {color};">{display_v}</td>')
            html_parts.append("</tr>")
        html_parts.append("</table>")
        html_parts.append("</div>")

        if self.config.ssr_enabled:
            retention_svg = self._generate_retention_chart_svg(matrix)
            if retention_svg:
                html_parts.append("<h2>6. 留存曲线图（SSR）</h2>")
                html_parts.append('<div class="chart-container">')
                html_parts.append(retention_svg)
                html_parts.append("</div>")

            funnel_svg = self._generate_funnel_svg(result)
            if funnel_svg:
                html_parts.append("<h2>7. 漏斗图（SSR）</h2>")
                html_parts.append('<div class="chart-container">')
                html_parts.append(funnel_svg)
                html_parts.append("</div>")

        html_parts.append("<h2>8. 各队列规模</h2>" if self.config.ssr_enabled else "<h2>6. 各队列规模</h2>")
        html_parts.append('<table style="max-width: 400px;">')
        html_parts.append("<tr><th>分群日期</th><th>用户数</th></tr>")
        for d in matrix.cohort_dates:
            size = matrix.cohort_sizes.get(d, 0)
            html_parts.append(f"<tr><td>{d.strftime('%Y-%m-%d')}</td><td>{self._fmt_int(size)}</td></tr>")
        html_parts.append("</table>")

        if self.config.ssr_enabled:
            data_script = self._embed_data_json(result)
            if data_script:
                html_parts.append(data_script)

        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        content = "\n".join(html_parts)
        if self.config.ssr_minify:
            content = self._minify_html(content)
        return content

    def save(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cohort_report_{timestamp}.html"

        filepath = os.path.join(self.config.output_dir, filename)
        content = self.generate(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath


class MatrixExporter:
    def __init__(self, config: ReportConfig):
        self.config = config

    def _prepare_df(self, result: CohortAnalysisResult, is_counts: bool) -> pd.DataFrame:
        if is_counts:
            df = result.matrix.retention_counts.copy()
        else:
            df = result.matrix.retention_rates.copy()
        df.index = [d.strftime("%Y-%m-%d") for d in result.matrix.cohort_dates]
        df.columns = [f"day_{d}" for d in result.matrix.days]
        df.index.name = "cohort_date"
        return df

    def export_counts_csv(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_counts_{timestamp}.csv"
        filepath = os.path.join(self.config.output_dir, filename)
        df = self._prepare_df(result, is_counts=True)
        df.to_csv(filepath, encoding="utf-8")
        return filepath

    def export_rates_csv(self, result: CohortAnalysisResult, filename: Optional[str] = None) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_rates_{timestamp}.csv"
        filepath = os.path.join(self.config.output_dir, filename)
        df = self._prepare_df(result, is_counts=False)
        df.to_csv(filepath, encoding="utf-8")
        return filepath

    def _prepare_parquet_df(self, df: pd.DataFrame, spark_compatible: bool) -> pd.DataFrame:
        if not spark_compatible:
            return df

        df = df.copy()
        parquet_cfg = self.config.parquet

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if parquet_cfg.coerce_timestamps == "ms":
                    df[col] = df[col].astype("datetime64[ms]")
                elif parquet_cfg.coerce_timestamps == "us":
                    df[col] = df[col].astype("datetime64[us]")

            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = df[col].astype(str)
                except Exception:
                    pass

        return df

    def export_counts_parquet(
        self,
        result: CohortAnalysisResult,
        filename: Optional[str] = None,
        spark_compatible: Optional[bool] = None,
    ) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_counts_{timestamp}.parquet"
        filepath = os.path.join(self.config.output_dir, filename)
        df = self._prepare_df(result, is_counts=True)

        use_spark = spark_compatible if spark_compatible is not None else self.config.parquet.spark_compatible
        if use_spark:
            df = self._prepare_parquet_df(df, use_spark)

        parquet_cfg = self.config.parquet
        kwargs = {
            "compression": parquet_cfg.compression,
            "use_deprecated_int96_timestamps": parquet_cfg.use_deprecated_int96_timestamps,
            "coerce_timestamps": parquet_cfg.coerce_timestamps,
        }
        if parquet_cfg.row_group_size:
            kwargs["row_group_size"] = parquet_cfg.row_group_size

        df.to_parquet(filepath, **kwargs)
        return filepath

    def export_rates_parquet(
        self,
        result: CohortAnalysisResult,
        filename: Optional[str] = None,
        spark_compatible: Optional[bool] = None,
    ) -> str:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retention_rates_{timestamp}.parquet"
        filepath = os.path.join(self.config.output_dir, filename)
        df = self._prepare_df(result, is_counts=False)

        use_spark = spark_compatible if spark_compatible is not None else self.config.parquet.spark_compatible
        if use_spark:
            df = self._prepare_parquet_df(df, use_spark)

        parquet_cfg = self.config.parquet
        kwargs = {
            "compression": parquet_cfg.compression,
            "use_deprecated_int96_timestamps": parquet_cfg.use_deprecated_int96_timestamps,
            "coerce_timestamps": parquet_cfg.coerce_timestamps,
        }
        if parquet_cfg.row_group_size:
            kwargs["row_group_size"] = parquet_cfg.row_group_size

        df.to_parquet(filepath, **kwargs)
        return filepath


__all__ = ["MarkdownReporter", "JsonReporter", "HtmlReporter", "MatrixExporter"]
