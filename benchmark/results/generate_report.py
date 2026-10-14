#!/usr/bin/env python3
"""Build a Chinese Markdown report and SVG charts from vLLM c*.json results.

Usage: python generate_report.py
Uses only the Python standard library; all paths are relative to this script.
"""
from pathlib import Path
from datetime import datetime
import argparse
from collections import Counter
import html
import json
import math

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / 'charts'
COLORS = ['#2563eb', '#ea580c', '#059669', '#9333ea']


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def chart(filename, title, panels, data):
    """Draw line plots with numeric, evenly scaled concurrency coordinates."""
    width, panel_height = 1120, 365
    height = 80 + panel_height * len(panels)
    models = ', '.join(sorted({r['model_id'] for r in data}))
    counts = ', '.join(str(n) for n in sorted({r['num_prompts'] for r in data}))
    outcomes = 'Failures present: interpret with care' if any(r['failed'] for r in data) else 'All requests succeeded in these runs'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        '<desc id="desc">Line charts of vLLM benchmark metrics versus configured maximum concurrency. Each marker represents one benchmark run. Lines connect observations only.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#334155} .heading{font-size:23px;font-weight:700} .label{font-size:13px} .tick{font-size:12px;fill:#64748b} .value{font-size:11px;font-weight:600}</style>',
        f'<text x="38" y="39" class="heading">{esc(title)}</text>',
        f'<text x="38" y="61" class="tick">{esc(models)} | {esc(counts)} requests per run | {esc(outcomes)}</text>',
    ]
    xs = [row['max_concurrency'] for row in data]
    xmin, xmax = min(xs), max(xs)
    for i, (subtitle, unit, series, fixed_max) in enumerate(panels):
        base = 80 + i * panel_height
        left, right, top, bottom = 90, 1060, base + 62, base + 295
        values = [float(row[key]) for key, label in series for row in data]
        raw_max = max(values) * 1.16 if values else 1
        if fixed_max is not None:
            ymax = fixed_max
        else:
            magnitude = 10 ** math.floor(math.log10(raw_max or 1))
            ymax = math.ceil(raw_max / magnitude * 2) / 2 * magnitude or 1
        xcoord = lambda x: left + (x - xmin) / (xmax - xmin or 1) * (right - left)
        ycoord = lambda y: bottom - y / ymax * (bottom - top)
        parts.append(f'<text x="38" y="{base + 24}" font-size="17" font-weight="600">{esc(subtitle)}</text>')
        parts.append(f'<text x="{left}" y="{top - 13}" class="tick">{esc(unit)}</text>')
        for tick in range(6):
            val = ymax * tick / 5
            y = ycoord(val)
            label = f'{val:,.0f}' if ymax >= 20 else f'{val:.2f}'
            parts.extend([
                f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e2e8f0"/>',
                f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" class="tick">{label}</text>',
            ])
        for x in xs:
            xp = xcoord(x)
            parts.extend([
                f'<line x1="{xp}" y1="{top}" x2="{xp}" y2="{bottom}" stroke="#e2e8f0" stroke-dasharray="3 5"/>',
                f'<text x="{xp}" y="{bottom + 22}" text-anchor="middle" class="tick">{x}</text>',
            ])
        parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 49}" text-anchor="middle" class="label">Configured maximum concurrency (requests)</text>')
        for j, (key, label) in enumerate(series):
            color = COLORS[j % len(COLORS)]
            lx = 430 + j * 210
            parts.extend([
                f'<line x1="{lx}" y1="{base + 20}" x2="{lx + 23}" y2="{base + 20}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx + 31}" y="{base + 24}" class="label">{esc(label)}</text>',
            ])
            points = ' '.join(f'{xcoord(row["max_concurrency"]):.2f},{ycoord(row[key]):.2f}' for row in data)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
            for row in data:
                x, y = xcoord(row['max_concurrency']), ycoord(row[key])
                parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}" stroke="white" stroke-width="1.5"><title>Concurrency {row["max_concurrency"]}: {esc(label)} = {row[key]:.4f} {esc(unit)}</title></circle>')
                if len(series) == 1:
                    parts.append(f'<text x="{x}" y="{y - 12}" text-anchor="middle" class="value">{fmt(row[key])}</text>')
    parts.append('</svg>')
    (CHARTS / filename).write_text('\n'.join(parts), encoding='utf-8')


def main():
    global CHARTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=ROOT,
                        help='Read results and write the report/charts in this directory')
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    CHARTS = result_dir / 'charts'
    sources = sorted(p for p in result_dir.glob('c[0-9]*.json') if p.stem[1:].isdigit())
    if not sources:
        raise SystemExit('No c<number>.json benchmark files found.')
    data = []
    for path in sources:
        row = json.loads(path.read_text(encoding='utf-8'))
        row['_source'] = path.name
        row['success_rate'] = 100 * row['completed'] / row['num_prompts']
        row['failure_rate'] = 100 * row['failed'] / row['num_prompts']
        data.append(row)
    data.sort(key=lambda row: row['max_concurrency'])
    CHARTS.mkdir(exist_ok=True)
    chart('throughput.svg', 'Throughput vs. concurrency', [
        ('Successful request throughput', 'requests/s', [('request_throughput', 'Request throughput')], None),
        ('Token throughput', 'tokens/s', [('output_throughput', 'Output tokens'), ('total_token_throughput', 'Input + output tokens')], None),
    ], data)
    chart('ttft.svg', 'Time to first token (TTFT)', [
        ('TTFT distribution summary', 'ms', [('mean_ttft_ms', 'Mean'), ('median_ttft_ms', 'Median'), ('p99_ttft_ms', 'P99')], None),
    ], data)
    chart('decode_latency.svg', 'Generation latency vs. concurrency', [
        ('Time per output token (TPOT)', 'ms/token', [('mean_tpot_ms', 'Mean'), ('median_tpot_ms', 'Median'), ('p99_tpot_ms', 'P99')], None),
        ('Inter-token latency (ITL)', 'ms', [('mean_itl_ms', 'Mean'), ('median_itl_ms', 'Median'), ('p99_itl_ms', 'P99')], None),
    ], data)
    chart('success_rate.svg', 'Request outcomes vs. concurrency', [
        ('Success / failure rate', '%', [('success_rate', 'Success rate'), ('failure_rate', 'Failure rate')], 100),
    ], data)
    chart('duration.svg', 'Total benchmark duration vs. concurrency', [
        ('Elapsed time for the complete run', 's', [('duration', 'Benchmark duration')], None),
    ], data)

    first = data[0]
    best = max(data, key=lambda r: r['output_throughput'])
    failed_groups = [r for r in data if r['failed']]
    intro = (
        f'> 本报告根据同目录下的原始 JSON 结果自动整理。{len(failed_groups)} 组有请求失败，'
        '以下吞吐量和延迟应结合成功率阅读；当前数据不足以确定服务的稳定容量或最佳并发数。'
        if failed_groups else
        '> 本报告根据同目录下的新实验 JSON 结果自动整理。本次各组请求全部成功；'
        '每组仅运行一次，结果反映本次工作负载的观测性能。'
    )
    observations = [
        f"- 并发 **{best['max_concurrency']}** 时，本次记录的输出吞吐量最高，为 "
        f"**{fmt(best['output_throughput'])} token/s**，请求吞吐量为 "
        f"**{fmt(best['request_throughput'], 4)} req/s**，成功率为 "
        f"**{fmt(best['success_rate'])}%（{best['completed']}/{best['num_prompts']}）**。",
        f"- 该组平均 TTFT 为 **{fmt(best['mean_ttft_ms'])} ms**，平均 TPOT 为 "
        f"**{fmt(best['mean_tpot_ms'])} ms**，平均 ITL 为 **{fmt(best['mean_itl_ms'])} ms**。",
        '- 各档失败率：' + '；'.join(
            f"并发 {r['max_concurrency']} 为 {fmt(r['failure_rate'])}%" for r in data
        ) + '。',
    ]
    sparse = [str(r['max_concurrency']) for r in data if r['completed'] <= 1]
    if sparse:
        observations.append('- 并发 ' + '、'.join(sparse) + ' 的成功请求不超过 1 个，延迟统计样本不足。')
    if failed_groups:
        observations.append('- 有请求失败，不能据此认定服务已稳定；应先核对下方错误摘要与完整日志。')
    else:
        observations.append('- 各档位均无失败；每档请求数较少，建议增加样本量与重复次数后再判断稳定容量。')
    error_rows = []
    for row in data:
        if 'errors' not in row:
            error_rows.append([row['max_concurrency'], '未保存 errors；无法恢复历史错误原因', '—'])
        else:
            counts = Counter(error for error in row['errors'] if error)
            if not counts:
                error_rows.append([row['max_concurrency'], '无错误', 0])
            for message, count in counts.most_common():
                summary = message.strip().splitlines()[-1][:400].replace('|', '&#124;')
                error_rows.append([row['max_concurrency'], summary, count])
    sections = [
        '# vLLM 不同并发数 Benchmark 测试报告',
        intro,
        '## 1. 测试配置与统计口径',
        table(['项目', '内容'], [
            ['模型', '`' + str(first['model_id']) + '`'],
            ['Tokenizer', '`' + str(first['tokenizer_id']) + '`'],
            ['接口 / Backend', f"`{first['endpoint_type']}` / `{first['backend']}`"],
            ['每组请求数', '、'.join(str(n) for n in sorted({r['num_prompts'] for r in data}))],
            ['并发上限', '、'.join(str(r['max_concurrency']) for r in data)],
            ['请求速率 request_rate', '、'.join(str(n) for n in sorted({str(r['request_rate']) for r in data})) + '（不限制发起速率，受并发上限约束）'],
            ['Burstiness', '、'.join(str(n) for n in sorted({r['burstiness'] for r in data}))],
        ]),
        '\n'.join([
            '- 横轴取 JSON 的 `max_concurrency`（配置的并发上限），按数值从小到大排列。',
            '- 成功率 = `completed / num_prompts × 100%`；失败率 = `failed / num_prompts × 100%`。',
            '- 请求吞吐量 = 成功请求数 / 整组测试耗时；输出吞吐量 = 输出 token 总数 / 整组测试耗时；总 token 吞吐量 = 输入与输出 token 总数之和 / 整组测试耗时。',
            '- TTFT 为首 token 延迟；TPOT 为首 token 之后的平均每输出 token 耗时；ITL 为相邻 token 的到达间隔。延迟表直接采用 benchmark 输出的统计字段，不将失败请求的零值延迟纳入自行计算。',
            '- 延迟的 Mean / Median / Std / P99 分别为均值、中位数、标准差和第 99 百分位；ITL 与请求级指标的样本粒度不同。',
            '- 表格保留 2 位小数，请求吞吐量保留 4 位；折线图使用未舍入的原始数值。折线仅连接本次观测点，不表示中间并发数的实测结果。',
            '- 日期按 JSON 中的 `date` 原样解析，源文件未注明时区。每个并发数仅有一次测试，图中不包含重复实验的误差条。',
        ]),
        '## 2. 请求完成情况',
        table(['并发上限', '请求数', '成功', '失败', '成功率', '失败率', '测试耗时 (s)', '源文件时间'], [
            [r['max_concurrency'], r['num_prompts'], r['completed'], r['failed'], fmt(r['success_rate']) + '%', fmt(r['failure_rate']) + '%', fmt(r['duration']), datetime.strptime(r['date'], '%Y%m%d-%H%M%S').strftime('%Y-%m-%d %H:%M:%S')]
            for r in data
        ]),
        '## 3. 吞吐量与 token 统计',
        table(['并发上限', '请求吞吐 (req/s)', '输出吞吐 (token/s)', '总 token 吞吐 (token/s)', '输入 token 总数', '输出 token 总数', '峰值输出 (token/s)', '记录的峰值并发'], [
            [r['max_concurrency'], fmt(r['request_throughput'], 4), fmt(r['output_throughput']), fmt(r['total_token_throughput']), r['total_input_tokens'], r['total_output_tokens'], fmt(r['max_output_tokens_per_s']), r['max_concurrent_requests']]
            for r in data
        ]),
        '峰值输出、记录的峰值并发分别直接取自 `max_output_tokens_per_s`、`max_concurrent_requests`。后者按秒时间桶累计活跃请求，可能与瞬时峰值、配置的并发上限不同；报告保留原值。',
        '## 4. 延迟明细',
        table(['并发上限', '指标', 'Mean (ms)', 'Median (ms)', 'Std (ms)', 'P99 (ms)'], [
            [r['max_concurrency'], name.upper(), *[fmt(r[f'{stat}_{name}_ms']) for stat in ['mean', 'median', 'std', 'p99']]]
            for r in data for name in ['ttft', 'tpot', 'itl']
        ]),
        '## 5. 折线图',
        '### 5.1 吞吐量',
        '上图展示成功请求吞吐量，下图展示输出与总 token 吞吐量，纵轴从 0 开始。',
        '![不同并发数对应的请求吞吐量和 token 吞吐量](charts/throughput.svg)',
        '### 5.2 首 token 延迟（TTFT）',
        '![不同并发数对应的 TTFT 均值、中位数和 P99](charts/ttft.svg)',
        '### 5.3 生成延迟（TPOT / ITL）',
        '![不同并发数对应的 TPOT 和 ITL 均值、中位数和 P99](charts/decode_latency.svg)',
        '### 5.4 请求成功率与失败率',
        '![不同并发数对应的请求成功率和失败率](charts/success_rate.svg)',
        '### 5.5 整组测试耗时',
        '![不同并发数对应的整组测试耗时](charts/duration.svg)',
        '## 6. 本次结果观察',
        '\n'.join(observations),
        '### 错误摘要',
        table(['并发上限', '错误末行摘要', '次数'], error_rows),
        '## 7. 原始文件与重新生成',
        '\n'.join(f"- 并发 {r['max_concurrency']}：[{r['_source']}]({r['_source']})" for r in data),
        '`request_goodput` 为 `null` 时不展示为有效吞吐量；`rtfx` 不作解释。逐请求数组及完整 errors（如已保存）见原始 JSON。',
        '图表为 SVG 格式，可直接在浏览器打开，缩放不失真。分享报告时请同时保留 `charts/` 目录。',
        '使用 `generate_report.py --result-dir <结果目录>` 可重新生成报告和图表（仅使用 Python 标准库）。',
    ]
    report = result_dir / 'benchmark_report.md'
    report.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'Generated {report.name} from {len(data)} result files.')
    print('Charts: ' + ', '.join(p.name for p in sorted(CHARTS.glob('*.svg'))))


if __name__ == '__main__':
    main()
