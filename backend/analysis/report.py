"""Reports and scientific plots drawn only from recorded samples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.metrics.statistics import summarize


def plot_cases(cases: list[dict[str, Any]], output: Path) -> None:
    # Avoid writing Matplotlib's cache into a user's global configuration directory.
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent.resolve() / ".mplcache"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, max(3, 0.55 * len(cases) + 1.8)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    labels, medians, lower, upper, colors = [], [], [], [], []
    for case in cases:
        c = case["config"]
        stats = summarize([sample["tokens_per_second"] for sample in case["samples"]])
        median = stats["median"]
        ci = stats["median_ci95"]
        labels.append(
            f"{c['kind']} {c['tokens']}t · {c['threads']} threads · "
            f"GPU {c['gpu_layers']} · b{c['batch_size']}/ub{c['ubatch_size']}"
        )
        medians.append(median)
        lower.append(median - ci[0] if ci else 0)
        upper.append(ci[1] - median if ci else 0)
        colors.append("#537b9c" if c["gpu_layers"] == 0 else "#258a8c")
    bars = axes.barh(labels, medians, xerr=[lower, upper], capsize=4, color=colors, alpha=0.9)
    axes.bar_label(bars, labels=[f"{value:.2f}" for value in medians], padding=5, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel("Synthetic workload throughput (tokens/second)")
    axes.set_title("Median throughput · whiskers: within-process 95% bootstrap CI")
    axes.set_xlim(left=0, right=max(m + e for m, e in zip(medians, upper, strict=True)) * 1.2)
    axes.grid(axis="x", alpha=0.18)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.savefig(output.with_suffix(".svg"))
    figure.savefig(output.with_suffix(".png"), dpi=150)
    figure.clear()


def report(experiment: dict[str, Any], output: Path) -> None:
    if experiment.get("config", {}).get("mode") == "request_warm":
        from backend.analysis.request_report import report_request

        report_request(experiment, output)
        return
    lines = [
        f"# {experiment['name']}",
        "",
        f"Experiment: `{experiment['id']}`",
        f"Status: **{experiment['status']}**",
        "",
        "Synthetic prefill/decode microbenchmark; no request latency or TTFT is measured.",
        "",
        "| Workload | Tokens | Threads | GPU layers | Batch / microbatch | n | Median tok/s | "
        "Mean | SD | Median 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    completed, failures = [], []
    for case in experiment["cases"]:
        c = case["config"]
        if not case["samples"]:
            failures.append(f"Case `{case['id']}`: {case['status']} — {case['error']}")
            continue
        stats = summarize([sample["tokens_per_second"] for sample in case["samples"]])
        ci = stats["median_ci95"]
        interval = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "unavailable"
        sd = f"{stats['stdev']:.2f}" if stats["stdev"] is not None else "unavailable"
        lines.append(
            f"| {c['kind']} | {c['tokens']} | {c['threads']} | {c['gpu_layers']} | "
            f"{c['batch_size']} / {c['ubatch_size']} | {stats['count']} | {stats['median']:.2f} | "
            f"{stats['mean']:.2f} | {sd} | {interval} |"
        )
        completed.append(case)
    lines += [
        "",
        *failures,
        "",
        "Confidence intervals resample within one process, not independent machine sessions.",
        "p95 requires at least 20 repetitions. Small samples cannot describe tails.",
        "",
        "Process RSS includes mapped host memory; it is not measured GPU VRAM.",
        "",
        "Full raw exports retain commands, timings, model hash, stderr and environment.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if completed:
        plot_cases(completed, output)
        lines += ["", f"![Measured throughput]({output.with_suffix('.svg').name})"]
    output.write_text("\n".join(lines) + "\n")
