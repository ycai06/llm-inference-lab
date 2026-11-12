"""Request latency and native engine throughput reports, separated by metric and case."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.metrics.statistics import summarize

METRICS = (
    ("request_latency_ms", "Request latency", "ms"),
    ("ttft_ms", "TTFT", "ms"),
    ("prefill_tokens_per_second", "Native prefill", "tokens/s"),
    ("decode_tokens_per_second", "Native decode", "tokens/s"),
    ("mean_inter_token_latency_ms", "Mean observed token interval", "ms"),
    ("median_inter_token_latency_ms", "Median observed token interval", "ms"),
)


def _label(case: dict[str, Any]) -> str:
    config = case["config"]
    fixture = Path(str(config.get("fixture", "fixture"))).name
    return (
        f"{fixture} · p{config.get('prompt_tokens', '?')} / "
        f"g{config.get('generation_tokens', '?')} · "
        f"{config.get('threads', '?')} threads · GPU {config.get('gpu_layers', '?')} · "
        f"b{config.get('batch_size', '?')}/ub{config.get('ubatch_size', '?')} · "
        f"ctx{config.get('context_size', '?')}"
    )


def _stats(samples: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    values = [sample[field] for sample in samples if sample.get(field) is not None]
    return summarize(values) if values else None


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


def _plot_label(case: dict[str, Any], metric: str) -> str:
    config = case["config"]
    fixture = Path(str(config.get("fixture", "fixture"))).name
    return (
        f"{fixture} · p{config.get('prompt_tokens', '?')}/"
        f"g{config.get('generation_tokens', '?')} · {metric}\n"
        f"{config.get('threads', '?')} threads · GPU {config.get('gpu_layers', '?')} · "
        f"b{config.get('batch_size', '?')}/ub{config.get('ubatch_size', '?')} · "
        f"ctx{config.get('context_size', '?')}"
    )


def _plot(cases: list[dict[str, Any]], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent.resolve() / ".mplcache"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, max(3.5, len(cases) * 1.2 + 1.5)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, selections, units, title in (
        (axes[0], METRICS[:2], "Milliseconds", "Client-observed latency"),
        (axes[1], METRICS[2:4], "Tokens/second", "Native server throughput"),
    ):
        labels, medians, lower, upper, colors = [], [], [], [], []
        for case in cases:
            for index, (field, name, _) in enumerate(selections):
                stats = _stats(case["request_samples"], field)
                if stats is None:
                    continue
                median, interval = stats["median"], stats["median_ci95"]
                # Keep all experiment controls visible while fitting a two-panel plot.
                labels.append(_plot_label(case, name))
                medians.append(median)
                lower.append(median - interval[0] if interval else 0)
                upper.append(interval[1] - median if interval else 0)
                colors.append(("#537b9c", "#258a8c")[index])
        axis.set_title(title)
        axis.set_xlabel(units)
        if not medians:
            axis.text(0.5, 0.5, "unavailable", transform=axis.transAxes, ha="center", va="center")
            axis.set_yticks([])
            continue
        positions = list(range(len(labels)))
        bars = axis.barh(positions, medians, xerr=[lower, upper], color=colors, capsize=3)
        axis.set_yticks(positions, labels)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in medians], padding=4, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(1, max(m + e for m, e in zip(medians, upper, strict=True))) * 1.2)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("Warm requests · whiskers: within-case 95% bootstrap median CI")
    figure.savefig(output.with_suffix(".svg"))
    figure.savefig(output.with_suffix(".png"), dpi=150)
    figure.clear()


def report_request(experiment: dict[str, Any], output: Path) -> None:
    """Render real request trials without pooling them with synthetic benchmark timings."""
    if experiment.get("config", {}).get("mode") != "request_warm":
        raise ValueError("Request reports require mode: request_warm")
    lines = [
        f"# {experiment['name']}",
        "",
        f"Experiment: `{experiment['id']}`",
        f"Status: **{experiment['status']}**",
        "",
        "Warm single-request measurements; model initialization and warmups are excluded.",
        "TTFT begins at request dispatch and ends at the first generated-token event.",
        "",
    ]
    completed = []
    for case in experiment["cases"]:
        samples = case.get("request_samples", [])
        lines += [f"## {_label(case)}", "", f"Case: `{case['id']}`", ""]
        if not samples:
            lines += [f"Case `{case['id']}`: {case['status']} — {case.get('error')}", ""]
            continue
        completed.append(case)
        token_counts = sorted({sample["generated_tokens"] for sample in samples})
        early = sum(sample["early_stop"] for sample in samples)
        lines += [
            f"Measured requests: {len(samples)}. Observed generated token counts: "
            f"{', '.join(map(str, token_counts))}. Early stops: {early}.",
            "",
            "| Metric | Unit | n | Median | Mean | Sample SD | p95 | Median 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for field, name, unit in METRICS:
            stats = _stats(samples, field)
            if stats is None:
                lines.append(
                    f"| {name} | {unit} | 0 | unavailable | unavailable | "
                    "unavailable | unavailable | unavailable |"
                )
                continue
            interval = stats["median_ci95"]
            ci = f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "unavailable"
            lines.append(
                f"| {name} | {unit} | {stats['count']} | {_number(stats['median'])} | "
                f"{_number(stats['mean'])} | {_number(stats['stdev'])} | "
                f"{_number(stats['p95'])} | {ci} |"
            )
        warnings = sorted({warning for sample in samples for warning in sample.get("warnings", [])})
        if warnings:
            lines += ["", "Recorded warnings:", "", *(f"- {warning}" for warning in warnings)]
        lines.append("")
    lines += [
        "Native prefill and decode throughput retain the server's timing definitions. "
        "Decode timing excludes the first generated token's prompt-logit evaluation.",
        "",
        "Token intervals are client-observed stream arrivals and may be unavailable when "
        "event coverage is incomplete or transport coalesces token events.",
        "",
        "Confidence intervals resample within a case and omit between-session variation. "
        "They require five samples; p95 requires twenty. Cases and operational modes "
        "are never pooled.",
        "",
        "Process RSS includes host and potentially unified-memory allocations and is not "
        "measured GPU VRAM. Raw request payloads, terminal responses and event arrivals "
        "remain in the JSON export.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if completed:
        _plot(completed, output)
        lines += ["", f"![Request measurements]({output.with_suffix('.svg').name})"]
    output.write_text("\n".join(lines) + "\n")
