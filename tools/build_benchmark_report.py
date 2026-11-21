"""Generate a benchmark report from committed measurements and local run records."""

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from onyx.profiles import DEFAULT_REGISTRY_PATH, load_registry, registry_digest  # noqa: E402
from check_profile_gates import check_matrix  # noqa: E402


RESULTS = ROOT / "benchmarks/results"
MATRIX_PATH = RESULTS / "profile-stage-matrix-v1.json"
REPORT_JSON = RESULTS / "profile-benchmark-report-v1.json"
REPORT_MD = ROOT / "docs/BENCHMARKS.md"


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt_seconds(value) -> str:
    return f"{value:.2f} s" if isinstance(value, (int, float)) else "n/a"


def fmt_gib(value) -> str:
    return f"{value / (1 << 30):.2f} GiB" if isinstance(value, (int, float)) and value > 0 else "n/a"


def collect_profile_matrix() -> dict | None:
    if not MATRIX_PATH.is_file():
        return None
    matrix = json.loads(MATRIX_PATH.read_text())
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    verdict = check_matrix(MATRIX_PATH, registry, registry_digest(DEFAULT_REGISTRY_PATH))
    return {
        "scope": matrix.get("scope"),
        "scope_note": matrix.get("scope_note"),
        "git": matrix.get("git"),
        "native_binary_sha256": matrix.get("native_binary", {}).get("sha256"),
        "interleaved": matrix.get("interleaved"),
        "groups": matrix.get("summaries", {}),
        "gate_status": verdict["status"],
        "gate_checks": verdict.get("checks", []),
        "comparative": verdict.get("comparative"),
    }


def collect_end_to_end_history() -> dict:
    rounds = []
    for name in (
        "native-baseline-v1.json",
        "native-round1-zero-cache.json",
        "native-round2-direct-pread.json",
        "native-round3-cpu-modulation.json",
        "native-round4-fused-attention-views.json",
    ):
        data = load(name)
        if not data:
            continue
        entry = {"source": f"benchmarks/results/{name}", "protocol": data.get("protocol")}
        if data["protocol"] == "onyx.native-baseline.v1":
            entry.update({"label": data.get("label"), "seconds": data["run"]["seconds"], "peak_bytes": data["run"]["peak_bytes"]})
        else:
            entry.update({
                "round": data.get("round"),
                "change": data.get("change"),
                "candidate_seconds_median": data.get("candidate_seconds_median"),
                "speedup_vs_baseline": data.get("speedup_vs_baseline") or data.get("speedup"),
                "candidate_peak_bytes_max": data.get("candidate_peak_bytes_max") or data.get("candidate_peak_bytes"),
                "all_outputs_identical": data.get("all_outputs_identical"),
                "decision": data.get("decision"),
            })
        rounds.append(entry)
    stability = load("native-v1-stability.json")
    return {
        "rounds": rounds,
        "v1_stability": {
            "source": "benchmarks/results/native-v1-stability.json",
            "native_1024": stability.get("native_1024") if stability else None,
            "native_4096": stability.get("native_4096") if stability else None,
            "community_comparison": stability.get("community_comparison") if stability else None,
        } if stability else None,
    }


def collect_operator_reference() -> dict | None:
    data = load("m5-max-36gb-q4-operator-v1.json")
    if not data:
        return None
    comparisons = []
    for item in data.get("comparisons", []):
        comparisons.append({
            "case_id": item.get("case_id"),
            "shape": item.get("shape"),
            "onyx_wall_median_ms": item.get("onyx", {}).get("wall_median_ms"),
            "mlx_wall_median_ms": item.get("mlx", {}).get("wall_median_ms"),
            "onyx_over_mlx_time_ratio": item.get("onyx_over_mlx_time_ratio"),
        })
    return {
        "primary_metric": data.get("primary_metric"),
        "scope_warning": data.get("scope_warning"),
        "comparisons": comparisons,
    }


def collect_kernel_variants() -> dict | None:
    data = load("m5-max-36gb-q4-tile32-v1.json")
    if not data:
        return None
    return {
        "primary_metric": data.get("primary_metric"),
        "results": [
            {
                "case_id": item.get("case_id"),
                "baseline_variant": item.get("baseline_variant"),
                "candidate_variant": item.get("candidate_variant"),
                "baseline_wall_median_ms": item.get("baseline_wall_median_ms"),
                "candidate_wall_median_ms": item.get("candidate_wall_median_ms"),
                "speedup": item.get("speedup"),
                "candidate_max_abs_error": item.get("candidate_max_abs_error"),
            }
            for item in data.get("results", [])
        ],
    }


def collect_community_comparison() -> dict | None:
    data = load("h3-teapot-1024-comparison-v1.json")
    if not data:
        return None
    community = data.get("community", {})
    native = data.get("onyx_native", {})
    return {
        "scope": data.get("scope"),
        "community": {key: community.get(key) for key in ("implementation", "precision", "seconds_median", "seconds_p95", "measured_runs")},
        "onyx_native": {
            "seconds": native.get("seconds"),
            "peak_bytes": native.get("peak_bytes"),
            "stages": native.get("stages"),
        },
        "native_over_community_time_ratio": data.get("native_over_community_time_ratio"),
        "native_image_similarity": data.get("native_image_similarity"),
    }


def collect_quality_evidence() -> dict:
    baseline = load("native-baseline-v1.json")
    adaln = load("h3-adaln-timestep-fix-v1.json")
    layouts = load("h3-original-layout-fixes-v1.json")
    return {
        "text_reference_correlation": baseline.get("text_reference_correlation") if baseline else None,
        "vae_reference_pixel_correlation": baseline.get("vae_reference_pixel_correlation") if baseline else None,
        "adaln_timestep_fix": {
            "before_min_part_correlation": adaln.get("before_min_part_correlation"),
            "after_final_correlation": adaln.get("after_final_correlation"),
        } if adaln else None,
        "layout_fixes": {
            "timestep_required_order": layouts.get("timestep", {}).get("required_order"),
            "qkv_required_layout": layouts.get("qkv", {}).get("required_layout"),
        } if layouts else None,
        "quality_scope": baseline.get("quality_scope") if baseline else None,
    }


def collect_recent_runs() -> dict:
    groups: dict[str, list] = {}
    for run_json in sorted((ROOT / "artifacts/benchmarks/onyx-native-h3").glob("*/*/run.json")):
        group = run_json.parent.parent.name
        try:
            record = json.loads(run_json.read_text())
        except json.JSONDecodeError:
            continue
        wall = record.get("total_wall_time_ns")
        peak = record.get("peak_memory_bytes")
        plan = record.get("plan_id")
        groups.setdefault(group, []).append({
            "seconds": int(wall) / 1e9 if wall else None,
            "peak_bytes": int(peak) if peak else None,
            "plan_id": plan,
        })
    summaries = {}
    for group, runs in sorted(groups.items()):
        seconds = [run["seconds"] for run in runs if run["seconds"]]
        peaks = [run["peak_bytes"] for run in runs if run["peak_bytes"]]
        summaries[group] = {
            "runs": len(runs),
            "seconds_median": statistics.median(seconds) if seconds else None,
            "seconds_min": min(seconds) if seconds else None,
            "seconds_max": max(seconds) if seconds else None,
            "peak_bytes_max": max(peaks) if peaks else None,
            "plan_ids": sorted({run["plan_id"] for run in runs if run["plan_id"]}),
        }
    return {"source_root": "artifacts/benchmarks/onyx-native-h3", "groups": summaries}


def render_markdown(report: dict) -> str:
    target = report["target"]
    lines = [
        "# Onyx Infer benchmark report",
        "",
        f"Generated {report['generated_at']} from revision `{report['git']['source_revision']}` (dirty={report['git']['source_dirty']}).",
        f"Target {target['chip']} ({target['target_id']}) with {target['physical_memory_bytes'] / (1 << 30):.0f} GB unified memory and {target['metal']}.",
        "",
        "Generated by `tools/build_benchmark_report.py` from recorded measurements. Each section states its workload, precision, and comparison scope.",
        "",
        "## Execution profiles",
        "",
        "The three profiles share one W4 model and six-evaluation PDD recipe. See the [profile registry and acceptance gates](PROFILES.md).",
        "",
        "| Profile | Use | VAE weights | Key checks |",
        "| --- | --- | --- | --- |",
        "| `max-quality` | Default MPSGraph path | resident | Replay consistency, peak ≤ 28.08 GiB |",
        "| `memory-min` | All-stage streaming | stream | Replay consistency, bitwise match to max-quality, provisional 2 GiB peak |",
        "| `speed-focus` | Experimental BF16 SIMD and DiT cache | resident | Replay consistency, peak ≤ 28.08 GiB, 3% latency comparison margin |",
        "",
    ]
    matrix = report.get("profile_stage_matrix")
    if matrix:
        lines += [
            f"### Stage matrix ({matrix['scope']})",
            "",
            f"> {matrix['scope_note']}",
            "",
            f"Gate status **{matrix['gate_status'].upper()}**. Interleaved: {matrix['interleaved']}. Report registry `{report['profiles']['registry_sha256'][:16]}…`.",
            "",
            "| Group | Samples | Median stage time | Peak process-tree memory | Identical output SHA-256 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for key, group in matrix["groups"].items():
            identical = "Yes" if len(group["output_sha256"]) == 1 else "No"
            lines.append(f"| {key} | {group['samples']} | {fmt_seconds(group['stage_seconds_median'])} | {fmt_gib(group['peak_process_bytes_max'])} | {identical} |")
        comparative = matrix.get("comparative") or {}
        per_size = comparative.get("per_size") or []
        margin_ratio = comparative.get("margin_ratio")
        if not isinstance(margin_ratio, (int, float)) and per_size:
            margin_ratio = per_size[0].get("margin_ratio")
        if per_size:
            lines += ["", f"Latency gate (must_be_fastest_within, margin {margin_ratio if isinstance(margin_ratio, (int, float)) else 0:.0%}). **{comparative['verdict']}**"]
            for item in per_size:
                lines.append(
                    f"- {item.get('size')}: {item.get('verdict')} (median {fmt_seconds(item.get('median_seconds'))}, "
                    f"gap {item.get('gap_ratio', 0):.1%}, dispersion {item.get('observed_dispersion', 0):.1%}). {item.get('reason')}"
                )
        elif comparative.get("median_seconds") is not None:
            others = ", ".join(f"{name} {fmt_seconds(value)}" for name, value in comparative.get("other_medians_seconds", {}).items())
            lines += [
                "",
                f"Latency gate (must_be_fastest_within, margin {comparative.get('margin_ratio', 0):.0%}). **{comparative['verdict']}**. speed-focus median {fmt_seconds(comparative.get('median_seconds'))}. Comparison medians {others or 'n/a'}.",
            ]
        lines.append("")
    else:
        lines += ["Run `uv run python benchmarks/run_profiles.py` to produce a stage matrix.", ""]

    lines += ["## Native 1024×1024 teapot benchmark", "", "| Configuration | Wall time | Peak memory | Identical output | Decision |", "| --- | ---: | ---: | --- | --- |"]
    for entry in report["end_to_end_history"]["rounds"]:
        if "round" in entry:
            seconds = fmt_seconds(entry.get("candidate_seconds_median"))
            peak = fmt_gib(entry.get("candidate_peak_bytes_max"))
            identical = entry.get("all_outputs_identical")
            lines.append(f"| {entry['change']} | {seconds} | {peak} | {'Yes' if identical else 'n/a'} | {entry.get('decision', 'n/a')} |")
        else:
            lines.append(f"| Native streaming baseline | {fmt_seconds(entry.get('seconds'))} | {fmt_gib(entry.get('peak_bytes'))} | Reference | Baseline |")
    stability = report["end_to_end_history"].get("v1_stability") or {}
    native_4096 = stability.get("native_4096") or {}
    if native_4096:
        lines += [
            "",
            f"The single 4096×4096 result took {fmt_seconds(native_4096.get('seconds'))} at {fmt_gib(native_4096.get('peak_bytes'))} peak. [Run record](../benchmarks/results/native-v1-stability.json)",
        ]
    lines.append("")

    operators = report.get("operator_reference")
    if operators:
        lines += [
            "## Q4 operators: Onyx MPSGraph streaming prototype and MLX",
            "",
            f"Metric: {operators['primary_metric']}",
            "",
            f"> {operators['scope_warning']}",
            "",
            "| Case | Shape m×n×k | Onyx (ms) | MLX (ms) | Ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for item in operators["comparisons"]:
            shape = item.get("shape", {})
            lines.append(f"| {item['case_id']} | {shape.get('m')}×{shape.get('n')}×{shape.get('k')} | {item['onyx_wall_median_ms']:.2f} | {item['mlx_wall_median_ms']:.2f} | {item['onyx_over_mlx_time_ratio']:.2f}× |")
        lines.append("")

    kernels = report.get("kernel_variants")
    if kernels:
        lines += ["## Q4 kernel variants: tile16 and tile32", "", f"Metric: {kernels['primary_metric']}", "", "| Case | tile16 (ms) | tile32 (ms) | Speedup | Max absolute error |", "| --- | ---: | ---: | ---: | ---: |"]
        for item in kernels["results"]:
            lines.append(f"| {item['case_id']} | {item['baseline_wall_median_ms']:.3f} | {item['candidate_wall_median_ms']:.3f} | {item['speedup']:.3f}× | {item['candidate_max_abs_error']:.2e} |")
        lines.append("")

    community = report.get("community_comparison")
    if community:
        lines += [
            "## Community comparison",
            "",
            f"{community.get('scope')}",
            "",
            f"- Community `h3-image-mlx` ({community['community'].get('precision')}): {fmt_seconds(community['community'].get('seconds_median'))} median, {community['community'].get('measured_runs')} samples",
            f"- Native Onyx W4: {fmt_seconds(community['onyx_native'].get('seconds'))}, peak {fmt_gib(community['onyx_native'].get('peak_bytes'))}, ratio {community['native_over_community_time_ratio']:.2f}×",
        ]
        psnr = (community.get("native_image_similarity") or {}).get("psnr_db")
        if isinstance(psnr, (int, float)):
            lines.append(f"- Image similarity: {psnr:.2f} dB PSNR for this request across different quantization paths.")
        lines.append("")

    quality = report.get("quality_evidence")
    if quality:
        lines += [
            "## Quality evidence",
            "",
            f"- Native text correlation with the W4 reference: {quality['text_reference_correlation']:.7f}" if quality.get("text_reference_correlation") else "- Text encoder correlation: n/a",
            f"- Native VAE pixel correlation with a shared input latent: {quality['vae_reference_pixel_correlation']:.7f}" if quality.get("vae_reference_pixel_correlation") else "- VAE correlation: n/a",
        ]
        adaln = quality.get("adaln_timestep_fix")
        if adaln:
            lines.append(f"- AdaLN timestep fix: correlation {adaln['before_min_part_correlation']:.4f} → {adaln['after_final_correlation']:.7f}")
        layouts = quality.get("layout_fixes")
        if layouts:
            lines.append(f"- Numerical layout: timestep `{layouts['timestep_required_order']}`，QKV `{layouts['qkv_required_layout']}`")
        if quality.get("quality_scope"):
            lines.append(f"- Quality scope: {quality['quality_scope']}")
        lines.append("")

    runs = report.get("recent_runs", {}).get("groups", {})
    if runs:
        lines += ["## Native RunRecord distribution", "", "| Case group | Runs | Median latency | Range | Maximum peak memory |", "| --- | ---: | ---: | --- | ---: |"]
        for group, summary in runs.items():
            if summary["seconds_median"] is None:
                continue
            lines.append(f"| {group} | {summary['runs']} | {fmt_seconds(summary['seconds_median'])} | {fmt_seconds(summary['seconds_min'])} – {fmt_seconds(summary['seconds_max'])} | {fmt_gib(summary['peak_bytes_max'])} |")
        lines += ["", "> This distribution can mix multiple builds and plans. Use the per-configuration evidence for performance comparisons.", ""]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "# Measure the VAE stage",
        "uv run python benchmarks/run_profiles.py --sizes 1024x1024,4096x4096 --runs 3 --output artifacts/benchmarks/profile-stage-matrix.json",
        "",
        "# Check the new matrix",
        "uv run python tools/check_profile_gates.py artifacts/benchmarks/profile-stage-matrix.json",
        "",
        "# Regenerate this report",
        "uv run python tools/build_benchmark_report.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    revision, dirty = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    report = {
        "protocol": "onyx.profile-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {"source_revision": revision, "source_dirty": dirty},
        "target": json.loads((ROOT / "config/targets/m5-max-36gb.json").read_text()),
        "profiles": {
            "registry": str(DEFAULT_REGISTRY_PATH.relative_to(ROOT)),
            "registry_sha256": registry_digest(DEFAULT_REGISTRY_PATH),
            "registry_valid": load_registry(DEFAULT_REGISTRY_PATH) is not None,
        },
        "profile_stage_matrix": collect_profile_matrix(),
        "end_to_end_history": collect_end_to_end_history(),
        "operator_reference": collect_operator_reference(),
        "kernel_variants": collect_kernel_variants(),
        "community_comparison": collect_community_comparison(),
        "quality_evidence": collect_quality_evidence(),
        "recent_runs": collect_recent_runs(),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(render_markdown(report))
    print(f"Report JSON: {REPORT_JSON}")
    print(f"Report MD:   {REPORT_MD}")


if __name__ == "__main__":
    main()
