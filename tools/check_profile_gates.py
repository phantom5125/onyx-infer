"""Evaluate acceptance gates for onyx.profile-matrix.v1 measurements.

The checker verifies registry identity, settings, sample counts, replay
hashes, peak memory, and interleaving. Comparative gates report accepted,
inconclusive, or rejected. Exit codes are 0 for pass or inconclusive and
1 for failure.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onyx.profiles import DEFAULT_REGISTRY_PATH, load_registry, registry_digest  # noqa: E402


MATRIX_SCHEMA = "onyx.profile-matrix.v1"
REPORT_SCHEMA = "onyx.profile-gate-report.v1"


def fail(checks: list, reason: str) -> None:
    checks.append({"status": "fail", "check": reason})


def group_key(run: dict) -> tuple:
    return (run["profile"], run["width"], run["height"])


def check_matrix(matrix_path: Path, registry: dict, registry_digest_value: str) -> dict:
    """Check one matrix and return its checks, groups, and comparative verdict."""
    checks: list[dict] = []
    try:
        matrix = json.loads(matrix_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(checks, f"unreadable matrix: {error}")
        return {"matrix": str(matrix_path), "readable": False, "checks": checks, "status": "fail"}

    if matrix.get("protocol") != MATRIX_SCHEMA:
        fail(checks, f"protocol must be {MATRIX_SCHEMA!r}, got {matrix.get('protocol')!r}")
    if matrix.get("registry_sha256") != registry_digest_value:
        fail(checks, "registry_sha256 does not match the current registry; regenerate the matrix (evidence must bind to the registry version)")
    if not isinstance(matrix.get("runs"), list) or not matrix["runs"]:
        fail(checks, "runs must be a non-empty list")
        return {"matrix": str(matrix_path), "readable": True, "checks": checks, "status": "fail"}
    if matrix.get("interleaved") is not True:
        fail(checks, "interleaved must be true; sequential A/B has thermal bias (LOOP.md §4)")

    profiles = registry["profiles"]
    groups: dict[tuple, dict] = {}
    for index, run in enumerate(matrix["runs"]):
        where = f"runs[{index}]"
        profile_id = run.get("profile")
        profile = profiles.get(profile_id)
        if profile is None:
            fail(checks, f"{where}: unknown profile {profile_id!r}")
            continue
        stage = run.get("stage")
        if stage not in ("text", "dit", "vae"):
            fail(checks, f"{where}: unknown stage {stage!r}")
            continue
        declared = profile["knobs"].get(f"{stage}_weights")
        if run.get("weights") != declared:
            fail(checks, f"{where}: weights {run.get('weights')!r} does not match profile knob {declared!r}")
        valid = True
        seconds = run.get("stage_seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not seconds > 0:
            fail(checks, f"{where}: stage_seconds must be a positive number")
            valid = False
        peak = run.get("peak_process_bytes")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
            fail(checks, f"{where}: peak_process_bytes must be a positive integer")
            valid = False
        if not run.get("memory_samples"):
            fail(checks, f"{where}: memory_samples must be >= 1")
            valid = False
        sha = run.get("output_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            fail(checks, f"{where}: output_sha256 must be a 64-char hex digest")
            valid = False
        if valid:
            group = groups.setdefault((profile_id, run.get("width"), run.get("height")), [])
            group.append(run)

    for (profile_id, width, height), runs in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])):
        label = f"{profile_id}@{width}x{height}"
        gates = profiles[profile_id]["gates"]
        if len(runs) < gates["min_samples"]:
            fail(checks, f"{label}: {len(runs)} samples < min_samples {gates['min_samples']}")
        if gates.get("output_replay_identical") and len({run["output_sha256"] for run in runs}) != 1:
            fail(checks, f"{label}: replay outputs differ within the profile")
        peak_max = max(run["peak_process_bytes"] for run in runs)
        if peak_max > gates["peak_process_bytes_max"]:
            fail(checks, f"{label}: peak {peak_max} exceeds gate {gates['peak_process_bytes_max']}")
        bitwise = gates.get("bitwise_equal_to")
        if bitwise:
            reference = groups.get((bitwise, width, height))
            if not reference:
                fail(checks, f"{label}: bitwise_equal_to={bitwise} but reference group is missing")
            elif {run["output_sha256"] for run in runs} != {run["output_sha256"] for run in reference}:
                fail(checks, f"{label}: outputs are not bitwise identical to {bitwise}")

    comparative = check_comparative(registry, groups, checks)
    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif comparative["verdict"] == "inconclusive":
        status = "inconclusive"
    else:
        status = "pass"
    return {"matrix": str(matrix_path), "readable": True, "checks": checks, "groups": {str(key): len(value) for key, value in groups.items()}, "comparative": comparative, "status": status}


def check_comparative(registry: dict, groups: dict, checks: list) -> dict:
    """Return the least favorable speed verdict across measured resolutions."""
    VERDICT_ORDER = {"accepted": 0, "inconclusive": 1, "rejected": 2}
    worst: dict | None = None
    for profile_id, profile in registry["profiles"].items():
        gate = profile["gates"].get("must_be_fastest_within")
        if not gate:
            continue
        sizes = sorted({(key[1], key[2]) for key in groups if key[0] == profile_id})
        if not sizes:
            result = {"gate": "must_be_fastest_within", "profile": profile_id, "verdict": "inconclusive", "reason": "no measurements for this profile"}
        else:
            verdicts = [compare_one_size(registry, groups, profile_id, gate, width, height) for width, height in sizes]
            selected = max(verdicts, key=lambda item: VERDICT_ORDER[item["verdict"]])
            result = {
                "gate": "must_be_fastest_within",
                "profile": profile_id,
                "verdict": selected["verdict"],
                "reason": selected.get("reason"),
                "per_size": verdicts,
            }
        if worst is None or VERDICT_ORDER[result["verdict"]] > VERDICT_ORDER[worst["verdict"]]:
            worst = result
    if worst is None:
        return {"gate": None, "verdict": "accepted", "reason": "no comparative gate declared"}
    if worst["verdict"] == "rejected":
        checks.append({"status": "fail", "check": "must_be_fastest_within rejected"})
    return worst


def group_dispersion(runs: list) -> float:
    """Compute within-group dispersion as range divided by median."""
    seconds = [run["stage_seconds"] for run in runs]
    median = statistics.median(seconds)
    if median <= 0 or len(seconds) < 2:
        return 0.0
    return (max(seconds) - min(seconds)) / median


def compare_one_size(registry: dict, groups: dict, profile_id: str, gate: dict, width: int, height: int) -> dict:
    mine = groups.get((profile_id, width, height))
    if not mine:
        return {"profile": profile_id, "size": f"{width}x{height}", "verdict": "inconclusive", "reason": "no measurements for this size"}
    others = {key: runs for key, runs in groups.items() if key[0] != profile_id and (key[1], key[2]) == (width, height)}
    if not others:
        return {"profile": profile_id, "size": f"{width}x{height}", "verdict": "inconclusive", "reason": "no comparison profiles in matrix"}
    incomplete = [key[0] for key, runs in others.items() if len(runs) < registry["profiles"][key[0]]["gates"]["min_samples"]]
    if incomplete:
        return {"profile": profile_id, "size": f"{width}x{height}", "verdict": "inconclusive", "reason": f"comparison groups below min_samples: {', '.join(sorted(incomplete))}"}
    margin = gate["margin_ratio"]
    mine_median = statistics.median(run["stage_seconds"] for run in mine)
    other_medians = {key[0]: statistics.median(run["stage_seconds"] for run in runs) for key, runs in others.items()}
    best_other = min(other_medians.values())
    dispersion = max(
        [group_dispersion(mine)] + [group_dispersion(runs) for runs in others.values()]
    )
    result = {
        "profile": profile_id,
        "size": f"{width}x{height}",
        "median_seconds": mine_median,
        "other_medians_seconds": other_medians,
        "margin_ratio": margin,
        "observed_dispersion": dispersion,
        "gap_ratio": mine_median / best_other - 1 if best_other > 0 else 0.0,
    }
    if mine_median <= best_other * (1 + margin):
        result["verdict"] = "accepted"
        result["reason"] = "median is the fastest or within the declared noise margin"
    elif result["gap_ratio"] > dispersion:
        # Reject a slowdown beyond both the margin and observed dispersion.
        result["verdict"] = gate["verdict_when_slower_beyond_margin"]
        result["reason"] = f"median exceeds the fastest comparison by {result['gap_ratio']:.0%}, beyond the margin and beyond observed dispersion"
    else:
        # The slowdown falls within measured dispersion, so the comparison is inconclusive.
        result["verdict"] = gate["verdict_when_missed"]
        result["reason"] = (
            f"median exceeds the fastest comparison by {result['gap_ratio']:.0%}, "
            f"within observed dispersion {dispersion:.0%}; thermal/noise-dominated comparison"
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="onyx-profile-gates")
    parser.add_argument("matrices", nargs="*", type=Path, help="onyx.profile-matrix.v1 result files")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--check-registry", action="store_true", help="Validate the registry structure")
    args = parser.parse_args(argv)

    try:
        registry = load_registry(args.registry)
    except ValueError as error:
        print(json.dumps({"protocol": REPORT_SCHEMA, "status": "fail", "reason": str(error)}, indent=2))
        return 1
    if args.check_registry and not args.matrices:
        print(json.dumps({"protocol": REPORT_SCHEMA, "status": "pass", "registry": str(args.registry), "profiles": sorted(registry["profiles"])}, indent=2))
        return 0
    if not args.matrices:
        parser.error("provide at least one matrix file or --check-registry")

    digest = registry_digest(args.registry)
    results = [check_matrix(path, registry, digest) for path in args.matrices]
    statuses = {result["status"] for result in results}
    if "fail" in statuses:
        overall = "fail"
    elif "inconclusive" in statuses:
        overall = "inconclusive"
    else:
        overall = "pass"
    report = {"protocol": REPORT_SCHEMA, "status": overall, "registry": str(args.registry), "registry_sha256": digest, "results": results}
    print(json.dumps(report, indent=2))
    return 0 if overall in ("pass", "inconclusive") else 1


if __name__ == "__main__":
    raise SystemExit(main())
