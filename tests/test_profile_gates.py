import copy
import json
import subprocess
import sys
from pathlib import Path

from onyx.profiles import registry_digest


REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "tools/check_profile_gates.py"
HEX = "550e75e260a9dc28dc9a97cf6389619a133d10aad2424457f1ee5547ef2fef4d"
OTHER_HEX = "aa" * 32


def run_checker(matrices, registry=None, check_registry=False):
    command = [sys.executable, str(CHECKER), *[str(path) for path in matrices]]
    if registry:
        command.extend(["--registry", str(registry)])
    if check_registry:
        command.append("--check-registry")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.stdout, f"checker produced no stdout; stderr: {result.stderr[-800:]}"
    report = json.loads(result.stdout)
    return result.returncode, report


def build_matrix(tmp_path: Path, mutate=None) -> Path:
    """Build a valid matrix that mutate(runs, matrix) can alter."""
    seconds = {"max-quality": 6.0, "memory-min": 18.0, "speed-focus": 5.9}
    peaks = {"max-quality": 1_700_000_000, "memory-min": 1_600_000_000, "speed-focus": 1_700_000_000}
    weights = {"max-quality": "resident", "memory-min": "stream", "speed-focus": "resident"}
    runs = []
    for profile_id in ("max-quality", "memory-min", "speed-focus"):
        for sample in range(3):
            runs.append({
                "profile": profile_id,
                "stage": "vae",
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "sample_index": sample,
                "weights": weights[profile_id],
                "output_sha256": HEX,
                "output_bytes": 1_400_000,
                "stage_seconds": seconds[profile_id],
                "monitor_seconds": seconds[profile_id] + 0.3,
                "peak_process_bytes": peaks[profile_id],
                "peak_footprint_bytes": 0,
                "peak_rss_bytes": peaks[profile_id],
                "memory_samples": 120,
            })
    matrix = {
        "protocol": "onyx.profile-matrix.v1",
        "label": "synthetic",
        "run_id": "00000000-0000-0000-0000-000000000000",
        "scope": "vae-stage",
        "registry_sha256": registry_digest(),
        "interleaved": True,
        "runs": runs,
    }
    if mutate:
        mutate(runs, matrix)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    return path


def test_valid_matrix_passes(tmp_path):
    code, report = run_checker([build_matrix(tmp_path)])
    assert code == 0, json.dumps(report, ensure_ascii=False)
    assert report["status"] == "pass"
    comparative = report["results"][0]["comparative"]
    assert comparative["verdict"] == "accepted"


def test_registry_mismatch_fails(tmp_path):
    def mutate(runs, matrix):
        matrix["registry_sha256"] = "00" * 32
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("registry_sha256" in check["check"] for check in report["results"][0]["checks"])


def test_memory_gate_violation_fails(tmp_path):
    def mutate(runs, matrix):
        runs[3]["peak_process_bytes"] = 3_000_000_000  # memory-min is capped at 2 GiB.
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("exceeds gate" in check["check"] for check in report["results"][0]["checks"])


def test_replay_mismatch_within_profile_fails(tmp_path):
    def mutate(runs, matrix):
        runs[1]["output_sha256"] = OTHER_HEX
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("replay outputs differ" in check["check"] for check in report["results"][0]["checks"])


def test_bitwise_cross_profile_mismatch_fails(tmp_path):
    def mutate(runs, matrix):
        for run in runs:
            if run["profile"] == "memory-min":
                run["output_sha256"] = OTHER_HEX
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("bitwise identical" in check["check"] for check in report["results"][0]["checks"])


def test_unknown_profile_fails(tmp_path):
    def mutate(runs, matrix):
        runs[0]["profile"] = "turbo"
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("unknown profile" in check["check"] for check in report["results"][0]["checks"])


def test_insufficient_samples_fails(tmp_path):
    def mutate(runs, matrix):
        del runs[0:2]
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("min_samples" in check["check"] for check in report["results"][0]["checks"])


def test_missing_interleaving_fails(tmp_path):
    def mutate(runs, matrix):
        matrix["interleaved"] = False
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    assert any("interleaved" in check["check"] for check in report["results"][0]["checks"])


def test_sequential_protocol_missing_is_rejected(tmp_path):
    def mutate(runs, matrix):
        del matrix["interleaved"]
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1


def test_matrix_without_comparative_profile_is_inconclusive(tmp_path):
    """Missing speed-focus measurements make the speed comparison inconclusive."""
    def mutate(runs, matrix):
        runs[:] = [run for run in runs if run["profile"] != "speed-focus"]
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 0, json.dumps(report, ensure_ascii=False)
    assert report["status"] == "inconclusive"
    assert report["results"][0]["checks"] == []
    assert report["results"][0]["comparative"]["verdict"] == "inconclusive"


def test_speed_profile_slower_beyond_margin_is_rejected(tmp_path):
    def mutate(runs, matrix):
        for run in runs:
            if run["profile"] == "speed-focus":
                run["stage_seconds"] = 25.0
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 1
    comparative = report["results"][0]["comparative"]
    assert comparative["verdict"] == "rejected"


def test_noise_dominated_comparison_is_inconclusive(tmp_path):
    """A roughly 23% gap exceeds the 3% margin but remains within 26% dispersion."""
    def mutate(runs, matrix):
        sf_samples = {0: 12.0, 1: 15.5, 2: 13.5}
        for run in runs:
            if run["profile"] == "speed-focus":
                run["stage_seconds"] = sf_samples[run["sample_index"]]
            elif run["profile"] == "max-quality":
                run["stage_seconds"] = 11.0
            elif run["profile"] == "memory-min":
                run["stage_seconds"] = 11.2
    code, report = run_checker([build_matrix(tmp_path, mutate)])
    assert code == 0, json.dumps(report, ensure_ascii=False)
    comparative = report["results"][0]["comparative"]
    assert comparative["verdict"] == "inconclusive"
    assert "dispersion" in comparative.get("reason", "")


def test_checker_fails_closed_on_broken_registry(tmp_path):
    registry = json.loads((REPO / "config/profiles/profiles.v1.json").read_text())
    registry["$schema"] = "onyx.profiles.v0"
    broken = tmp_path / "broken-registry.json"
    broken.write_text(json.dumps(registry))
    code, report = run_checker([], registry=broken)
    assert code == 1
    assert report["status"] == "fail"


def test_check_registry_only_passes(tmp_path):
    code, report = run_checker([], registry=REPO / "config/profiles/profiles.v1.json", check_registry=True)
    assert code == 0
    assert report["status"] == "pass"
    assert sorted(report["profiles"]) == ["m5-fast", "max-quality", "memory-min", "speed-focus"]


def test_checker_without_inputs_errors(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--registry", str(REPO / "config/profiles/profiles.v1.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "at least one matrix" in result.stderr


def test_checker_with_missing_registry_fails(tmp_path):
    code, report = run_checker([], registry=tmp_path / "missing.json")
    assert code == 1
    assert report["status"] == "fail"
