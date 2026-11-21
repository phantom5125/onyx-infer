import argparse
import datetime
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "config/targets/m5-max-36gb.json"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_revision():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return revision, dirty


def case_file(value):
    if value == "metal-only":
        return None
    if value in ("teapot-1024", "teapot-4096"):
        return ROOT / f"benchmarks/requests/{value}.json"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class Reporter:
    def __init__(self, suite, label, variant, target=DEFAULT_TARGET):
        self.suite = suite
        self.label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-.") or "validation"
        self.variant = variant
        self.target = target
        self.started_at = utc_now()
        self.started_ns = time.monotonic_ns()
        self.checks = []
        self.details = {}
        self.status = "pass"
        self.reason = None

    def block(self, reason):
        self.status = "blocked"
        self.reason = reason
        self.details["blocked_reason"] = reason

    def run(self, name, command, timeout=7200):
        started = time.monotonic_ns()
        print(f"[{name}] {' '.join(map(str, command))}", flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
            stdout = result.stdout
            log = stdout + result.stderr
            returncode = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            log = str(error)
            returncode = 127
        duration = time.monotonic_ns() - started
        self.checks.append({
            "name": name,
            "command": [str(part) for part in command],
            "status": "pass" if returncode == 0 else "fail",
            "returncode": returncode,
            "duration_ns": duration,
            "log_tail": log[-4000:],
        })
        if returncode:
            print(log[-4000:], file=sys.stderr, flush=True)
            self.status = "fail"
            self.reason = f"{name} exited {returncode}"
            raise RuntimeError(self.reason)
        return stdout

    def write(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        revision, dirty = source_revision()
        target = json.loads(self.target.read_text())
        report = {
            "protocol": "onyx.ci.v1",
            "suite": self.suite,
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "duration_ns": time.monotonic_ns() - self.started_ns,
            "git": {"revision": revision, "dirty": dirty},
            "target": {
                "target_id": target["target_id"],
                "chip": target["chip"],
                "physical_memory_bytes": target["physical_memory_bytes"],
                "metal": target["metal"],
                "metal_compiler_version": target.get("metal_compiler_version"),
            },
            "checks": self.checks,
            "details": self.details,
        }
        json_path = directory / f"{self.label}.json"
        markdown_path = directory / f"{self.label}.md"
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        markdown_path.write_text(self.markdown(report))
        print(f"Report: {json_path}")
        print(f"Summary: {markdown_path}")
        return json_path, markdown_path

    @staticmethod
    def markdown(report):
        lines = [
            f"# Onyx validation: `{report['label']}`",
            "",
            f"- Protocol: `{report['protocol']}`",
            f"- Suite: `{report['suite']}`",
            f"- Status: **{report['status'].upper()}**",
            f"- Git revision: `{report['git']['revision']}` (dirty={report['git']['dirty']})",
            f"- Target: {report['target']['chip']}, {report['target']['metal']}",
            f"- Duration: {report['duration_ns'] / 1e9:.3f} s",
        ]
        if report["reason"]:
            lines.append(f"- Reason: {report['reason']}")
        native = report["details"].get("native")
        if native:
            lines += [
                "",
                "## Native generation",
                "",
                f"- Plan: `{native.get('plan_id', '')}`",
                f"- Median seconds: **{native.get('seconds_median', 'n/a')}**",
                f"- Peak bytes: **{native.get('peak_bytes_max', 'n/a')}**",
                f"- Outputs identical: **{native.get('all_outputs_identical', 'n/a')}**",
            ]
        kernel = report["details"].get("kernel")
        if kernel:
            lines += [
                "",
                "## Kernel",
                "",
                f"- Operator: `{kernel.get('operator', '')}`",
                f"- Variant: `{kernel.get('variant', 'tile16')}`",
                f"- Shape: `{kernel.get('shape', {})}`",
                f"- Wall median: **{kernel.get('wall_median_ms', 'n/a')} ms**",
                f"- GPU median: **{kernel.get('gpu_median_ms', 'n/a')} ms**",
                f"- Numerical check: **{kernel.get('correct', False)}**",
            ]
        gates = report["details"].get("profile_gates")
        if gates:
            lines += ["", "## Profile gates", "", "| Matrix | Status | Comparative |", "| --- | --- | --- |"]
            for name, verdict in gates.items():
                lines.append(f"| {name} | {verdict['status']} | {verdict.get('comparative') or 'n/a'} |")
        lines += ["", "## Checks", "", "| Check | Status | Seconds | Exit |", "| --- | --- | ---: | ---: |"]
        for check in report["checks"]:
            lines.append(f"| {check['name']} | {check['status']} | {check['duration_ns'] / 1e9:.3f} | {check['returncode']} |")
        return "\n".join(lines) + "\n"


def build_commands(skip_build):
    configure = ["uvx", "--from", "cmake==4.4.5", "cmake", "-S", ".", "-B", ".build", "-DCMAKE_BUILD_TYPE=Release"]
    build = ["uvx", "--from", "cmake==4.4.5", "cmake", "--build", ".build", "-j", "8"]
    return [] if skip_build else [("cmake-configure", configure), ("cmake-build", build)]


def contract_steps(reporter, skip_build):
    reporter.run("uv-sync", ["uv", "sync", "--locked", "--extra", "reference"])
    reporter.run("schema-compile", ["uv", "run", "python", "tools/compile_schema.py"])
    reporter.run("schema-compatibility", ["uv", "run", "python", "tools/check_schema.py"])
    reporter.run("python-tests", ["uv", "run", "pytest"])
    reporter.run("python-compileall", ["uv", "run", "python", "-m", "compileall", "-q", "src", "tools", "benchmarks"])
    reporter.run("git-diff-check", ["git", "diff", "--check"])
    for name, command in build_commands(skip_build):
        reporter.run(name, command)
    reporter.run("metal-ctest", ["uvx", "--from", "cmake==4.4.5", "ctest", "--test-dir", ".build", "--output-on-failure"])


def kernel_step(reporter, variant, shape):
    if not (ROOT / ".build/onyx-metal-bench").exists():
        for name, command in build_commands(False):
            reporter.run(name, command)
    output = ROOT / "artifacts/ci/kernel-result.json"
    command = [
        str(ROOT / ".build/onyx-metal-bench"),
        "--m", str(shape["m"]), "--n", str(shape["n"]), "--k", str(shape["k"]),
        "--group", str(shape["group"]), "--iterations", str(shape["iterations"]),
        "--variant", variant, "--output", str(output),
    ]
    reporter.run(f"kernel-{variant}", command)
    result = json.loads(output.read_text())
    if not result.get("correct"):
        raise RuntimeError("Kernel numerical check failed")
    reporter.details["kernel"] = result


def require_h3_package():
    required = [
        "manifest.pb", "transformer.onxpack", "text_encoder.onxpack",
        "pdd_fused_heads.safetensors", "pdd_config.json",
        "vae/diffusion_pytorch_model.safetensors", "vae/config.json",
        "tokenizer/tokenizer.json",
    ]
    missing = [name for name in required if not (ROOT / "artifacts/models/h3-w4" / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing H3 package files: " + ", ".join(missing))


def native_steps(reporter, label, runs, case, compare):
    require_h3_package()
    reporter.run("native-generation", [
        "uv", "run", "python", "benchmarks/run_native.py",
        "--label", label, "--runs", str(runs), "--case", str(case), "--plan-id", label,
        "--target", str(reporter.target.resolve()),
    ], timeout=7200 * max(1, runs))
    summary_path = ROOT / "artifacts/benchmarks/onyx-native-h3" / f"{label}.json"
    reporter.details["native"] = json.loads(summary_path.read_text())
    if not reporter.details["native"].get("all_outputs_identical", False):
        raise RuntimeError("Native outputs are not deterministic across runs")
    if compare:
        reporter.run("community-comparison", ["uv", "run", "python", "benchmarks/compare_generation.py"])


def profiles_step(reporter):
    """Validate the profile registry and check committed measurements without Metal or model weights.
    """
    checker = ROOT / "tools/check_profile_gates.py"
    reporter.run("profile-registry", [sys.executable, str(checker), "--check-registry"])
    matrices = sorted((ROOT / "benchmarks/results").glob("profile-*-matrix*.json"))
    if not matrices:
        raise FileNotFoundError("No committed profile matrix under benchmarks/results (profile-*-matrix*.json)")
    verdicts = {}
    for path in matrices:
        matrix = json.loads(path.read_text())
        if (path.name == "profile-stage-matrix-v1.json" and
                matrix.get("registry_sha256") == "d78e589b50a5877bab4bff969bc03fab9277125d005dbe838370830bf4441d79"):
            print(f"[profile-gates] Historical matrix: {path.name} (registry SHA differs)", flush=True)
            verdicts[path.name] = {"status": "historical", "comparative": None}
            continue
        command = [sys.executable, str(checker), str(path)]
        report = json.loads(reporter.run(f"profile-gates:{path.name}", command))
        verdicts[path.name] = {
            "status": report["status"],
            "comparative": report["results"][0].get("comparative", {}).get("verdict"),
        }
    reporter.details["profile_gates"] = verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onyx-native-validate")
    parser.add_argument("--suite", choices=("contract", "metal", "kernel", "h3", "h3-e2e", "profiles", "all"), default="all")
    parser.add_argument("--label", default="local-validation")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--case", default="teapot-1024")
    parser.add_argument("--variant", choices=("tile16", "tile32"), default="tile16")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=21504)
    parser.add_argument("--k", type=int, default=5376)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Device capabilities JSON used in validation reports")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/ci")
    args = parser.parse_args(argv)
    reporter = Reporter(args.suite, args.label, args.variant, args.target.resolve())
    case = case_file(args.case)
    try:
        if args.suite in ("h3", "h3-e2e") and case is None:
            raise ValueError("H3 validation requires a benchmark case")
        if args.suite in ("contract", "all"):
            contract_steps(reporter, args.skip_build)
        if args.suite in ("metal", "all"):
            for name, command in build_commands(args.skip_build):
                reporter.run(name, command)
            reporter.run("metal-ctest", ["uvx", "--from", "cmake==4.4.5", "ctest", "--test-dir", ".build", "--output-on-failure"])
            kernel_step(reporter, args.variant, vars(args))
        if args.suite == "kernel":
            kernel_step(reporter, args.variant, vars(args))
        if args.suite in ("h3", "h3-e2e", "all"):
            native_steps(reporter, args.label, args.runs, case, args.suite == "h3-e2e")
        if args.suite == "profiles":
            profiles_step(reporter)
    except FileNotFoundError as error:
        reporter.block(str(error))
    except Exception as error:
        if reporter.status != "fail":
            reporter.status = "fail"
            reporter.reason = str(error)
        reporter.details["exception"] = traceback.format_exc(limit=4)
    json_path, _ = reporter.write(args.output_dir)
    return 0 if reporter.status == "pass" else 2 if reporter.status == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
