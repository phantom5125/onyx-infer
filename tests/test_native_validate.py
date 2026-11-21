import json
import sys

import pytest

from tools.native_validate import Reporter, case_file


def test_reporter_writes_structured_pass_report(tmp_path):
    reporter = Reporter("contract", "unit-test", "tile16")
    reporter.run("smoke", [sys.executable, "-c", "print('ok')"])
    reporter.write(tmp_path)
    report = json.loads((tmp_path / "unit-test.json").read_text())
    assert report["protocol"] == "onyx.ci.v1"
    assert report["status"] == "pass"
    assert report["checks"][0]["returncode"] == 0
    assert report["git"]["revision"]
    assert "Onyx validation" in (tmp_path / "unit-test.md").read_text()


def test_reporter_records_failure_and_blocked_state(tmp_path):
    reporter = Reporter("contract", "failure-test", "tile16")
    with pytest.raises(RuntimeError):
        reporter.run("expected-failure", [sys.executable, "-c", "raise SystemExit(3)"])
    reporter.write(tmp_path)
    report = json.loads((tmp_path / "failure-test.json").read_text())
    assert report["status"] == "fail"
    assert report["checks"][0]["returncode"] == 3
    assert "expected-failure" in report["reason"]

    blocked = Reporter("h3", "blocked-test", "tile16")
    blocked.block("model package is not staged")
    blocked.write(tmp_path)
    blocked_report = json.loads((tmp_path / "blocked-test.json").read_text())
    assert blocked_report["status"] == "blocked"
    assert blocked_report["details"]["blocked_reason"]


def test_report_label_is_filesystem_safe(tmp_path):
    reporter = Reporter("contract", "../../unsafe label", "tile16")
    assert reporter.label == "unsafe-label"
    reporter.write(tmp_path)
    assert (tmp_path / "unsafe-label.json").is_file()


def test_case_aliases_resolve_to_real_requests():
    assert case_file("teapot-1024").is_file()
    assert case_file("teapot-4096").is_file()
    assert case_file("metal-only") is None
