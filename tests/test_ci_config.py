from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_contract_and_uploads_reports():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "pull_request" in workflow[True]
    steps = workflow["jobs"]["contract-and-metal"]["steps"]
    text = "\n".join(str(step.get("run", "")) for step in steps)
    assert "--suite contract" in text
    assert "--suite metal" in text
    assert "--suite profiles" in text
    assert "native_validate.py" in text
    assert any("upload-artifact@v4" in str(step.get("uses", "")) for step in steps)


def test_profiles_suite_needs_no_gpu_or_model_package():
    """Profile checks must run without a Metal device."""
    source = (ROOT / "tools/native_validate.py").read_text()
    assert 'choices=("contract", "metal", "kernel", "h3", "h3-e2e", "profiles", "all")' in source
    assert 'glob("profile-*-matrix*.json")' in source


def test_native_validation_is_manual_and_hardware_scoped():
    workflow = yaml.safe_load((ROOT / ".github/workflows/native-validation.yml").read_text())
    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
    assert "pull_request_target" not in triggers
    runner = workflow["jobs"]["native"]["runs-on"]
    assert "self-hosted" in runner
    assert "onyx-metal" in runner
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "h3-e2e" in inputs["suite"]["options"]
    assert "teapot-4096" in inputs["case"]["options"]
    assert "tile32" in inputs["variant"]["options"]
