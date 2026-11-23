import copy
import os
from pathlib import Path

import pytest

from onyx.native import (
    DEFAULT_MODULATION_CACHE,
    main as native_main,
    prune_modulation_cache,
    resolve_modulation_cache,
    resolve_dit_prefetch,
    resolve_weight_prefetch,
)
from onyx.profiles import (
    DEFAULT_REGISTRY_PATH,
    ProfileError,
    dit_cache_bytes,
    load_registry,
    registry_digest,
    resolve_profile,
    stage_weights,
    validate_registry,
)


EXPECTED_PROFILES = {"max-quality", "memory-min", "m5-fast", "speed-focus"}


def test_registry_loads_and_declares_four_profiles():
    registry = load_registry()
    assert set(registry["profiles"]) == EXPECTED_PROFILES
    assert registry["$schema"] == "onyx.profiles.v1"
    assert registry["model_semantic"]["recipe_lock"]["nfe"] == 6


def test_default_registry_exists_and_is_structurally_valid():
    assert DEFAULT_REGISTRY_PATH.is_file()
    assert validate_registry(load_registry()) == []


def test_knob_domain_respected_by_every_profile():
    registry = load_registry()
    domain = registry["knob_domain"]
    for profile_id, profile in registry["profiles"].items():
        for knob, value in profile["knobs"].items():
            assert knob in domain, profile_id
            assert value in domain[knob], profile_id


def test_streaming_only_stages_stay_streaming():
    registry = load_registry()
    for profile_id in EXPECTED_PROFILES:
        knobs = registry["profiles"][profile_id]["knobs"]
        assert knobs["text_weights"] == "stream"
        assert knobs["dit_weights"] == "stream"


def test_resolve_profile_returns_knobs_and_gates():
    registry = load_registry()
    resolved = resolve_profile(registry, "memory-min")
    assert resolved["knobs"]["vae_weights"] == "stream"
    assert resolved["gates"]["bitwise_equal_to"] == "max-quality"
    assert resolved["gates"]["peak_process_bytes_max"] > 0


def test_resolve_profile_unknown_lists_choices():
    registry = load_registry()
    with pytest.raises(ProfileError) as error:
        resolve_profile(registry, "turbo")
    message = str(error.value)
    for profile_id in EXPECTED_PROFILES:
        assert profile_id in message


def test_dit_block_cache_knob_layout():
    registry = load_registry()
    assert registry["profiles"]["speed-focus"]["knobs"]["dit_block_cache"] == "lru"
    for profile_id in ("max-quality", "memory-min"):
        assert registry["profiles"][profile_id]["knobs"]["dit_block_cache"] == "none"


def test_q4_kernel_knob_layout():
    registry = load_registry()
    assert registry["profiles"]["max-quality"]["knobs"]["q4_kernel"] == "mpsgraph"
    assert registry["profiles"]["memory-min"]["knobs"]["q4_kernel"] == "mpsgraph"
    assert registry["profiles"]["speed-focus"]["knobs"]["q4_kernel"] == "simd"
    # m5-fast keeps every max-quality setting except the Q4 kernel.
    assert registry["profiles"]["m5-fast"]["knobs"]["q4_kernel"] == "nax"
    fast = dict(registry["profiles"]["m5-fast"]["knobs"], q4_kernel="mpsgraph")
    assert fast == registry["profiles"]["max-quality"]["knobs"]
    parameters = registry["knob_parameters"]["q4_kernel_impl"]
    assert set(parameters) == {"mpsgraph", "steel", "simd", "nax"}


def test_numeric_equal_to_gate():
    registry = load_registry()
    gates = registry["profiles"]["speed-focus"]["gates"]
    assert "bitwise_equal_to" not in gates
    num = gates["numeric_equal_to"]
    assert num["reference"] == "max-quality"
    assert num["max_abs_error"] > 0
    # memory-min uses MPSGraph and requires bitwise agreement.
    assert "bitwise_equal_to" in registry["profiles"]["memory-min"]["gates"]
    parameters = registry["knob_parameters"]["dit_block_cache_bytes"]
    assert parameters["none"] == 0
    assert parameters["lru"] > 0


def test_weight_prefetch_knob_layout():
    registry = load_registry()
    assert registry["knob_domain"]["weight_prefetch"] == ["off", "next"]
    # Only the default profile and m5-fast stream ahead; memory-min keeps its peak-memory
    # gate and speed-focus caches DiT weights instead.
    assert registry["profiles"]["max-quality"]["knobs"]["weight_prefetch"] == "next"
    assert registry["profiles"]["m5-fast"]["knobs"]["weight_prefetch"] == "next"
    for profile_id in ("memory-min", "speed-focus"):
        assert registry["profiles"][profile_id]["knobs"]["weight_prefetch"] == "off"


def test_resolve_weight_prefetch_follows_profile_unless_requested():
    assert resolve_weight_prefetch("next", None)
    assert not resolve_weight_prefetch("off", None)
    assert resolve_weight_prefetch("off", True)
    assert not resolve_weight_prefetch("next", False)
    with pytest.raises(ProfileError):
        resolve_weight_prefetch("maybe", None)


def test_resolve_modulation_cache_defaults_to_the_user_cache():
    chosen = Path("/tmp/chosen-modulation")
    assert resolve_modulation_cache(None, False) == DEFAULT_MODULATION_CACHE
    assert resolve_modulation_cache(chosen, False) == chosen
    assert resolve_modulation_cache(None, True) is None
    assert resolve_modulation_cache(chosen, True) is None


def test_prune_modulation_cache_keeps_the_newest_entries(tmp_path: Path):
    names = [f"{index:064x}.adaln" for index in range(4)]
    for index, name in enumerate(names):
        entry = tmp_path / name
        entry.write_bytes(b"x")
        os.utime(entry, (1000 + index, 1000 + index))
    (tmp_path / "notes.txt").write_text("not an entry")
    assert prune_modulation_cache(tmp_path, keep=2) == names[:2]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names[2:] + ["notes.txt"])
    assert prune_modulation_cache(tmp_path, keep=2) == []


def test_dit_prefetch_yields_to_the_weight_cache_unless_requested():
    # The profile default: prefetch while streaming, and let a weight cache replace it.
    assert resolve_dit_prefetch(True, None, False, 0)
    assert not resolve_dit_prefetch(True, None, False, 10240 * 2**20)
    # An explicit request wins in both directions.
    assert resolve_dit_prefetch(False, True, False, 0)
    assert not resolve_dit_prefetch(True, False, False, 0)
    assert not resolve_dit_prefetch(True, False, False, 10240 * 2**20)
    assert resolve_dit_prefetch(False, None, True, 0)
    # Requesting prefetch next to a cache returns True, which main() turns into an error.
    assert resolve_dit_prefetch(True, True, False, 10240 * 2**20)
    assert resolve_dit_prefetch(True, None, True, 10240 * 2**20)


def test_weight_cache_budget_replaces_dit_prefetch(capsys):
    # The documented tuning, --dit-cache-mib 10240 with the default profile, must parse. The
    # run then stops at the missing tokenizer of the fixture instead of at an argument error.
    with pytest.raises(Exception) as error:
        native_main(["--models", "fixture", "--prompt", "test", "--output", "fixture.png", "--dit-cache-mib", "10240"])
    assert "prefetch" not in str(error.value)
    assert "prefetch" not in capsys.readouterr().err
    # Asking for both is still an error.
    with pytest.raises(SystemExit) as exit:
        native_main(["--models", "fixture", "--prompt", "test", "--output", "fixture.png",
                     "--dit-cache-mib", "10240", "--dit-prefetch"])
    assert exit.value.code == 2
    assert "requires a zero DiT weight-cache budget" in capsys.readouterr().err


def test_dit_cache_bytes_resolution():
    registry = load_registry()
    assert dit_cache_bytes(registry, "speed-focus") == registry["knob_parameters"]["dit_block_cache_bytes"]["lru"]
    assert dit_cache_bytes(registry, "max-quality") == 0
    assert dit_cache_bytes(registry, "memory-min") == 0
    with pytest.raises(ProfileError):
        dit_cache_bytes(registry, "turbo")


@pytest.mark.parametrize("budget", ["-1", str(2**44)])
def test_native_rejects_unrepresentable_dit_cache_budget(budget, capsys):
    with pytest.raises(SystemExit) as exit:
        native_main(["--models", "fixture", "--prompt", "test", "--output", "fixture.png", "--dit-cache-mib", budget])
    assert exit.value.code == 2
    assert "nonnegative budget representable in 64-bit bytes" in capsys.readouterr().err


def test_knob_parameter_mapping_is_validated():
    registry = load_registry()
    broken = copy.deepcopy(registry)
    del broken["knob_parameters"]["dit_block_cache_bytes"]["lru"]
    assert any("dit_block_cache_bytes" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["knob_parameters"]["dit_block_cache_bytes"]["lru"] = -1
    assert any("non-negative integers" in error for error in validate_registry(broken))


def test_stage_weights_rejects_resident_for_streaming_stages():
    with pytest.raises(ProfileError):
        stage_weights("fake", {"text_weights": "resident", "dit_weights": "stream", "vae_weights": "stream"}, "text")
    assert stage_weights("fake", {"vae_weights": "resident"}, "vae") == "resident"
    with pytest.raises(ProfileError):
        stage_weights("fake", {}, "vae")
    with pytest.raises(ProfileError):
        stage_weights("fake", {"vae_weights": "stream"}, "no-such-stage")


def test_validate_registry_reports_structural_errors():
    registry = load_registry()

    broken = copy.deepcopy(registry)
    broken["profiles"]["max-quality"]["knobs"]["vae_weights"] = "fuzzy"
    assert any("not in knob_domain" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["profiles"]["max-quality"]["knobs"]["nope"] = "stream"
    assert any("unknown knob" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["profiles"]["memory-min"]["gates"]["peak_process_bytes_max"] = -1
    assert any("peak_process_bytes_max" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["profiles"]["memory-min"]["gates"]["bitwise_equal_to"] = "memory-min"
    assert any("different profile" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["profiles"]["speed-focus"]["gates"]["must_be_fastest_within"]["margin_ratio"] = 1.5
    assert any("margin_ratio" in error for error in validate_registry(broken))

    broken = copy.deepcopy(registry)
    broken["model_semantic"]["recipe_lock"]["nfe"] = 0
    assert any("recipe_lock" in error for error in validate_registry(broken))


def test_registry_digest_binds_evidence_to_registry_version(tmp_path: Path):
    copied = tmp_path / "profiles.json"
    copied.write_text(DEFAULT_REGISTRY_PATH.read_text())
    assert registry_digest(copied) == registry_digest()
    copied.write_text(copied.read_text() + "\n")
    assert registry_digest(copied) != registry_digest()


def test_load_registry_fails_closed_on_unknown_file(tmp_path: Path):
    with pytest.raises(ProfileError):
        load_registry(tmp_path / "missing.json")
