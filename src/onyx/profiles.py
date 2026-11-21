"""Load and validate the execution profile registry.

The registry declares settings and acceptance gates. The resolver validates
the registry and looks up profile values.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = ROOT / "config/profiles/profiles.v1.json"
REGISTRY_SCHEMA = "onyx.profiles.v1"
PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
NATIVE_STAGES = ("text", "dit", "vae")


class ProfileError(ValueError):
    """The registry or requested profile is invalid."""


def registry_digest(path: Path | None = None) -> str:
    """Return the SHA-256 that binds evidence to this registry."""
    return hashlib.sha256((path or DEFAULT_REGISTRY_PATH).read_bytes()).hexdigest()


def load_registry(path: Path | None = None) -> dict:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    try:
        registry = json.loads(registry_path.read_text())
    except OSError as error:
        raise ProfileError(f"Could not read profile registry: {registry_path} ({error})") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"Profile registry is not valid JSON: {registry_path}: {error}") from error
    errors = validate_registry(registry)
    if errors:
        raise ProfileError(f"Invalid profile registry {registry_path}:\n" + "\n".join(f"- {error}" for error in errors))
    return registry


def validate_registry(registry: object) -> list[str]:
    """Return structural errors, if any."""
    errors: list[str] = []
    if not isinstance(registry, dict):
        return [f"registry must be an object, got {type(registry).__name__}"]
    if registry.get("$schema") != REGISTRY_SCHEMA:
        errors.append(f"$schema must be {REGISTRY_SCHEMA!r}, got {registry.get('$schema')!r}")

    semantic = registry.get("model_semantic")
    if not isinstance(semantic, dict):
        errors.append("model_semantic must be an object")
    else:
        for key in ("numeric_profile", "quantization_profile", "plan_id"):
            if not isinstance(semantic.get(key), str) or not semantic[key]:
                errors.append(f"model_semantic.{key} must be a non-empty string")
        lock = semantic.get("recipe_lock")
        if not isinstance(lock, dict) or not isinstance(lock.get("nfe"), int) or lock.get("nfe", 0) <= 0 or not isinstance(lock.get("cfg"), bool):
            errors.append("model_semantic.recipe_lock must declare integer nfe > 0 and boolean cfg")

    domain = registry.get("knob_domain")
    if not isinstance(domain, dict) or not domain:
        errors.append("knob_domain must be a non-empty object mapping knob -> list of allowed values")
        domain = {}
    else:
        for knob, values in domain.items():
            if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
                errors.append(f"knob_domain.{knob} must be a non-empty list of strings")

    parameters = registry.get("knob_parameters", {})
    if not isinstance(parameters, dict):
        errors.append("knob_parameters must be an object when present")
        parameters = {}
    else:
        for parameter, mapping in parameters.items():
            if parameter.endswith("_note"):
                continue
            if not isinstance(mapping, dict):
                errors.append(f"knob_parameters.{parameter} must be an object")
            elif all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in mapping.values()):
                continue  # Numeric mapping, such as dit_block_cache_bytes.
            elif all(isinstance(value, str) and value for value in mapping.values()):
                continue  # Text mapping, such as q4_kernel_impl.
            else:
                errors.append(f"knob_parameters.{parameter} must map values to non-negative integers or non-empty strings")

    profiles = registry.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        errors.append("profiles must be a non-empty object")
        return errors
    for profile_id, profile in profiles.items():
        if not PROFILE_ID_PATTERN.match(str(profile_id)):
            errors.append(f"profile id {profile_id!r} must match {PROFILE_ID_PATTERN.pattern}")
        if not isinstance(profile, dict):
            errors.append(f"profile {profile_id} must be an object")
            continue
        for key in ("title", "summary"):
            if not isinstance(profile.get(key), str) or not profile[key]:
                errors.append(f"profile {profile_id}.{key} must be a non-empty string")
        knobs = profile.get("knobs")
        if not isinstance(knobs, dict):
            errors.append(f"profile {profile_id}.knobs must be an object")
            continue
        for knob, value in knobs.items():
            allowed = domain.get(knob)
            if allowed is None:
                errors.append(f"profile {profile_id} declares unknown knob {knob!r}")
            elif value not in allowed:
                errors.append(f"profile {profile_id} knob {knob}={value!r} not in knob_domain {allowed}")
            elif isinstance(parameters.get(f"{knob}_bytes"), dict) and value not in parameters[f"{knob}_bytes"]:
                errors.append(f"profile {profile_id} knob {knob}={value!r} missing from knob_parameters.{knob}_bytes")
        errors.extend(_validate_gates(profile_id, profile.get("gates"), profiles))
    return errors


def _validate_gates(profile_id: str, gates: object, profiles: dict) -> list[str]:
    if not isinstance(gates, dict):
        return [f"profile {profile_id}.gates must be an object"]
    errors: list[str] = []
    for key in ("output_replay_identical", "profile_echo_must_match"):
        if not isinstance(gates.get(key), bool):
            errors.append(f"profile {profile_id}.gates.{key} must be a boolean")
    peak = gates.get("peak_process_bytes_max")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        errors.append(f"profile {profile_id}.gates.peak_process_bytes_max must be a positive integer")
    samples = gates.get("min_samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        errors.append(f"profile {profile_id}.gates.min_samples must be a positive integer")
    bitwise = gates.get("bitwise_equal_to")
    numeric = gates.get("numeric_equal_to")
    if bitwise is not None:
        if bitwise not in profiles:
            errors.append(f"profile {profile_id}.gates.bitwise_equal_to {bitwise!r} is not a known profile")
        elif bitwise == profile_id:
            errors.append(f"profile {profile_id}.gates.bitwise_equal_to must reference a different profile")
        if numeric is not None:
            errors.append(f"profile {profile_id} cannot declare both bitwise_equal_to and numeric_equal_to")
    if numeric is not None:
        if not isinstance(numeric, dict):
            errors.append(f"profile {profile_id}.gates.numeric_equal_to must be an object")
        else:
            ref = numeric.get("reference")
            if ref not in profiles:
                errors.append(f"profile {profile_id}.gates.numeric_equal_to.reference {ref!r} is not a known profile")
            elif ref == profile_id:
                errors.append(f"profile {profile_id}.gates.numeric_equal_to.reference must reference a different profile")
            tol = numeric.get("max_abs_error")
            if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
                errors.append(f"profile {profile_id}.gates.numeric_equal_to.max_abs_error must be a positive number")
    fastest = gates.get("must_be_fastest_within")
    if fastest is not None:
        if not isinstance(fastest, dict):
            errors.append(f"profile {profile_id}.gates.must_be_fastest_within must be an object")
        else:
            margin = fastest.get("margin_ratio")
            if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not 0 < margin <= 1:
                errors.append(f"profile {profile_id}.gates.must_be_fastest_within.margin_ratio must be in (0, 1]")
            if fastest.get("compare_to") != "all":
                errors.append(f"profile {profile_id}.gates.must_be_fastest_within.compare_to must be \"all\"")
            if fastest.get("verdict_when_missed") not in ("inconclusive", "rejected"):
                errors.append(f"profile {profile_id}.gates.must_be_fastest_within.verdict_when_missed must be inconclusive|rejected")
            if fastest.get("verdict_when_slower_beyond_margin") != "rejected":
                errors.append(f"profile {profile_id}.gates.must_be_fastest_within.verdict_when_slower_beyond_margin must be \"rejected\"")
    return errors


def resolve_profile(registry: dict, profile_id: str) -> dict:
    """Resolve a profile or raise ProfileError with the valid names."""
    profiles = registry.get("profiles", {})
    if profile_id not in profiles:
        raise ProfileError(f"Unknown profile {profile_id!r}; available: {', '.join(sorted(profiles))}")
    profile = profiles[profile_id]
    return {"profile": profile, "knobs": dict(profile["knobs"]), "gates": dict(profile["gates"])}


def stage_weights(profile_id: str, knobs: dict, stage: str) -> str:
    """Return the stage weight policy. Text and DiT always stream."""
    if stage not in NATIVE_STAGES:
        raise ProfileError(f"Unknown native stage {stage!r}")
    key = f"{stage}_weights"
    if key not in knobs:
        raise ProfileError(f"Profile {profile_id} does not declare knob {key!r}")
    if stage in ("text", "dit") and knobs[key] != "stream":
        raise ProfileError(f"Profile {profile_id} requires streaming weights for the {stage} stage")
    return str(knobs[key])


def dit_cache_bytes(registry: dict, profile_id: str) -> int:
    """Return the DiT LRU budget in bytes. The none policy has a zero budget."""
    knobs = registry.get("profiles", {}).get(profile_id, {}).get("knobs")
    if not knobs:
        raise ProfileError(f"Unknown profile {profile_id!r}")
    mode = knobs.get("dit_block_cache", "none")
    mapping = registry.get("knob_parameters", {}).get("dit_block_cache_bytes")
    if not isinstance(mapping, dict) or mode not in mapping:
        raise ProfileError(f"Registry is missing dit_block_cache_bytes[{mode!r}]")
    value = mapping[mode]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProfileError(f"dit_block_cache_bytes[{mode!r}] must be a non-negative integer")
    return value


def choice_help(registry: dict) -> str:
    return ", ".join(sorted(registry.get("profiles", {})))
