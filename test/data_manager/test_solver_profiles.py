from __future__ import annotations

import pytest

from hydrogym.data_manager import _SENTINEL_TO_PROFILE, SOLVER_PROFILES

_PROFILE_NAMES = sorted(SOLVER_PROFILES.keys())

_REQUIRED_KEYS = (
    "sentinel",
    "required_files",
    "required_dirs",
    "optional_files",
    "workspace_files",
    "workspace_dirs",
)

_EXPECTED_PROFILE_SET = {
    "MAIA_LB",
    "MAIA_STRCTRD",
    "JAX",
    "JAXFLUIDS",
    "NEK5000_v19",
    "NEK5000_v17",
    "FIREDRAKE",
}

_NEK_PROFILES = ("NEK5000_v19", "NEK5000_v17")

# Profiles that use glob-based workspace selection (workspace_glob_groups)
# instead of literal source filenames in workspace_files. Spec #8 says to
# specialize the workspace_files-subset assertion for non-glob profiles.
_GLOB_PROFILES = set(_NEK_PROFILES)

# Pinned latent anomaly: MAIA_LB lists "environment_config.yaml" as a key in
# workspace_files (hydrogym/data_manager.py line 87) but does not declare it
# in required_files or optional_files. The workspace-prep code at
# hydrogym/data_manager.py:451-462 silently logs a warning if the file is
# missing, so a typo here would not be caught by validation. This set pins
# the *current* exception; any *new* (profile, source) entry that violates
# the subset relation will fail the test.
_WORKSPACE_FILE_KEY_EXCEPTIONS: set[tuple[str, str]] = {
    ("MAIA_LB", "environment_config.yaml"),
}


@pytest.mark.parametrize("name", _PROFILE_NAMES)
def test_every_profile_has_required_keys(name):
    profile = SOLVER_PROFILES[name]
    for key in _REQUIRED_KEYS:
        assert key in profile, f"profile {name!r} missing required key {key!r}"

    assert isinstance(profile["sentinel"], str)
    assert profile["sentinel"].startswith(".")
    assert isinstance(profile["required_files"], list)
    assert isinstance(profile["required_dirs"], list)
    assert isinstance(profile["optional_files"], list)
    assert isinstance(profile["workspace_files"], dict)
    assert isinstance(profile["workspace_dirs"], dict)


def test_sentinels_unique():
    sentinels = {p["sentinel"] for p in SOLVER_PROFILES.values()}
    assert len(sentinels) == len(SOLVER_PROFILES)


@pytest.mark.parametrize("name", _PROFILE_NAMES)
def test_sentinels_start_with_dot(name):
    assert SOLVER_PROFILES[name]["sentinel"].startswith(".")


def test_sentinel_to_profile_cardinality_matches():
    assert len(_SENTINEL_TO_PROFILE) == len(SOLVER_PROFILES)


@pytest.mark.parametrize("name", _PROFILE_NAMES)
def test_sentinel_to_profile_round_trip(name):
    sentinel = SOLVER_PROFILES[name]["sentinel"]
    assert _SENTINEL_TO_PROFILE[sentinel] == name


@pytest.mark.parametrize("name", _NEK_PROFILES)
def test_nek_profiles_have_required_any_files(name):
    profile = SOLVER_PROFILES[name]
    assert "required_any_files" in profile
    assert "workspace_glob_groups" in profile
    raf = profile["required_any_files"]
    assert isinstance(raf, list)
    for group in raf:
        assert isinstance(group, list), f"{name} required_any_files group not a list: {group!r}"


@pytest.mark.parametrize("name", _NEK_PROFILES)
def test_nek_profiles_have_workspace_glob_groups_match_required_any(name):
    profile = SOLVER_PROFILES[name]
    required_any = [tuple(g) for g in profile["required_any_files"]]
    workspace_groups = [tuple(g) for g in profile["workspace_glob_groups"]]
    # Conservative pin: every workspace glob group must be a declared
    # required_any group, so workspace prep cannot try to link a pattern
    # that validation does not also require. NEK5000_v19 and NEK5000_v17
    # both have identical lists today (hydrogym/data_manager.py:142-163
    # for v19, :177-210 for v17).
    for group in workspace_groups:
        assert group in required_any, (
            f"{name}: workspace_glob_groups entry {group!r} not in required_any_files {required_any!r}"
        )


@pytest.mark.parametrize("name", _PROFILE_NAMES)
def test_workspace_files_target_keys_are_subset_of_known_sources(name):
    # For glob profiles (_GLOB_PROFILES) the bulk of source files are
    # matched via workspace_glob_groups, not via literal workspace_files
    # keys; only the literal entries (e.g. environment_config.yaml) are
    # checked here, which is the same as the non-glob path.
    profile = SOLVER_PROFILES[name]
    known_sources = set(profile["required_files"]) | set(profile["optional_files"])
    for source_name in profile["workspace_files"].keys():
        if (name, source_name) in _WORKSPACE_FILE_KEY_EXCEPTIONS:
            continue
        assert source_name in known_sources, (
            f"{name}: workspace_files source {source_name!r} not declared in "
            f"required_files or optional_files"
        )


def test_profile_names_are_uppercase_or_canonical():
    assert set(SOLVER_PROFILES.keys()) == _EXPECTED_PROFILE_SET
