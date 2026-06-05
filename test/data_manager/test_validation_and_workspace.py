from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from hydrogym.data_manager import SOLVER_PROFILES, HFDataManager

# ---------------------------------------------------------------------------
# Helpers (local to this module by design — no shared conftest fixtures)
# ---------------------------------------------------------------------------


def _make_required_file(root: Path, name: str) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _make_required_dir(root: Path, name: str, *, populate: bool = True) -> Path:
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    if populate:
        (p / "_placeholder").touch()
    return p


def _materialize_profile(env_dir: Path, profile: str) -> None:
    """Create files/dirs that satisfy the *required* parts of a profile."""
    spec = SOLVER_PROFILES[profile]
    for name in spec.get("required_files", []):
        _make_required_file(env_dir, name)
    for name in spec.get("required_dirs", []):
        _make_required_dir(env_dir, name, populate=True)
    # required_any_files is list of pattern groups; satisfy each with a concrete file.
    for pattern_group in spec.get("required_any_files", []):
        first = pattern_group[0]
        concrete = first.replace("*", "case")
        _make_required_file(env_dir, concrete)


def _make_manager(tmp_path: Path, **kwargs) -> HFDataManager:
    kwargs.setdefault("repo_id", "dynamicslab/HydroGym-environments")
    kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
    return HFDataManager(**kwargs)


# ---------------------------------------------------------------------------
# _validate_environment_files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    ["MAIA_LB", "MAIA_STRCTRD", "JAX", "JAXFLUIDS", "NEK5000_v19", "FIREDRAKE"],
)
def test_validate_all_required_present_returns_true(tmp_path, profile):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, profile)
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile=profile) is True


def test_validate_missing_required_file_returns_false(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, "MAIA_LB")
    # Delete a required file
    (env_dir / "geometry.toml").unlink()
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile="MAIA_LB") is False


@pytest.mark.parametrize("profile", ["MAIA_LB", "NEK5000_v19"])
def test_validate_empty_required_dir_returns_false_for_strict_profiles(tmp_path, profile):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, profile)
    # Empty the required dir(s); pins os.listdir == [] check in data_manager.py:699
    for dname in SOLVER_PROFILES[profile]["required_dirs"]:
        d = env_dir / dname
        for child in d.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile=profile) is False


def test_validate_unknown_profile_returns_true_with_warning(tmp_path, caplog):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    dm = _make_manager(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hydrogym.data_manager"):
        # pins data_manager.py:670-672 -- unknown profile silently passes with warning
        result = dm._validate_environment_files(str(env_dir), profile="DOES_NOT_EXIST")
    assert result is True
    assert any("Unknown solver profile" in r.message for r in caplog.records)


def test_validate_firedrake_short_circuits_true_with_empty_dir(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    dm = _make_manager(tmp_path)
    # FIREDRAKE has empty required_files/required_dirs (data_manager.py:215-224)
    assert dm._validate_environment_files(str(env_dir), profile="FIREDRAKE") is True


def test_validate_jaxfluids_short_circuits_true_with_empty_dir(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile="JAXFLUIDS") is True


def test_validate_nek_required_any_files_all_groups_match(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _make_required_file(env_dir, "environment_config.yaml")
    _make_required_file(env_dir, "int_pos")
    _make_required_dir(env_dir, "restart_files", populate=True)
    _make_required_file(env_dir, "case.re2")
    _make_required_file(env_dir, "case.ma2")
    _make_required_file(env_dir, "case.par")
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile="NEK5000_v19") is True


def test_validate_nek_missing_one_required_any_group_returns_false(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _make_required_file(env_dir, "environment_config.yaml")
    _make_required_file(env_dir, "int_pos")
    _make_required_dir(env_dir, "restart_files", populate=True)
    _make_required_file(env_dir, "case.re2")
    _make_required_file(env_dir, "case.ma2")
    # no .par file
    dm = _make_manager(tmp_path)
    assert dm._validate_environment_files(str(env_dir), profile="NEK5000_v19") is False


# ---------------------------------------------------------------------------
# prepare_working_directory
# ---------------------------------------------------------------------------


def test_prepare_working_directory_explicit_profile(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, "MAIA_LB")
    # MAIA_LB workspace_files also includes environment_config.yaml (optional source)
    _make_required_file(env_dir, "environment_config.yaml")

    work_dir = tmp_path / "work"
    dm = _make_manager(tmp_path)
    paths = dm.prepare_working_directory(str(env_dir), str(work_dir), profile="MAIA_LB")

    assert paths["solver_profile"] == "MAIA_LB"
    for target_rel in SOLVER_PROFILES["MAIA_LB"]["workspace_files"].values():
        target = work_dir / target_rel
        assert target.is_symlink(), f"expected symlink at {target}"
    for target_rel in SOLVER_PROFILES["MAIA_LB"]["workspace_dirs"].values():
        target = work_dir / target_rel
        assert target.is_symlink(), f"expected dir symlink at {target}"


def test_prepare_working_directory_auto_detect_profile_from_sentinel(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, "MAIA_LB")
    _make_required_file(env_dir, "environment_config.yaml")
    # Drop the MAIA_LB sentinel
    (env_dir / ".MAIA_LB").touch()

    work_dir = tmp_path / "work"
    # fallback set to a different profile so we can prove auto-detection won
    dm = _make_manager(tmp_path, fallback_profile="JAX")
    paths = dm.prepare_working_directory(str(env_dir), str(work_dir), profile=None)
    assert paths["solver_profile"] == "MAIA_LB"


def test_prepare_working_directory_no_sentinel_uses_fallback(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, "MAIA_LB")
    _make_required_file(env_dir, "environment_config.yaml")
    # No sentinel file present

    work_dir = tmp_path / "work"
    dm = _make_manager(tmp_path, fallback_profile="MAIA_LB")
    paths = dm.prepare_working_directory(str(env_dir), str(work_dir), profile=None)
    assert paths["solver_profile"] == "MAIA_LB"


def test_prepare_working_directory_missing_source_logs_warning_not_raises(tmp_path, caplog):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _materialize_profile(env_dir, "MAIA_LB")
    # Intentionally do NOT create environment_config.yaml, which is in workspace_files
    # but NOT in required_files. prepare_working_directory should warn, not raise.

    work_dir = tmp_path / "work"
    dm = _make_manager(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hydrogym.data_manager"):
        # pins data_manager.py:462 -- missing source warns and skips
        paths = dm.prepare_working_directory(str(env_dir), str(work_dir), profile="MAIA_LB")

    assert paths["solver_profile"] == "MAIA_LB"
    assert any("Workspace source not found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _link_path branches
# ---------------------------------------------------------------------------


def test_link_path_creates_new_symlink(tmp_path):
    dm = _make_manager(tmp_path)
    src = tmp_path / "a"
    src.touch()
    dst = tmp_path / "b"
    dm._link_path(src, dst)
    assert dst.is_symlink()
    # pins data_manager.py:725 -- symlink_to(source.resolve()) stores an absolute path,
    # NOT the original name "a"
    assert dst.readlink() == src.resolve()


def test_link_path_replaces_existing_symlink(tmp_path):
    dm = _make_manager(tmp_path)
    old_target = tmp_path / "old"
    old_target.touch()
    dst = tmp_path / "b"
    dst.symlink_to(old_target.resolve())
    assert dst.is_symlink()

    new_target = tmp_path / "new"
    new_target.touch()
    dm._link_path(new_target, dst)
    assert dst.is_symlink()
    assert dst.readlink() == new_target.resolve()


def test_link_path_removes_existing_dir_via_rmtree(tmp_path):
    dm = _make_manager(tmp_path)
    dst = tmp_path / "b"
    dst.mkdir()
    (dst / "inner.txt").write_text("hello")
    (dst / "nested").mkdir()
    (dst / "nested" / "more.txt").write_text("world")

    src = tmp_path / "a"
    src.touch()
    dm._link_path(src, dst)
    assert dst.is_symlink()
    # The old dir contents should no longer be accessible as real files
    assert dst.readlink() == src.resolve()


def test_link_path_removes_existing_file(tmp_path):
    dm = _make_manager(tmp_path)
    dst = tmp_path / "b"
    dst.write_text("old contents")
    src = tmp_path / "a"
    src.touch()
    dm._link_path(src, dst)
    assert dst.is_symlink()
    assert dst.readlink() == src.resolve()


def test_link_path_creates_parent_dirs(tmp_path):
    dm = _make_manager(tmp_path)
    src = tmp_path / "a"
    src.touch()
    dst = tmp_path / "sub" / "deep" / "b"
    assert not dst.parent.exists()
    dm._link_path(src, dst)
    assert dst.parent.is_dir()
    assert dst.is_symlink()


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_env_name_removes_symlink(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    real_target = tmp_path / "real"
    real_target.mkdir()
    (cache_dir / "env1").symlink_to(real_target.resolve())

    dm = HFDataManager(cache_dir=str(cache_dir), use_clean_cache=True)
    dm.clear_cache("env1")
    assert not (cache_dir / "env1").exists()
    assert not (cache_dir / "env1").is_symlink()
    # Real target untouched
    assert real_target.exists()


def test_clear_cache_env_name_removes_directory(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    env_path = cache_dir / "env1"
    env_path.mkdir()
    (env_path / "file.txt").write_text("data")

    dm = HFDataManager(cache_dir=str(cache_dir), use_clean_cache="copy")
    dm.clear_cache("env1")
    assert not env_path.exists()


def test_clear_cache_all_wipes_and_recreates(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for name in ("envA", "envB", "envC"):
        d = cache_dir / name
        d.mkdir()
        (d / "x").touch()

    dm = HFDataManager(cache_dir=str(cache_dir), use_clean_cache=True)
    dm.clear_cache()
    assert cache_dir.exists()
    assert list(cache_dir.iterdir()) == []


def test_clear_cache_with_clean_cache_false_is_noop(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Note: with use_clean_cache=False, cache_dir is NOT auto-created in __init__,
    # so create it manually to ensure the noop branch does not touch it.
    sentinel = cache_dir / "marker"
    sentinel.touch()

    dm = HFDataManager(cache_dir=str(cache_dir), use_clean_cache=False)
    # pins data_manager.py:739-741 -- prints note and returns; no raise
    dm.clear_cache("anything")
    out = capsys.readouterr().out
    assert "huggingface_hub" in out
    # Cache contents are untouched
    assert sentinel.exists()


# ---------------------------------------------------------------------------
# HF_AVAILABLE = False degradation
# ---------------------------------------------------------------------------


def test_manager_init_warns_when_hf_unavailable(tmp_path, monkeypatch, caplog):
    import hydrogym.data_manager as dm_mod

    monkeypatch.setattr(dm_mod, "HF_AVAILABLE", False)
    with caplog.at_level(logging.WARNING, logger="hydrogym.data_manager"):
        HFDataManager(cache_dir=str(tmp_path / "cache"), use_clean_cache=True)
    # pins data_manager.py:288-289 -- emits a warning when HF Hub is unavailable
    assert any("Hugging Face Hub not available" in r.message for r in caplog.records)
