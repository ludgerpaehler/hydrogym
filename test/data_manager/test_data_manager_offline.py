from __future__ import annotations

import inspect
import logging
import os

import pytest

import hydrogym.data_manager as dm_mod
from hydrogym.data_manager import HFDataManager


def _make_env(root, name: str, sentinels: list[str] | None = None, files: list[str] | None = None) -> str:
    env_dir = root / name
    env_dir.mkdir(parents=True, exist_ok=True)
    for sentinel in sentinels or []:
        (env_dir / sentinel).touch()
    for fname in files or []:
        (env_dir / fname).touch()
    return str(env_dir)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(dm_mod, "HF_AVAILABLE", False)
    return monkeypatch


def test_detect_solver_profile_local_cache_precedence(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()
    _make_env(cache_dir, "env1", sentinels=[".MAIA_LB"])
    _make_env(fallback_dir, "env1", sentinels=[".NEK5000"])

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    assert manager._detect_solver_profile("env1") == "MAIA_LB"


def test_detect_solver_profile_local_fallback_when_no_cache(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()
    _make_env(fallback_dir, "env1", sentinels=[".NEK5000"])

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    # .NEK5000 sentinel maps to NEK5000_v19 (data_manager.py:135 / _SENTINEL_TO_PROFILE)
    assert manager._detect_solver_profile("env1") == "NEK5000_v19"


def test_detect_solver_profile_falls_back_to_default(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()

    manager = HFDataManager(
        cache_dir=str(cache_dir),
        local_fallback_dir=str(fallback_dir),
        fallback_profile="JAXFLUIDS",
    )
    assert manager._detect_solver_profile("env1") == "JAXFLUIDS"


def test_detect_solver_profile_default_fallback_is_maia_lb():
    sig = inspect.signature(HFDataManager.__init__)
    assert sig.parameters["fallback_profile"].default == "MAIA_LB"


def test_download_environment_alias_of_get_environment_path(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))

    calls = []

    def fake_get(env_name, force_download=False):
        calls.append((env_name, force_download))
        return "/fake/path"

    manager.get_environment_path = fake_get  # type: ignore[method-assign]
    result = manager.download_environment("env_alpha", force_download=True)
    assert result == "/fake/path"
    assert calls == [("env_alpha", True)]


def test_get_available_environments_local_fallback_when_no_hf(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()
    _make_env(fallback_dir, "env_a")
    _make_env(fallback_dir, "env_b")

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    envs = manager.get_available_environments()
    assert sorted(envs) == ["env_a", "env_b"]


def test_get_available_environments_returns_empty_when_neither(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    assert manager.get_available_environments() == []


def test_get_environment_path_offline_uses_local_fallback(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()
    # JAXFLUIDS profile has no required files/dirs (data_manager.py:126-133) so validation auto-passes
    env_path = _make_env(fallback_dir, "env_x", sentinels=[".JAXFLUIDS"])

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    result = manager.get_environment_path("env_x")
    assert result == env_path
    assert os.path.isdir(result)


def test_get_environment_path_offline_raises_when_unavailable(tmp_path, offline):
    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()

    manager = HFDataManager(cache_dir=str(cache_dir), local_fallback_dir=str(fallback_dir))
    with pytest.raises(FileNotFoundError):
        manager.get_environment_path("missing")


def test_hf_unreachable_query_caught_logs_warning(tmp_path, monkeypatch, caplog):
    pytest.importorskip("huggingface_hub")
    import requests

    cache_dir = tmp_path / "cache"
    fallback_dir = tmp_path / "fallback"
    cache_dir.mkdir()
    fallback_dir.mkdir()

    # HF_AVAILABLE stays True so we exercise the network path
    def boom(self, *args, **kwargs):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(dm_mod.HfApi, "list_repo_files", boom)

    manager = HFDataManager(
        cache_dir=str(cache_dir),
        local_fallback_dir=str(fallback_dir),
        fallback_profile="JAXFLUIDS",
    )

    with caplog.at_level(logging.WARNING, logger="hydrogym.data_manager"):
        profile = manager._detect_solver_profile("nonexistent_env")

    assert profile == "JAXFLUIDS"
    assert any("Could not query HF for solver profile" in rec.message for rec in caplog.records)
