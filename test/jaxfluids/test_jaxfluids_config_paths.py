"""Tests for the JAX-Fluids backend's config-resolution surface.

Focused on:

* the small public exception type (``ConfigError``),
* the ``_init_from_hf`` guard rails for missing ``environment_name`` /
  configuration files,
* the four-case ``_resolve_configuration_file`` resolver and its
  ``_find_configuration_file`` helper,
* a regression guard that the ``JAXFLUIDS`` solver profile stays
  registered in :data:`hydrogym.data_manager.SOLVER_PROFILES`,
* a cross-backend drift guard: the same ``_resolve_configuration_file``
  contract exists verbatim in ``hydrogym.maia.env_core`` and
  ``hydrogym.jax.env_core``; this test exercises all three resolvers
  with the same inputs and asserts they return the same paths.

The whole module is skipped when ``jaxfluids_rl`` is not installed
(typical on CI runners and on hosts without the JAX-Fluids stack).
The cross-backend test skips the MAIA / JAX variants individually
when their import-time dependencies (``mpi4py``, ``jax``) are missing.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.requires_jaxfluids

# Skip the entire module if the JAX-Fluids runtime is not installed —
# importing hydrogym.jaxfluids.env_core requires jaxfluids_rl at module load.
pytest.importorskip("jaxfluids_rl", reason="JAX-Fluids backend tests need jaxfluids installed")

import hydrogym.data_manager  # noqa: E402
from hydrogym.jaxfluids import env_core as jf_env_core  # noqa: E402


def _make_bare_env(env_data_path: str | None = None, environment_name: str = "stub_env"):
    """Build a JAXFluidsFlowEnv without invoking JAXFluidsEnv.__init__.

    The resolver methods only read ``self.env_data_path`` and
    ``self.environment_name``; constructing the full upstream env
    would require a real configuration tree on disk.
    """
    inst = object.__new__(jf_env_core.JAXFluidsFlowEnv)
    if env_data_path is not None:
        inst.env_data_path = env_data_path
    inst.environment_name = environment_name
    return inst


# ----------------------------------------------------------------------
# ConfigError shape
# ----------------------------------------------------------------------


def test_config_error_is_exception_subclass():
    """``ConfigError`` must be a plain Exception subclass — callers catch it as such."""
    assert issubclass(jf_env_core.ConfigError, Exception)


# ----------------------------------------------------------------------
# _init_from_hf guard rails
# ----------------------------------------------------------------------


def test_init_from_hf_missing_name_raises():
    """No ``environment_name`` in env_config -> ConfigError before any HF I/O matters."""
    inst = object.__new__(jf_env_core.JAXFluidsFlowEnv)
    # Patch HFDataManager to a no-op so the constructor doesn't try to
    # mkdir the cache or touch the network.
    with patch.object(jf_env_core, "HFDataManager", autospec=True) as mock_dm:
        mock_dm.return_value = object()
        with pytest.raises(jf_env_core.ConfigError, match="environment_name"):
            jf_env_core.JAXFluidsFlowEnv._init_from_hf(inst, {})


def test_init_from_hf_missing_config_raises(tmp_path):
    """Environment name supplied, but no config file in env_data_path -> ConfigError."""
    inst = object.__new__(jf_env_core.JAXFluidsFlowEnv)

    # Empty env data dir -> _find_configuration_file returns None
    env_dir = tmp_path / "envdata"
    env_dir.mkdir()

    fake_dm = type("FakeDM", (), {"get_environment_path": lambda self, name: str(env_dir)})()

    with patch.object(jf_env_core, "HFDataManager", autospec=True) as mock_dm:
        mock_dm.return_value = fake_dm
        env_config = {"environment_name": "any_env"}
        with pytest.raises(jf_env_core.ConfigError, match="No configuration file"):
            jf_env_core.JAXFluidsFlowEnv._init_from_hf(inst, env_config)


# ----------------------------------------------------------------------
# _resolve_configuration_file: the four input cases
# ----------------------------------------------------------------------


def test_resolve_configuration_file_none(tmp_path):
    """None input delegates to _find_configuration_file and picks up config.yaml."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dummy: 1\n")

    inst = _make_bare_env(env_data_path=str(tmp_path))
    resolved = inst._resolve_configuration_file(None)
    assert resolved == str(cfg)


def test_resolve_configuration_file_absolute(tmp_path):
    """Absolute path that exists is returned verbatim; missing absolute path -> ConfigError."""
    cfg = tmp_path / "abs_config.yaml"
    cfg.write_text("dummy: 1\n")

    inst = _make_bare_env(env_data_path=str(tmp_path))
    resolved = inst._resolve_configuration_file(str(cfg))
    assert resolved == str(cfg)

    missing = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises(jf_env_core.ConfigError, match="not found"):
        inst._resolve_configuration_file(missing)


def test_resolve_configuration_file_relative(tmp_path, monkeypatch):
    """Inputs starting with ./ or ../ are resolved against cwd via os.path.abspath."""
    cfg = tmp_path / "rel_config.yaml"
    cfg.write_text("dummy: 1\n")

    # Chdir so "./rel_config.yaml" lives next to us
    monkeypatch.chdir(tmp_path)

    inst = _make_bare_env(env_data_path=str(tmp_path))
    resolved = inst._resolve_configuration_file("./rel_config.yaml")
    assert os.path.abspath(resolved) == str(cfg)

    # Missing relative -> ConfigError mentioning the resolved absolute path
    with pytest.raises(jf_env_core.ConfigError, match="not found"):
        inst._resolve_configuration_file("./missing.yaml")


def test_resolve_configuration_file_bare(tmp_path, monkeypatch):
    """Bare filename: first checks cwd, then env_data_path; falls through to ConfigError."""
    env_dir = tmp_path / "envdata"
    env_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    inst = _make_bare_env(env_data_path=str(env_dir))

    # 1) bare filename present in env_data_path
    env_cfg = env_dir / "bare.yaml"
    env_cfg.write_text("dummy: 1\n")
    resolved = inst._resolve_configuration_file("bare.yaml")
    assert resolved == str(env_cfg)

    # 2) bare filename also present in cwd should win over env_data_path
    cwd_cfg = cwd_dir / "bare.yaml"
    cwd_cfg.write_text("dummy: 2\n")
    resolved = inst._resolve_configuration_file("bare.yaml")
    assert os.path.abspath(resolved) == str(cwd_cfg)

    # 3) bare filename present in neither -> ConfigError
    with pytest.raises(jf_env_core.ConfigError, match="not found"):
        inst._resolve_configuration_file("totally_missing.yaml")


# ----------------------------------------------------------------------
# _find_configuration_file priority
# ----------------------------------------------------------------------


def test_find_configuration_file_prefers_environment_config(tmp_path):
    """When both ``config.yaml`` and ``environment_config.yaml`` exist,
    ``config.yaml`` wins — that's the first name in the search list.
    """
    (tmp_path / "config.yaml").write_text("a: 1\n")
    (tmp_path / "environment_config.yaml").write_text("b: 2\n")

    inst = _make_bare_env(env_data_path=str(tmp_path))
    resolved = inst._find_configuration_file()
    assert resolved == str(tmp_path / "config.yaml")


# ----------------------------------------------------------------------
# Regression: JAXFLUIDS solver profile stays wired in data_manager
# ----------------------------------------------------------------------


def test_solver_profiles_has_jaxfluids_entry():
    """JAXFluidsFlowEnv hard-codes ``fallback_profile='JAXFLUIDS'``; that key must exist."""
    assert "JAXFLUIDS" in hydrogym.data_manager.SOLVER_PROFILES
    profile = hydrogym.data_manager.SOLVER_PROFILES["JAXFLUIDS"]
    # Profile contract: every entry has a sentinel filename used for auto-detection.
    assert "sentinel" in profile
    assert profile["sentinel"] == ".JAXFLUIDS"


# ----------------------------------------------------------------------
# Cross-backend drift guard
# ----------------------------------------------------------------------


def _try_import(module_name: str):
    """Return the imported module, or None if any import-time dep is missing."""
    try:
        import importlib

        return importlib.import_module(module_name)
    except Exception:
        return None


def _make_bare_for_backend(backend_module, env_data_path: str, environment_name: str = "stub_env"):
    """Build a bare instance of the backend's flow env class with the resolver attrs set.

    Each backend exposes a different class name; their resolvers all read
    ``self.env_data_path`` / ``self.environment_name`` though, so we just
    pick whichever ``*FlowEnv`` class lives on the module.
    """
    cls = None
    for name in ("MaiaFlowEnv", "JAXFlowEnv", "JAXFluidsFlowEnv"):
        if hasattr(backend_module, name):
            cls = getattr(backend_module, name)
            break
    assert cls is not None, f"no *FlowEnv class on {backend_module!r}"
    inst = object.__new__(cls)
    inst.env_data_path = env_data_path
    inst.environment_name = environment_name
    return inst


@pytest.mark.parametrize(
    "input_arg",
    [None, "./bare.yaml", "abs_marker", "bare.yaml"],
    ids=["none", "relative_dot", "absolute", "bare"],
)
def test_config_resolver_consistency_across_backends(input_arg, tmp_path, monkeypatch):
    """The same ``_resolve_configuration_file`` body lives in maia / jax / jaxfluids.

    Drift guard: build identical on-disk layouts, run each resolver, and
    assert they all return the same path. Backends whose import-time
    dependencies are unavailable in this interpreter are skipped from
    the comparison set (but jaxfluids must always be present — module
    is skipped wholesale otherwise).
    """
    env_dir = tmp_path / "envdata"
    env_dir.mkdir()
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    # Lay down config files the resolvers will look for.
    (env_dir / "config.yaml").write_text("dummy: 1\n")  # for None case
    (cwd_dir / "bare.yaml").write_text("dummy: 2\n")  # for bare case (cwd wins)
    (cwd_dir / "rel_config.yaml").write_text("dummy: 3\n")
    abs_cfg = tmp_path / "absolute_cfg.yaml"
    abs_cfg.write_text("dummy: 4\n")

    # Translate the abstract input_arg into the concrete value handed to the resolver.
    if input_arg == "./bare.yaml":
        concrete = "./rel_config.yaml"
    elif input_arg == "abs_marker":
        concrete = str(abs_cfg)
    else:
        concrete = input_arg  # None or "bare.yaml"

    # Always exercise jaxfluids — the module-level importorskip guarantees it.
    jf_inst = _make_bare_for_backend(jf_env_core, str(env_dir))
    expected = jf_inst._resolve_configuration_file(concrete)

    backends_checked = ["jaxfluids"]

    # Try MAIA (needs mpi4py). Skip the comparison if unavailable, don't fail.
    maia_env_core = _try_import("hydrogym.maia.env_core")
    if maia_env_core is not None:
        maia_inst = _make_bare_for_backend(maia_env_core, str(env_dir))
        got = maia_inst._resolve_configuration_file(concrete)
        assert got == expected, f"maia resolver diverged: {got!r} != {expected!r}"
        backends_checked.append("maia")

    # Try JAX (needs jax + navix + chex + flax + gymnax).
    jax_env_core = _try_import("hydrogym.jax.env_core")
    if jax_env_core is not None:
        jax_inst = _make_bare_for_backend(jax_env_core, str(env_dir))
        got = jax_inst._resolve_configuration_file(concrete)
        assert got == expected, f"jax resolver diverged: {got!r} != {expected!r}"
        backends_checked.append("jax")

    # If neither MAIA nor JAX are importable we only verified jaxfluids
    # against itself — surface that as a skip so the test result is honest.
    if backends_checked == ["jaxfluids"]:
        pytest.skip("only jaxfluids resolver available; need maia or jax to compare against")
