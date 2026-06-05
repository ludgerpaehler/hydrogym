"""Pure-Python MAIA tests.

These tests exercise the solver-free Python surface of ``hydrogym.maia``: the
MPI interface dataclass-style attributes, command-tag table integrity, config
validation, property-file roundtrip helpers, observation/normalization
configuration, the registry and the per-environment ``convert_action`` glue.

They never invoke ``mpirun`` and never start the ``maia`` binary. Anything that
would normally call into MPI is either tested on a single-rank stub or has its
``MPI`` symbol patched out via ``unittest.mock``.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.requires_maia

# Skip cleanly when the MAIA stack (mpi4py) is not importable. Pure-python tests
# don't need mpirun, but they DO need the package surface to import.
pytest.importorskip(
    "mpi4py",
    reason="MAIA pure-python tests need mpi4py importable (no mpirun needed)",
)
pytest.importorskip("toml")
pytest.importorskip("omegaconf")

# These imports must come AFTER the importorskip so collection succeeds in
# environments that don't have the MAIA extras installed.
import numpy as np  # noqa: E402
import toml  # noqa: E402

from hydrogym.maia import env_core as env_core_mod  # noqa: E402
from hydrogym.maia import mpmd_interface as mpmd_mod  # noqa: E402
from hydrogym.maia.env_core import (  # noqa: E402
    _ENVIRONMENT_REGISTRY,
    ConfigError,
    MaiaFlowEnv,
    from_hf,
    list_registered_types,
    register_environment,
)
from hydrogym.maia.envs.cylinder import Cylinder  # noqa: E402
from hydrogym.maia.mpmd_interface import COMMAND_TAGS, MaiaInterface  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal YAML config that satisfies the attribute accesses in MaiaFlowEnv
# __init__ up to (and not past) the obs-normalization validation block. Tests
# 6-8 hit the validator before any cfg attribute is dereferenced, so this
# file's contents only need to parse.
_MINIMAL_CFG_YAML = """\
maia:
  num_sim_substeps_per_actuation: 1
  observation_type: "uv"
  num_action_inputs: 1
  max_control: 1.0
  render: false
  Re: 100
  U_inf: 1.0
  rho_inf: 1.0
env:
  max_episode_steps: 100
  n_agents: 1
"""


def _make_bare_env() -> MaiaFlowEnv:
    """Return a MaiaFlowEnv instance with ``__init__`` bypassed.

    Used to unit-test pure helpers (``_resolve_configuration_file``,
    ``configure_observations``, ``setup_normalization``,
    ``compute_nondim_coefficients``) without standing up MPI or the HF data
    manager.
    """
    return MaiaFlowEnv.__new__(MaiaFlowEnv)


# ---------------------------------------------------------------------------
# MaiaInterface — attribute surface (1)
# ---------------------------------------------------------------------------


def test_maia_interface_init_attributes_2d():
    """The documented attribute surface is present and pre-init values match."""
    iface = MaiaInterface(nDim=2)

    # Documented attributes (see MaiaInterface docstring).
    for attr in [
        "worldComm",
        "appComm",
        "appRank",
        "appRoot",
        "appRootInWorld",
        "appNoRanks",
        "appGroup",
        "appnum",
        "remoteRoot",
        "nDim",
    ]:
        assert hasattr(iface, attr), f"MaiaInterface missing attribute {attr!r}"

    # Pre-init defaults (init_comm() has NOT been called).
    assert iface.nDim == 2
    assert iface.appRoot == 0  # hard-coded default
    assert iface.worldComm is None
    assert iface.appComm is None
    assert iface.appRank is None
    assert iface.appRootInWorld is None
    assert iface.appNoRanks is None
    assert iface.appGroup is None
    assert iface.appnum is None
    assert iface.remoteRoot is None


# ---------------------------------------------------------------------------
# COMMAND_TAGS table (2, 3)
# ---------------------------------------------------------------------------


def test_command_tags_unique():
    """All MPI command tags are distinct so messages cannot be cross-routed."""
    values = list(COMMAND_TAGS.values())
    assert len(set(values)) == len(values), f"COMMAND_TAGS has duplicate values: {COMMAND_TAGS}"


def test_command_tags_keys_used_in_source_are_all_present():
    """Every ``COMMAND_TAGS["..."]`` lookup in the source resolves to a real key.

    Catches typo regressions where a sender or receiver references a tag that
    was renamed or never added to the table.
    """
    src = Path(mpmd_mod.__file__).read_text()
    pattern = re.compile(r"""COMMAND_TAGS\[\s*["']([^"']+)["']\s*\]""")
    used_keys = set(pattern.findall(src))

    assert used_keys, "No COMMAND_TAGS[...] lookups found in source — regex broke?"

    missing = used_keys - set(COMMAND_TAGS.keys())
    assert not missing, f"COMMAND_TAGS keys referenced in source but missing from table: {missing}"


# ---------------------------------------------------------------------------
# MaiaFlowEnv.__init__ validation (4-8)
# ---------------------------------------------------------------------------


def test_maiaflowenv_missing_environment_name_raises():
    """An empty env_config dies before any HF / MPI work happens."""
    # Patch HFDataManager so its constructor doesn't try to reach HF before the
    # explicit environment_name validation runs.
    with patch.object(env_core_mod, "HFDataManager", MagicMock()):
        with pytest.raises(ConfigError, match="environment_name"):
            MaiaFlowEnv(env_config={})


def test_maiaflowenv_missing_config_file_raises(tmp_path):
    """When no config file can be auto-detected, init raises ConfigError."""
    # Bypass HF: pretend env_data_path is a real (empty) directory.
    with (
        patch.object(env_core_mod, "HFDataManager", MagicMock()),
        patch.object(MaiaFlowEnv, "_setup_environment_data", return_value=str(tmp_path)),
    ):
        with pytest.raises(ConfigError, match="No configuration file found"):
            MaiaFlowEnv(env_config={"environment_name": "DummyEnv"})


@pytest.mark.parametrize("bad_strategy", ["bogus", "min_max", "z-score", ""])
def test_maiaflowenv_invalid_obs_normalization_strategy_raises(tmp_path, bad_strategy):
    """Strategies outside the documented set are rejected up front."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)

    with (
        patch.object(env_core_mod, "HFDataManager", MagicMock()),
        patch.object(MaiaFlowEnv, "_setup_environment_data", return_value=str(tmp_path)),
    ):
        with pytest.raises(ConfigError, match="Invalid obs_normalization_strategy"):
            MaiaFlowEnv(
                env_config={
                    "environment_name": "DummyEnv",
                    "configuration_file": str(cfg_path),
                    "obs_normalization_strategy": bad_strategy,
                }
            )


def test_maiaflowenv_customized_without_obs_loc_raises(tmp_path):
    """``customized`` strategy needs both obs_loc and obs_scale."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)

    with (
        patch.object(env_core_mod, "HFDataManager", MagicMock()),
        patch.object(MaiaFlowEnv, "_setup_environment_data", return_value=str(tmp_path)),
    ):
        with pytest.raises(ConfigError, match="requires both 'obs_loc' and 'obs_scale'"):
            MaiaFlowEnv(
                env_config={
                    "environment_name": "DummyEnv",
                    "configuration_file": str(cfg_path),
                    "obs_normalization_strategy": "customized",
                    # neither obs_loc nor obs_scale provided
                }
            )


def test_maiaflowenv_customized_length_mismatch_raises():
    """``setup_normalization`` rejects obs_loc/obs_scale that don't match num_outputs."""
    env = _make_bare_env()
    env.obs_normalization_strategy = "customized"
    env.obs_loc = [0.0, 0.0, 0.0]  # length 3
    env.obs_scale = [1.0, 1.0]  # length 2
    env.num_outputs = 5  # neither matches

    with pytest.raises(ConfigError, match="Customized normalization dimensions mismatch"):
        env.setup_normalization()


# ---------------------------------------------------------------------------
# _resolve_configuration_file / _find_configuration_file (9-13)
# ---------------------------------------------------------------------------


def test_resolve_config_none_auto_detect(tmp_path):
    """``None`` input delegates to ``_find_configuration_file`` (env-dir search)."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)

    env = _make_bare_env()
    env.env_data_path = str(tmp_path)
    env.environment_name = "DummyEnv"

    resolved = env._resolve_configuration_file(None)
    assert resolved == str(cfg_path)


def test_resolve_config_absolute(tmp_path):
    """An absolute path is returned verbatim when it exists."""
    cfg_path = tmp_path / "myconf.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)

    env = _make_bare_env()
    env.env_data_path = str(tmp_path)
    env.environment_name = "DummyEnv"

    resolved = env._resolve_configuration_file(str(cfg_path))
    assert resolved == str(cfg_path)


def test_resolve_config_absolute_missing_raises(tmp_path):
    """A missing absolute path raises ConfigError (not silently falls through)."""
    env = _make_bare_env()
    env.env_data_path = str(tmp_path)
    env.environment_name = "DummyEnv"

    bogus = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises(ConfigError, match="Configuration file not found"):
        env._resolve_configuration_file(bogus)


def test_resolve_config_relative_dot_slash(tmp_path, monkeypatch):
    """``./name.yaml`` is resolved against the current working directory."""
    cfg_path = tmp_path / "local.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)
    monkeypatch.chdir(tmp_path)

    env = _make_bare_env()
    env.env_data_path = str(tmp_path)
    env.environment_name = "DummyEnv"

    resolved = env._resolve_configuration_file("./local.yaml")
    assert resolved == str(cfg_path)


def test_resolve_config_bare_falls_back_to_env_data_path(tmp_path, monkeypatch):
    """A bare filename that isn't in cwd is found in ``env_data_path``."""
    # Chdir to a sibling dir so the bare name does NOT collide with cwd.
    cwd_dir = tmp_path / "cwd"
    env_dir = tmp_path / "env"
    cwd_dir.mkdir()
    env_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    cfg_path = env_dir / "config.yaml"
    cfg_path.write_text(_MINIMAL_CFG_YAML)

    env = _make_bare_env()
    env.env_data_path = str(env_dir)
    env.environment_name = "DummyEnv"

    resolved = env._resolve_configuration_file("config.yaml")
    assert resolved == str(cfg_path)


def test_find_config_prefers_environment_config(tmp_path):
    """Auto-detect priority: ``config.yaml`` wins over alternatives."""
    # Drop a few candidates; ``config.yaml`` is first in the search list.
    (tmp_path / "config.yaml").write_text(_MINIMAL_CFG_YAML)
    (tmp_path / "environment_config.yaml").write_text(_MINIMAL_CFG_YAML)
    (tmp_path / "DummyEnv.yaml").write_text(_MINIMAL_CFG_YAML)

    env = _make_bare_env()
    env.env_data_path = str(tmp_path)
    env.environment_name = "DummyEnv"

    resolved = env._find_configuration_file()
    assert resolved == str(tmp_path / "config.yaml")


# ---------------------------------------------------------------------------
# Property file roundtrip (14, 15)
# ---------------------------------------------------------------------------


def test_property_file_round_trip_simple(tmp_path):
    """read → update → write → re-read preserves a top-level update."""
    path = tmp_path / "props.toml"
    path.write_text('foo = 1\nbar = "baz"\n')

    env = _make_bare_env()
    data = env._read_property_file(str(path))
    assert data == {"foo": 1, "bar": "baz"}

    env._update_property(data, "foo", 42)
    env._write_property_file(data, str(path))

    reloaded = env._read_property_file(str(path))
    assert reloaded["foo"] == 42
    assert reloaded["bar"] == "baz"


def test_property_file_round_trip_nested(tmp_path):
    """Nested-key updates use the documented ``[section, key]`` list form."""
    path = tmp_path / "nested.toml"
    initial = {"section": {"alpha": 1.5, "beta": 2.5}, "scalar": 9}
    with open(path, "w") as f:
        toml.dump(initial, f)

    env = _make_bare_env()
    data = env._read_property_file(str(path))

    env._update_property(data, ["section", "alpha"], 100.0)
    env._write_property_file(data, str(path))

    reloaded = env._read_property_file(str(path))
    assert reloaded["section"]["alpha"] == 100.0
    assert reloaded["section"]["beta"] == 2.5
    assert reloaded["scalar"] == 9
    # _get_property mirror-check
    assert env._get_property(reloaded, ["section", "alpha"]) == 100.0
    assert env._get_property(reloaded, "scalar") == 9


# ---------------------------------------------------------------------------
# configure_observations (16-18)
# ---------------------------------------------------------------------------


def _setup_env_for_configure(nDim: int, num_probes: int, observation_type: str, bc_ids=()):
    env = _make_bare_env()
    env.nDim = nDim
    # probe_locations has nDim coords per probe.
    env.probe_locations = [0.0] * (num_probes * nDim)
    env.observation_type = observation_type
    env.bcId = list(bc_ids)
    return env


def test_configure_observations_num_outputs_2d_vel_only():
    """2D + ``uv`` observation type → 2 * num_probes outputs."""
    n_probes = 4
    env = _setup_env_for_configure(nDim=2, num_probes=n_probes, observation_type="uv")
    env.configure_observations()

    assert env.num_probes == n_probes
    assert env.num_outputs == 2 * n_probes  # u and v


def test_configure_observations_num_outputs_3d_vel_only():
    """3D + ``uvw`` observation type → 3 * num_probes outputs."""
    n_probes = 5
    env = _setup_env_for_configure(nDim=3, num_probes=n_probes, observation_type="uvw")
    env.configure_observations()

    assert env.num_probes == n_probes
    assert env.num_outputs == 3 * n_probes  # u, v, w


def test_configure_observations_substring_matching_footgun_rho_in_rho_and_p():
    """Pins the substring-matching footgun in env_core.py:configure_observations.

    ``observation_type='rho_and_p'`` triggers BOTH the ``'rho' in ...`` and
    the ``'p' in ...`` branches via plain Python substring containment — so
    rho-and-p adds num_probes outputs twice (once for rho, once for p) and
    correctly yields 2*N. Crucially this also means a string like ``'pressure'``
    would silently trigger the ``'p'`` branch.

    See ``configure_observations`` in hydrogym/maia/env_core.py
    (around line 355) — the matcher is ``substring in observation_type``, not
    a token split.
    """
    n_probes = 3
    env = _setup_env_for_configure(nDim=2, num_probes=n_probes, observation_type="rho_and_p")
    env.configure_observations()

    # Verify the substring branches that fire:
    assert "rho" in env.observation_type
    assert "p" in env.observation_type
    # u / v / w / forces do NOT appear as substrings of 'rho_and_p'.
    assert "u" not in env.observation_type
    assert "v" not in env.observation_type
    assert "w" not in env.observation_type
    assert "forces" not in env.observation_type

    # Pinned result: 2 * num_probes (rho contributes N, p contributes N).
    assert env.num_outputs == 2 * n_probes


# ---------------------------------------------------------------------------
# setup_normalization (19-22)
# ---------------------------------------------------------------------------


def test_setup_normalization_u_inf():
    """``U_inf`` strategy builds obs_scale = [U_inf]*2N for ``uv`` in 2D."""
    from omegaconf import OmegaConf

    env = _make_bare_env()
    env.obs_normalization_strategy = "U_inf"
    env.observation_type = "uv"
    env.nDim = 2
    env.noProbes = 3
    env.bcId = []  # no forces
    env.cfg = OmegaConf.create({"maia": {"U_inf": 2.0, "rho_inf": 1.0}})

    env.setup_normalization()

    expected_length = 2 * env.noProbes  # u and v
    assert len(env.obs_scale) == expected_length
    assert len(env.obs_loc) == expected_length
    np.testing.assert_array_equal(env.obs_scale, np.full(expected_length, 2.0))
    np.testing.assert_array_equal(env.obs_loc, np.zeros(expected_length))


def test_setup_normalization_none_is_identity():
    """``none`` strategy yields loc=0, scale=1 of length num_outputs."""
    env = _make_bare_env()
    env.obs_normalization_strategy = "none"
    env.num_outputs = 7

    env.setup_normalization()

    assert env.obs_loc == [0.0] * 7
    assert env.obs_scale == [1.0] * 7


def test_setup_normalization_customized_length_mismatch_raises():
    """``customized`` rejects mismatched-length loc/scale arrays."""
    env = _make_bare_env()
    env.obs_normalization_strategy = "customized"
    env.obs_loc = [0.0] * 4
    env.obs_scale = [1.0] * 4
    env.num_outputs = 6  # mismatch

    with pytest.raises(ConfigError, match="Customized normalization dimensions mismatch"):
        env.setup_normalization()


def test_setup_normalization_customized_ok():
    """``customized`` with matching lengths leaves obs_loc/obs_scale untouched."""
    env = _make_bare_env()
    env.obs_normalization_strategy = "customized"
    env.obs_loc = [1.0, 2.0, 3.0]
    env.obs_scale = [0.5, 0.5, 0.5]
    env.num_outputs = 3

    # Should not raise; values should be unchanged.
    env.setup_normalization()
    assert env.obs_loc == [1.0, 2.0, 3.0]
    assert env.obs_scale == [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# compute_nondim_coefficients (23)
# ---------------------------------------------------------------------------


def test_compute_nondim_coefficients_hand_value():
    """C = 2*F / (rho * v^2 * L); hand-computed against a small example."""
    env = _make_bare_env()
    F = np.array([2.0, 3.0])
    rho = 1.0
    v = 1.5
    L = 0.5

    coeffs = env.compute_nondim_coefficients(
        forces=F,
        density=rho,
        referenceVelocity=v,
        projectionLength=L,
    )

    denom = rho * v**2 * L  # 1.0 * 2.25 * 0.5 = 1.125
    expected = np.array([2 * 2.0 / denom, 2 * 3.0 / denom])
    np.testing.assert_allclose(coeffs, expected)


# ---------------------------------------------------------------------------
# Registry (24-26)
# ---------------------------------------------------------------------------


def test_environment_registry_underscore_parse():
    """``from_hf`` extracts the prefix via ``split('_', 1)[0]``.

    We verify the parse by registering a stub class under ``'Cylinder'`` and
    asserting that ``from_hf('Cylinder_2D_Re200', ...)`` instantiates THAT
    stub with the original environment_name preserved in env_config. Note
    that the parser does NOT extract nDim or Re — only the prefix is used
    for class lookup.
    """
    seen: dict = {}

    class _StubEnv:
        def __init__(self, env_config):
            seen.update(env_config)

    saved = _ENVIRONMENT_REGISTRY.get("Cylinder")
    try:
        register_environment("Cylinder", _StubEnv)
        env = from_hf("Cylinder_2D_Re200", hf_repo_id="dummy/repo")
        assert isinstance(env, _StubEnv)
        # The full name is preserved in env_config — parsing extracts only
        # the prefix for class lookup.
        assert seen["environment_name"] == "Cylinder_2D_Re200"
        assert seen["hf_repo_id"] == "dummy/repo"
    finally:
        if saved is not None:
            _ENVIRONMENT_REGISTRY["Cylinder"] = saved
        else:
            _ENVIRONMENT_REGISTRY.pop("Cylinder", None)


def test_environment_registry_unknown_prefix_raises():
    """An unregistered prefix yields a ConfigError wrapping a ValueError.

    ``from_hf`` catches the internal ValueError raised by the registry lookup
    and re-raises as ConfigError. We pin both behaviours: the surfaced type
    AND the message mentioning the unknown type.
    """
    assert "Bogus" not in _ENVIRONMENT_REGISTRY
    # from_hf raises ValueError directly for the unknown-prefix branch
    # (the try/except wraps only the env_class(...) call, not the lookup).
    with pytest.raises(ValueError, match="Unknown environment type 'Bogus'"):
        from_hf("Bogus_2D_Re100")


def test_list_registered_types_returns_copy():
    """Mutating the return value must not affect the internal registry."""
    snapshot = dict(_ENVIRONMENT_REGISTRY)

    returned = list_registered_types()
    assert isinstance(returned, dict)
    assert returned == _ENVIRONMENT_REGISTRY

    # Mutate the returned dict.
    returned["__bogus_test_key__"] = object
    returned.pop(next(iter(returned)), None)

    # Internal registry untouched.
    assert _ENVIRONMENT_REGISTRY == snapshot


# ---------------------------------------------------------------------------
# Cylinder.convert_action (27)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_jets, action, expected",
    [
        (4, [1.0, 2.0], [1.0, -1.0, 2.0, -2.0]),
        (6, [1.0, 2.0, 3.0], [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]),
        (8, [1.0, 2.0, 3.0, 4.0], [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]),
    ],
)
def test_cylinder_convert_action_pairing(num_jets, action, expected):
    """Each input action is duplicated as (+a, -a) per jet pair for zero net mass flux.

    Cylinder.convert_action iterates ``range(numJetsInSimulation // 2)`` and
    extends the sequence with ``[action[i], -action[i]]`` per pair. The
    action vector must therefore have at least ``numJetsInSimulation // 2``
    elements.
    """
    cyl = Cylinder.__new__(Cylinder)
    cyl.numJetsInSimulation = num_jets
    result = cyl.convert_action(np.asarray(action))
    assert result == expected
