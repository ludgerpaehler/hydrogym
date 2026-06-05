"""
Pure-Python unit tests for the NEK backend.

Covers the solver-free Python surface area of ``hydrogym.nek``:
the ``tag_dict`` table, ``mpi_split`` validation, ``Config`` dataclasses,
config loading helpers, backwards-compat aliases, runtime override mapping,
normalization helpers, ``RingBuffer``, ``NekParallelEnv`` array<->dict
conversion, AFC controllers, ``make_afc_controller`` factory branches,
and ``integrate``'s save-path fallback cascade.

These tests deliberately bypass MPI: every test that touches a class which
would otherwise call ``mpi_split`` constructs the object by hand
(``object.__new__`` + attribute injection) or mocks the MPI primitives.
"""

import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Module-level marker so the suite can be selectively enabled in CI.
pytestmark = pytest.mark.requires_nek

# mpi4py must be importable to even read tag_dict (it embeds MPI.* dtypes).
pytest.importorskip("mpi4py", reason="NEK pure-python tests need mpi4py importable")
pytest.importorskip("omegaconf", reason="NEK config tests need omegaconf importable")
pytest.importorskip("gymnasium", reason="NEK env tests need gymnasium importable")


# --- Imports of the modules under test (after importorskip guards) -----------
#
# NOTE: ``hydrogym/nek/integrate.py`` does
#     ``from hydrogym.nek.nek_lib.nek_utils import show_end, show_title``
# but those symbols don't exist in nek_utils. That makes
# ``import hydrogym.nek`` fail in pure-Python environments even though every
# other submodule is importable in isolation. We work around the latent bug
# by pre-installing a stub ``hydrogym.nek.nek_lib.nek_utils`` module in
# ``sys.modules`` that carries the missing names alongside the symbols the
# rest of the package needs (``remove_sch``). When the real ``nek_utils``
# would otherwise be loaded, the cached stub wins and downstream imports
# (``integrate.py`` -> ``show_end, show_title``; ``env.py`` -> ``remove_sch``)
# all succeed.
# (See "Latent bugs" in the test report - this is bug #1.)
import importlib.machinery as _importlib_machinery  # noqa: E402

if "hydrogym.nek.nek_lib.nek_utils" not in sys.modules:
    _stub = types.ModuleType("hydrogym.nek.nek_lib.nek_utils")
    _stub.__spec__ = _importlib_machinery.ModuleSpec("hydrogym.nek.nek_lib.nek_utils", loader=None)
    _stub.show_end = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _stub.show_title = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _stub.remove_sch = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["hydrogym.nek.nek_lib.nek_utils"] = _stub

from omegaconf import OmegaConf  # noqa: E402

import hydrogym.nek as nek_pkg  # noqa: E402
from hydrogym.nek import AFC as afc_module  # noqa: E402, F401
from hydrogym.nek import configs as configs_module  # noqa: E402, F401
from hydrogym.nek import env as env_module  # noqa: E402

# Note: ``hydrogym.nek.integrate`` (the *function*, via __init__'s
# ``from .integrate import integrate``) shadows the *submodule* of the same
# name once the package has finished initializing. Fetch the submodule
# explicitly via ``sys.modules`` so ``integrate_module.integrate(...)`` works.
integrate_module = sys.modules["hydrogym.nek.integrate"]
from hydrogym.nek import parallel_env as parallel_module  # noqa: E402, F401
from hydrogym.nek.AFC import (  # noqa: E402
    BLCtrl,
    OppoCtrl,
    SinWave,
    ZeroCtrl,
    make_afc_controller,
)
from hydrogym.nek.configs import Config, parse_cli  # noqa: E402
from hydrogym.nek.env import (  # noqa: E402
    NekEnv,
    RingBuffer,
    mpi_split,
    tag_dict,
)
from hydrogym.nek.parallel_env import NekParallelEnv  # noqa: E402

# =========================================================================
# tag_dict integrity
# =========================================================================


REQUIRED_TAG_KEYS = ["NID", "NUMCTRL", "GLLID", "STATE", "REWRD", "ACTION", "COMMAND"]


def test_tag_dict_required_keys_present():
    """All MPI tags the spec mentions are present in tag_dict."""
    for key in REQUIRED_TAG_KEYS:
        assert key in tag_dict, f"tag_dict missing required key: {key}"


def test_tag_dict_no_cross_category_collisions():
    """Every tag integer is unique - reusing tags between sends causes deadlocks."""
    all_tag_values = [entry["tag"] for entry in tag_dict.values()]
    assert len(set(all_tag_values)) == len(all_tag_values), (
        f"Tag collisions in tag_dict: values={sorted(all_tag_values)}"
    )


def test_tag_dict_entry_shape():
    """Every entry has the required sub-keys: tag, mpi_dtype, py_dtype, cate."""
    required_sub_keys = {"tag", "mpi_dtype", "py_dtype", "cate"}
    for key, entry in tag_dict.items():
        assert isinstance(entry, dict), f"tag_dict[{key!r}] should be a dict"
        missing = required_sub_keys - set(entry.keys())
        assert not missing, f"tag_dict[{key!r}] missing sub-keys: {missing}"
        # The category should be one of a small set of known string labels.
        assert isinstance(entry["cate"], str)
        assert isinstance(entry["tag"], int)


# =========================================================================
# Magic-number bug pin (xfail until fixed)
# =========================================================================


def _read_start_simulation_source() -> str:
    """Return the source slice for NekEnv._start_simulation."""
    env_path = Path(env_module.__file__)
    src = env_path.read_text()
    # Grab a window of lines around the _start_simulation definition.
    match = re.search(r"def _start_simulation\(self\):.*?(?=\n    def |\nclass )", src, re.DOTALL)
    assert match, "could not locate _start_simulation in env.py"
    return match.group(0)


@pytest.mark.xfail(
    reason="magic number 22 instead of tag_dict['COMMAND']['tag']; nek/env.py: _start_simulation",
    strict=False,
)
def test_start_simulation_uses_literal_tag_22_xfail():
    """
    Survey-identified bug: ``_start_simulation`` uses literal ``tag=22`` instead
    of looking up ``tag_dict['COMMAND']['tag']``.

    The xfail asserts that the literal has been removed. The body also
    cross-checks that the magic number is at least consistent with the table.
    """
    body = _read_start_simulation_source()
    # Bug-present condition: a literal tag=22 in the Send call.
    has_magic = bool(re.search(r"tag\s*=\s*22\b", body))
    # Consistency check (always true today): 22 is the COMMAND tag.
    assert tag_dict["COMMAND"]["tag"] == 22
    # When the bug is fixed, has_magic will become False and this xfail
    # will start passing naturally (turning into XPASS).
    assert not has_magic, "literal tag=22 still present in _start_simulation"


# =========================================================================
# mpi_split validation
# =========================================================================


def test_mpi_split_invalid_raises():
    """mpi_split must reject world sizes < 2 (no Nek workers to talk to)."""
    fake_comm = MagicMock()
    fake_comm.Get_rank.return_value = 0
    fake_comm.Get_size.return_value = 1
    with pytest.raises(RuntimeError, match="MPI world size must be >= 2"):
        mpi_split(fake_comm, nproc=None)


def test_mpi_split_nproc_mismatch_raises():
    """If nproc is given, mpi_split must reject mismatched world sizes."""
    fake_comm = MagicMock()
    fake_comm.Get_rank.return_value = 0
    fake_comm.Get_size.return_value = 4  # 1 ctl + 3 wrk
    with pytest.raises(RuntimeError, match="MPI world size mismatch"):
        mpi_split(fake_comm, nproc=10)  # expects 11


# =========================================================================
# Config dataclasses
# =========================================================================


def test_config_defaults():
    """Pin the documented defaults of Config()."""
    cfg = Config()
    # Runner
    assert cfg.runner.RL_algorithm == "DDPG"
    assert cfg.runner.nb_interactions == 3000
    assert cfg.runner.nb_episodes == 100
    assert cfg.runner.normalize_input == "utau"
    assert cfg.runner.rescale_actions is True
    assert cfg.runner.rew_mode == "Homo"
    assert cfg.runner.npl_state == 2
    # Simulation
    assert cfg.simulation.CASENAME == "phill"
    assert cfg.simulation.lx1 == 6
    assert cfg.simulation.nproc == 14
    assert cfg.simulation.TOTCTRL == 10
    assert cfg.simulation.ndrl == 3
    assert cfg.simulation.exeName == "nek5000"


def test_omegaconf_round_trip():
    """Structured Config -> OmegaConf -> dict -> OmegaConf preserves data."""
    cfg = Config()
    oc = OmegaConf.structured(cfg)
    as_container = OmegaConf.to_container(oc, resolve=True)
    # Resurrect from the container and compare a representative field set.
    re_oc = OmegaConf.create(as_container)
    assert re_oc.runner.RL_algorithm == "DDPG"
    assert re_oc.simulation.CASENAME == "phill"
    assert re_oc.simulation.lx1 == 6
    assert re_oc.runner.nb_interactions == 3000


# =========================================================================
# load_nek_config / parse_cli
# =========================================================================


def test_load_nek_config_from_tmp_yaml(tmp_path):
    """load_nek_config should layer a YAML on top of structured defaults."""
    yaml_path = tmp_path / "conf.yaml"
    yaml_path.write_text("simulation:\n  CASENAME: custom_case\n  lx1: 8\nrunner:\n  RL_algorithm: PPO\n")
    conf = nek_pkg.load_nek_config(str(yaml_path))
    assert conf.simulation.CASENAME == "custom_case"
    assert conf.simulation.lx1 == 8
    assert conf.runner.RL_algorithm == "PPO"
    # untouched defaults survive
    assert conf.simulation.nproc == 14


def test_load_nek_config_with_dotlist_overrides(tmp_path):
    yaml_path = tmp_path / "conf.yaml"
    yaml_path.write_text("simulation:\n  CASENAME: from_yaml\n")
    conf = nek_pkg.load_nek_config(
        str(yaml_path),
        overrides=["simulation.lx1=9", "runner.RL_algorithm=SAC"],
    )
    assert conf.simulation.CASENAME == "from_yaml"
    assert conf.simulation.lx1 == 9
    assert conf.runner.RL_algorithm == "SAC"


def test_parse_cli_with_dotlist_overrides(tmp_path, capsys):
    """parse_cli should accept a YAML file and dotlist overrides without raising."""
    yaml_path = tmp_path / "conf.yml"
    yaml_path.write_text("simulation:\n  CASENAME: parsed_case\n")
    # parse_cli currently only prints the merged YAML, so capture stdout
    # and verify the overridden values appear in the dump.
    parse_cli([str(yaml_path), "simulation.lx1=12", "runner.RL_algorithm=TD3"])
    captured = capsys.readouterr().out
    assert "parsed_case" in captured
    assert "lx1: 12" in captured
    assert "TD3" in captured


def test_parse_cli_rejects_unrecognized_arg():
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_cli(["not_a_yaml_or_override"])


# =========================================================================
# Backwards-compat aliases
# =========================================================================


def test_legacy_aliases():
    """NekMARLGymWrapper and parallel_env are kept as legacy aliases."""
    assert nek_pkg.NekMARLGymWrapper is NekEnv
    assert nek_pkg.parallel_env is NekParallelEnv


# =========================================================================
# _resolve_configuration_file / _find_configuration_file mirror
# =========================================================================


def _make_env_with_data_path(tmp_path):
    """Construct a bare NekEnv shell with just the attrs _resolve_configuration_file needs."""
    env = object.__new__(NekEnv)
    env.env_data_path = str(tmp_path)
    env.environment_name = "TestEnv"
    return env


def test_resolve_configuration_file_none_auto_detects_environment_config(tmp_path):
    """When config_file_input is None and environment_config.yaml exists, return it."""
    env = _make_env_with_data_path(tmp_path)
    cfg_file = tmp_path / "environment_config.yaml"
    cfg_file.write_text("foo: bar\n")
    result = env._resolve_configuration_file(None)
    assert result == str(cfg_file)


def test_resolve_configuration_file_none_falls_back_to_config_yaml(tmp_path):
    """Without environment_config.yaml, _resolve_configuration_file accepts config.yaml."""
    env = _make_env_with_data_path(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("foo: bar\n")
    result = env._resolve_configuration_file(None)
    assert result == str(cfg_file)


def test_resolve_configuration_file_none_returns_none_when_missing(tmp_path):
    """If neither candidate exists, _resolve_configuration_file returns None."""
    env = _make_env_with_data_path(tmp_path)
    assert env._resolve_configuration_file(None) is None


def test_resolve_configuration_file_absolute_path_present(tmp_path):
    """Absolute path that exists is returned verbatim."""
    env = _make_env_with_data_path(tmp_path)
    abs_cfg = tmp_path / "elsewhere.yaml"
    abs_cfg.write_text("x: 1\n")
    result = env._resolve_configuration_file(str(abs_cfg))
    assert result == str(abs_cfg)


def test_resolve_configuration_file_relative_name_in_env_dir(tmp_path):
    """Bare filename is resolved relative to env_data_path."""
    env = _make_env_with_data_path(tmp_path)
    rel = tmp_path / "alt.yaml"
    rel.write_text("x: 1\n")
    result = env._resolve_configuration_file("alt.yaml")
    assert result == str(rel)


def test_resolve_configuration_file_absolute_missing_returns_none(tmp_path):
    env = _make_env_with_data_path(tmp_path)
    bogus = tmp_path / "does_not_exist.yaml"
    assert env._resolve_configuration_file(str(bogus)) is None


# =========================================================================
# Runtime overrides
# =========================================================================


RUNTIME_OVERRIDE_CASES = [
    ("normalize_input", "normalization", "normalize_input", "std"),
    ("nb_interactions", "episode", "max_interactions", 1234),
    ("random_init", "initial_conditions", "random_init", -1),
    ("rescale_actions", "rl_interface", "rescale_actions", False),
    ("rew_mode", "episode", "reward_mode", "MovingAverage"),
]


@pytest.mark.parametrize(("key", "section", "param", "value"), RUNTIME_OVERRIDE_CASES)
def test_apply_runtime_overrides_table_driven(key, section, param, value):
    """
    Every key in the override_map should write to the corresponding
    (section, param) of the loaded config.
    """
    # Build a config skeleton that has the target sections/params.
    conf = OmegaConf.create(
        {
            "normalization": {"normalize_input": "None"},
            "episode": {"max_interactions": 0, "reward_mode": "Homo"},
            "initial_conditions": {"random_init": 1},
            "rl_interface": {"rescale_actions": True},
        }
    )
    env = object.__new__(NekEnv)
    env.conf = conf
    env._apply_runtime_overrides({key: value})
    assert getattr(getattr(conf, section), param) == value


def test_apply_runtime_overrides_ignores_unknown_keys():
    """Keys not in override_map should be silently ignored (no crash)."""
    conf = OmegaConf.create({"normalization": {"normalize_input": "None"}})
    env = object.__new__(NekEnv)
    env.conf = conf
    env._apply_runtime_overrides({"totally_unknown_key": 42})
    # Untouched.
    assert conf.normalization.normalize_input == "None"


# =========================================================================
# Normalization helpers
# =========================================================================


def _make_env_for_normalize():
    env = object.__new__(NekEnv)
    return env


def test_normalize_state_utau():
    env = _make_env_for_normalize()
    env.normalize_input = "utau"
    env.utau = 2.0
    state = np.array([2.0, 4.0, 6.0])
    out = env._normalize_state(state.copy())
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))


def test_normalize_state_std():
    env = _make_env_for_normalize()
    env.normalize_input = "std"
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = env._normalize_state(state.copy())
    np.testing.assert_allclose(out, state / np.std(state))


def test_normalize_state_minmax():
    env = _make_env_for_normalize()
    env.normalize_input = "minmax"
    state = np.array([10.0, 20.0, 30.0])
    out = env._normalize_state(state.copy())
    # Expect [-1, 0, 1] after 2*(x-min)/(max-min) - 1
    np.testing.assert_allclose(out, np.array([-1.0, 0.0, 1.0]))


def test_normalize_state_none_passthrough():
    env = _make_env_for_normalize()
    env.normalize_input = "None"
    state = np.array([1.0, 2.0, 3.0])
    out = env._normalize_state(state.copy())
    np.testing.assert_allclose(out, state)


def test_normalize_reward_branches():
    env = _make_env_for_normalize()
    env.baseline_dudy = 2.0
    # 1 - (reward / baseline)
    assert env._normalize_reward(2.0) == pytest.approx(0.0)
    assert env._normalize_reward(0.0) == pytest.approx(1.0)
    assert env._normalize_reward(1.0) == pytest.approx(0.5)
    assert env._normalize_reward(4.0) == pytest.approx(-1.0)


# =========================================================================
# _apply_znmf
# =========================================================================


def test_apply_znmf_branch_naive_minus_one():
    env = object.__new__(NekEnv)
    env.znmf_avg = -1
    action = np.array([1.0, 3.0, 5.0])
    out = env._apply_znmf(action)
    np.testing.assert_allclose(out, action - np.mean(action))


def test_apply_znmf_branch_gll_weighted_minus_two():
    """The weighted branch should subtract a GLL-weighted average of the action."""
    env = object.__new__(NekEnv)
    env.znmf_avg = -2
    env.n_actuators = 3
    # gll_weight is indexed by ix-1, iz-1 -> pick simple ones to keep arithmetic checkable.
    env.gll_weight = np.array([0.5, 1.0, 2.0])
    env.actuator_info = {
        "ix": np.array([1, 2, 3], dtype=np.int32),
        "iz": np.array([1, 2, 3], dtype=np.int32),
    }
    action = np.array([1.0, 2.0, 3.0])
    # Expected weighted mean:
    #   wx*wz per actuator: 0.25, 1.0, 4.0  -> wxz = 5.25
    #   numerator = 1*0.25 + 2*1.0 + 3*4.0 = 14.25
    #   mean_action = 14.25 / 5.25
    expected_mean = (1 * 0.25 + 2 * 1.0 + 3 * 4.0) / (0.25 + 1.0 + 4.0)
    out = env._apply_znmf(action)
    np.testing.assert_allclose(out, action - expected_mean)


@pytest.mark.parametrize("znmf_avg", [0, 1])
def test_apply_znmf_branch_done_by_nek_passthrough(znmf_avg):
    env = object.__new__(NekEnv)
    env.znmf_avg = znmf_avg
    action = np.array([1.0, 2.0, 3.0])
    out = env._apply_znmf(action)
    # Untouched.
    np.testing.assert_allclose(out, action)


def test_apply_znmf_else_raises_not_implemented():
    env = object.__new__(NekEnv)
    env.znmf_avg = 99
    with pytest.raises(NotImplementedError):
        env._apply_znmf(np.array([1.0, 2.0]))


# =========================================================================
# _name_agent width and drift
# =========================================================================


def test_name_agent_width_8_digits_in_nekenv():
    name = NekEnv._name_agent(nid=5, gllid=2, iface=3, ix=4, iy=6, iz=7)
    # 8-digit zero-padded numbers should appear in the string.
    assert "np00000005" in name
    assert "gid00000002" in name
    assert "ix00000004" in name
    assert "iy00000006" in name
    assert "iz00000007" in name
    # Sanity: iface is rendered as-is (no padding).
    assert "iface3" in name


def test_name_agent_cross_equality_with_parallel():
    """NekEnv._name_agent and NekParallelEnv._name_agent should produce identical strings."""
    args = dict(nid=1, gllid=2, iface=3, ix=4, iy=5, iz=6)
    a = NekEnv._name_agent(**args)
    b = NekParallelEnv._name_agent(**args)
    assert a == b


@pytest.mark.xfail(
    reason="known width drift; SinWave uses 5 digits, NekEnv uses 8",
    strict=False,
)
def test_sinwave_name_agent_width_matches_nekenv():
    """
    Survey-identified bug: ``SinWave.nameAgent`` uses 5-digit widths whereas
    ``NekEnv._name_agent`` uses 8-digit widths. When SinWave is widened to 8
    digits this test will start passing.
    """
    args = dict(nid=1, gllid=2, iface=3, ix=4, iy=5, iz=6)
    sin_name = SinWave.nameAgent(**args)
    nek_name = NekEnv._name_agent(**args)
    assert sin_name == nek_name


# =========================================================================
# RingBuffer
# =========================================================================


def test_ring_buffer_extend_and_average():
    rb = RingBuffer(length=3, dim=(2,))
    rb.extend(np.array([1.0, 2.0]))
    rb.extend(np.array([3.0, 4.0]))
    rb.extend(np.array([5.0, 6.0]))
    np.testing.assert_allclose(rb.average(), np.array([3.0, 4.0]))


def test_ring_buffer_wrap_around_at_length():
    rb = RingBuffer(length=2, dim=(1,))
    rb.extend(np.array([1.0]))
    rb.extend(np.array([2.0]))
    # 3rd extend wraps around: index resets via x_index = self.index % length
    rb.extend(np.array([3.0]))
    # Buffer should now contain [3.0, 2.0] (slot 0 was overwritten)
    np.testing.assert_allclose(rb.data[:, 0], np.array([3.0, 2.0]))


def test_ring_buffer_extend_shape_mismatch():
    rb = RingBuffer(length=2, dim=(2,))
    with pytest.raises(AssertionError):
        rb.extend(np.array([1.0]))


# =========================================================================
# NekParallelEnv array<->dict
# =========================================================================


def _make_parallel_env(n_actuators=3, obs_per_actuator=2):
    """Construct a NekParallelEnv shell without calling __init__."""
    base = object.__new__(NekEnv)
    base.n_actuators = n_actuators
    base.obs_per_actuator = obs_per_actuator
    # Provide deterministic actuator_info so agent names are reproducible.
    base.actuator_info = {
        "NID": np.arange(n_actuators, dtype=np.int32),
        "GLLID": np.arange(n_actuators, dtype=np.int32),
        "FACEID": np.zeros(n_actuators, dtype=np.int32),
        "ix": np.arange(n_actuators, dtype=np.int32),
        "iy": np.arange(n_actuators, dtype=np.int32),
        "iz": np.arange(n_actuators, dtype=np.int32),
    }
    # action_space.low/high are read by NekParallelEnv.action_space; not needed for
    # the conversion tests below.
    penv = object.__new__(NekParallelEnv)
    penv.env = base
    penv.possible_agents = [
        NekParallelEnv._name_agent(
            nid=base.actuator_info["NID"][i],
            gllid=base.actuator_info["GLLID"][i],
            iface=base.actuator_info["FACEID"][i],
            ix=base.actuator_info["ix"][i],
            iy=base.actuator_info["iy"][i],
            iz=base.actuator_info["iz"][i],
        )
        for i in range(n_actuators)
    ]
    penv.agents = penv.possible_agents[:]
    penv.obs_per_agent = obs_per_actuator
    penv.act_per_agent = 1
    return penv


def test_parallel_dict_to_array():
    """Dict of actions -> flat array preserves order."""
    penv = _make_parallel_env(n_actuators=3)
    actions = {agent: np.array([float(i)]) for i, agent in enumerate(penv.agents)}
    arr = penv._concat_actions(actions)
    np.testing.assert_allclose(arr, np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_parallel_array_to_dict():
    """Flat observation array -> dict keyed by agent name."""
    penv = _make_parallel_env(n_actuators=2, obs_per_actuator=3)
    obs = np.arange(6, dtype=np.float32)  # 2 agents * 3 obs each
    obs_dict = penv._split_observations(obs)
    assert set(obs_dict.keys()) == set(penv.agents)
    np.testing.assert_allclose(obs_dict[penv.agents[0]], np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_allclose(obs_dict[penv.agents[1]], np.array([3, 4, 5], dtype=np.float32))


def test_parallel_missing_agent_raises():
    """A dict missing one agent should raise ValueError matching 'Missing action for agent'."""
    penv = _make_parallel_env(n_actuators=3)
    actions = {agent: np.array([1.0]) for agent in penv.agents[:-1]}  # drop last
    with pytest.raises(ValueError, match="Missing action for agent"):
        penv._concat_actions(actions)


# =========================================================================
# AFC controllers
# =========================================================================


def test_oppoctrl_zero_input_zero_output():
    ctrl = OppoCtrl(agent_list=["a"], ctrl_max_amp=0.5)
    obs = np.zeros((2, 1, 1))  # observation[1, 0, 0] is 0
    action = ctrl.policy(obs)
    np.testing.assert_allclose(action, np.array([0.0], dtype=np.float32))


def test_oppoctrl_nonzero():
    ctrl = OppoCtrl(agent_list=["a"], ctrl_max_amp=2.0)
    obs = np.zeros((2, 1, 1))
    obs[1, 0, 0] = 1.5
    action = ctrl.policy(obs)
    np.testing.assert_allclose(action, np.array([-3.0], dtype=np.float32))


def test_blctrl_pure_numpy():
    ctrl = BLCtrl(agent_list=["a"], ctrl_max_amp=0.7)
    obs = np.array([0.123])  # blctrl ignores observation
    action = ctrl.policy(obs)
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, np.array([0.7], dtype=np.float32))


def test_zeroctrl_returns_zeros():
    ctrl = ZeroCtrl(agent_list=["a", "b"])
    obs = np.array([10.0, -7.0])
    action = ctrl.policy(obs)
    np.testing.assert_allclose(action, np.array([0.0], dtype=np.float32))


def test_sinwave_periodicity():
    """SinWave should produce a sinusoid in x with period Lx / Kx."""
    Lx, Lz, Kx, Kz = 4.0, 2.0, 1.0, 1.0
    alpha = 1.0
    sw = SinWave(agent_list=["a"], ctrl_max_amp=alpha, Kx=Kx, Kz=Kz, Lx=Lx, Lz=Lz)

    # The action depends only on x via sin(Kx * 2 pi * x / Lx).
    # period Lx/Kx == 4.0, so action(x) == action(x + 4.0).
    samples = []
    for x in np.linspace(0.0, Lx, num=9):
        samples.append(float(sw.policy((x, 0.0))[0]))
    samples = np.array(samples)
    # At x in {0, Lx/2, Lx} -> sin = 0
    np.testing.assert_allclose(samples[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(samples[-1], 0.0, atol=1e-6)
    # Bracket: peak amplitude bounded by alpha
    assert np.max(np.abs(samples)) <= alpha + 1e-6


# =========================================================================
# make_afc_controller branches
# =========================================================================


class _ConfStub:
    """A barebones stand-in for env.conf used by make_afc_controller."""

    class _Runner:
        ctrl_max_amp = 0.5

    class _Simulation:
        Lx = 4.0
        Lz = 2.0

    runner = _Runner()
    simulation = _Simulation()


def _make_env_for_factory(n_actuators=2, obs_per_actuator=1):
    base = object.__new__(NekEnv)
    base.n_actuators = n_actuators
    base.obs_per_actuator = obs_per_actuator
    base.actuator_info = {
        "NID": np.arange(n_actuators, dtype=np.int32),
        "GLLID": np.arange(n_actuators, dtype=np.int32),
        "FACEID": np.zeros(n_actuators, dtype=np.int32),
        "ix": np.arange(n_actuators, dtype=np.int32),
        "iy": np.arange(n_actuators, dtype=np.int32),
        "iz": np.arange(n_actuators, dtype=np.int32),
        "x": np.arange(n_actuators, dtype=np.float64),
        "z": np.arange(n_actuators, dtype=np.float64),
    }
    base.conf = _ConfStub()
    return base


def test_make_afc_controller_oppoctrl():
    env = _make_env_for_factory()
    ctrl = make_afc_controller(env, ctrl_type="OC")
    # NekEnv branch returns a ControllerAdapter; the inner controller is OppoCtrl.
    assert hasattr(ctrl, "afc_controller")
    assert isinstance(ctrl.afc_controller, OppoCtrl)


def test_make_afc_controller_blctrl():
    env = _make_env_for_factory()
    ctrl = make_afc_controller(env, ctrl_type="BL")
    assert isinstance(ctrl.afc_controller, BLCtrl)


def test_make_afc_controller_sinwave():
    env = _make_env_for_factory()
    ctrl = make_afc_controller(env, ctrl_type="SIN")
    assert isinstance(ctrl.afc_controller, SinWave)
    # load_node_info should have been called for array-based envs that have actuator_info.
    assert hasattr(ctrl.afc_controller, "Node_Info")


def test_make_afc_controller_zeroctrl():
    env = _make_env_for_factory()
    ctrl = make_afc_controller(env, ctrl_type="ZERO")
    assert isinstance(ctrl.afc_controller, ZeroCtrl)


def test_make_afc_controller_none_treated_as_zero():
    env = _make_env_for_factory()
    ctrl = make_afc_controller(env, ctrl_type=None)
    assert isinstance(ctrl.afc_controller, ZeroCtrl)


def test_make_afc_controller_unknown_returns_none(capsys):
    """An unknown ctrl_type should warn and return None (defers to SB3 path in integrate())."""
    env = _make_env_for_factory()
    result = make_afc_controller(env, ctrl_type="totally-unknown")
    assert result is None


# =========================================================================
# integrate save_path fallback cascade
# =========================================================================


class _DummyEnv:
    """Minimal env stub for testing integrate's save-path logic.

    integrate(env, t_span=...) drives the env via reset/step. We expose just
    enough surface for the save-path branch to be exercised; the actual
    simulation calls return tiny arrays so the integration loop terminates
    quickly.
    """

    def __init__(self, *, conf=None, run_folder=None, action_shape=(1,)):
        self.conf = conf
        if run_folder is not None:
            self.run_folder = run_folder
        # Action space stub used by integrate()'s default-action path.
        self.action_space = types.SimpleNamespace(shape=action_shape)
        self._step_count = 0

    def reset(self):
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def step(self, action):
        self._step_count += 1
        # done after 2 steps to bound the loop
        done = self._step_count >= 2
        return (np.zeros(self.action_space.shape, dtype=np.float32), 0.0, done, {})


def _make_conf_with_record(run_name="testrun", evaluation=False, rank=0, agent_run_name=None):
    return OmegaConf.create(
        {
            "runner": {
                "vars_record": True,
                "vars_record_freq": 1,
                "evaluation": evaluation,
                "rank": rank,
                **({"agent_run_name": agent_run_name} if agent_run_name is not None else {}),
            },
            "logging": {"run_name": run_name},
            "simulation": {"dt": 1.0},
        }
    )


def test_integrate_save_path_uses_run_folder(tmp_path, monkeypatch):
    """When the env exposes .run_folder, integrate should save there."""
    monkeypatch.chdir(tmp_path)
    conf = _make_conf_with_record(run_name="abc", evaluation=False)
    env = _DummyEnv(conf=conf, run_folder=str(tmp_path / "my_run"))
    integrate_module.integrate(env, t_span=(0.0, 5.0), dt=1.0, max_steps=3)
    saved = list((tmp_path / "my_run").glob("vars_record_*.mat"))
    assert saved, f"expected a vars_record_*.mat in {tmp_path / 'my_run'}"


def test_integrate_save_path_run_path_file_takes_priority(tmp_path, monkeypatch):
    """If dir-files/RUN_PATH_<agent_run_name>.txt exists, integrate should read it
    and save there in preference to env.run_folder."""
    monkeypatch.chdir(tmp_path)
    # Set up the RUN_PATH file
    (tmp_path / "dir-files").mkdir()
    target_dir = tmp_path / "from_run_path_file"
    (tmp_path / "dir-files" / "RUN_PATH_7.txt").write_text(f"{target_dir}\n")
    conf = _make_conf_with_record(run_name="ignored", evaluation=False, agent_run_name=7)
    # provide a competing run_folder that should be ignored
    env = _DummyEnv(conf=conf, run_folder=str(tmp_path / "should_be_ignored"))
    integrate_module.integrate(env, t_span=(0.0, 5.0), dt=1.0, max_steps=3)
    saved = list(target_dir.glob("vars_record_*.mat"))
    assert saved, f"expected a vars_record_*.mat in {target_dir}"


def test_integrate_save_path_wrapped_env_uses_inner_run_folder(tmp_path, monkeypatch):
    """For wrapped envs (e.g. NekParallelEnv), the inner env.env.run_folder is used."""
    monkeypatch.chdir(tmp_path)
    conf = _make_conf_with_record(run_name="w")
    inner = _DummyEnv(conf=conf, run_folder=str(tmp_path / "inner_run"))
    # Wrapped env: outer has no run_folder, but env.env does.
    outer = _DummyEnv(conf=conf, run_folder=None)
    outer.env = inner
    integrate_module.integrate(outer, t_span=(0.0, 5.0), dt=1.0, max_steps=3)
    saved = list((tmp_path / "inner_run").glob("vars_record_*.mat"))
    assert saved


def test_integrate_save_path_constructed_from_config_evaluation(tmp_path, monkeypatch):
    """No run_folder and no RUN_PATH file: fall back to runs/<run_name>/env_<rank:03d>."""
    monkeypatch.chdir(tmp_path)
    conf = _make_conf_with_record(run_name="eval_run", evaluation=True, rank=4)
    env = _DummyEnv(conf=conf, run_folder=None)
    integrate_module.integrate(env, t_span=(0.0, 5.0), dt=1.0, max_steps=3)
    expected = tmp_path / "runs" / "eval_run" / "env_004"
    saved = list(expected.glob("vars_record_*.mat"))
    assert saved, f"expected a vars_record_*.mat in {expected}"


def test_integrate_save_path_constructed_from_config_train(tmp_path, monkeypatch):
    """Non-evaluation fallback: runs/<run_name>/train."""
    monkeypatch.chdir(tmp_path)
    conf = _make_conf_with_record(run_name="train_run", evaluation=False)
    env = _DummyEnv(conf=conf, run_folder=None)
    integrate_module.integrate(env, t_span=(0.0, 5.0), dt=1.0, max_steps=3)
    expected = tmp_path / "runs" / "train_run" / "train"
    saved = list(expected.glob("vars_record_*.mat"))
    assert saved, f"expected a vars_record_*.mat in {expected}"
