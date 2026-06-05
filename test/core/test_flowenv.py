from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hydrogym.core import FlowEnv


@pytest.fixture
def stub_classes(stub_pde, stub_solver):
    """Expose the StubPDE / StubSolver classes (FlowEnv instantiates them itself)."""
    return type(stub_pde), type(stub_solver)


@pytest.fixture
def base_env_config(stub_classes):
    PDECls, SolverCls = stub_classes

    def _build(**overrides) -> dict:
        cfg = {
            "flow": PDECls,
            "flow_config": {},
            "solver": SolverCls,
            "solver_config": {},
        }
        cfg.update(overrides)
        return cfg

    return _build


# ---------- Construction + env_config validation ----------


def test_construct_minimal(base_env_config, stub_classes):
    PDECls, SolverCls = stub_classes
    env = FlowEnv(base_env_config())
    assert isinstance(env.flow, PDECls)
    assert isinstance(env.solver, SolverCls)


def test_construct_with_solver_dt(base_env_config):
    env = FlowEnv(base_env_config(solver_config={"dt": 0.05}))
    assert env.solver.dt == 0.05


@pytest.mark.parametrize("bad_substeps", [0, -1])
def test_construct_actuation_config_invalid_num_substeps(base_env_config, bad_substeps):
    with pytest.raises(ValueError, match="num_substeps must be >= 1"):
        FlowEnv(base_env_config(actuation_config={"num_substeps": bad_substeps}))


def test_construct_actuation_config_invalid_reward_aggregation(base_env_config):
    with pytest.raises(ValueError, match="reward_aggregation must be"):
        FlowEnv(base_env_config(actuation_config={"reward_aggregation": "min"}))


def test_construct_deprecated_num_sim_substeps_per_actuation_warns(base_env_config):
    with pytest.warns(DeprecationWarning, match="num_sim_substeps_per_actuation is deprecated"):
        env = FlowEnv(base_env_config(actuation_config={"num_sim_substeps_per_actuation": 4}))
    assert env.num_substeps == 4


def test_construct_deprecated_reward_aggreation_rule_warns(base_env_config):
    # NB: typo "aggreation" preserved upstream at core.py:398-404
    with pytest.warns(DeprecationWarning, match="reward_aggreation_rule is deprecated"):
        env = FlowEnv(base_env_config(actuation_config={"reward_aggreation_rule": "sum"}))
    assert env.reward_aggregation == "sum"


# ---------- Reward semantics ----------


def test_step_reward_single_substep_sign_and_dt_scaling(base_env_config):
    env = FlowEnv(base_env_config())
    env.flow.set_objective(5.0)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(env.action_space.shape))
    assert reward == pytest.approx(-0.5)  # -dt * obj = -0.1 * 5


@pytest.mark.parametrize(
    "aggregation, expected_reward",
    [
        ("mean", -0.2),    # mean([1,2,3]) = 2 -> -dt*2 = -0.2
        ("sum", -0.6),     # sum([1,2,3]) = 6 -> -dt*6 = -0.6
        ("median", -0.2),  # median([1,2,3]) = 2 -> -dt*2 = -0.2
    ],
)
def test_step_reward_multi_substep_aggregation(base_env_config, monkeypatch, aggregation, expected_reward):
    env = FlowEnv(
        base_env_config(actuation_config={"num_substeps": 3, "reward_aggregation": aggregation})
    )
    env.reset()

    # Patch solver.solve to inject the canonical [1,2,3] reward vector.
    def fake_solve(*args, **kwargs):
        return env.flow, np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(env.solver, "solve", fake_solve)

    _, reward, _, _, _ = env.step(np.zeros(env.action_space.shape))
    assert reward == pytest.approx(expected_reward)


# ---------- Restart sampling ----------


def test_restart_empty_list_raises_indexerror(base_env_config):
    # Pins surprising asymmetry: PDEBase silently no-ops on empty restart list
    # (core.py:62 `if len(restart) > 0`) but FlowEnv unconditionally indexes
    # initial_states[0] at core.py:436 -> IndexError.
    with pytest.raises(IndexError):
        FlowEnv(base_env_config(flow_config={"restart": []}))


def test_restart_single_string_calls_load_once(base_env_config):
    # PDEBase.__init__ loads the single string at core.py:58; FlowEnv's string
    # branch at core.py:422-425 does NOT reload. Net: exactly one load_checkpoint.
    env = FlowEnv(base_env_config(flow_config={"restart": "x.h5"}))
    assert env.flow.checkpoints_loaded == ["x.h5"]


def test_restart_list_loads_all_initial_states(base_env_config):
    # PDEBase loads restart[0] (core.py:63), then FlowEnv iterates the full list
    # at core.py:431-433. So a 3-item list produces 4 load_checkpoint calls:
    # ["a.h5", "a.h5", "b.h5", "c.h5"].
    env = FlowEnv(base_env_config(flow_config={"restart": ["a.h5", "b.h5", "c.h5"]}))
    assert env.flow.checkpoints_loaded == ["a.h5", "a.h5", "b.h5", "c.h5"]
    assert len(env.initial_states) == 3


def test_reset_samples_checkpoint_index_deterministically(base_env_config):
    env = FlowEnv(base_env_config(flow_config={"restart": ["a.h5", "b.h5", "c.h5"]}))
    _, info1 = env.reset(seed=42)
    assert "checkpoint_index" in info1
    assert info1["checkpoint_index"] in {0, 1, 2}

    _, info2 = env.reset(seed=42)
    assert info2["checkpoint_index"] == info1["checkpoint_index"]


def test_reset_single_restart_info_has_no_checkpoint_index(base_env_config):
    # core.py:545-551: with len(initial_states) == 1 the info dict is the
    # bare empty {} -- "checkpoint_index" is omitted entirely.
    env = FlowEnv(base_env_config(flow_config={"restart": "x.h5"}))
    _, info = env.reset()
    assert "checkpoint_index" not in info
    assert info == {}


# ---------- max_steps boundary ----------


def test_max_steps_truncates_on_strict_overrun(base_env_config):
    # check_complete() is `self.iter > self.max_steps` (core.py:529, strict >).
    # With max_steps=3, num_substeps=1: iter advances 1,2,3,4. Truncation
    # therefore fires on the 4th step when iter becomes 4 > 3.
    env = FlowEnv(base_env_config(max_steps=3))
    env.reset()
    action = np.zeros(env.action_space.shape)

    truncs = []
    for _ in range(4):
        _, _, _, truncated, _ = env.step(action)
        truncs.append(truncated)

    assert truncs == [False, False, False, True]


# ---------- Callbacks ----------


def test_callbacks_invoked_in_order(base_env_config):
    cb1, cb2 = MagicMock(), MagicMock()
    env = FlowEnv(base_env_config(callbacks=[cb1, cb2]))
    env.reset()  # reset does NOT call callbacks (core.py:531-561)

    parent = MagicMock()
    parent.attach_mock(cb1, "cb1")
    parent.attach_mock(cb2, "cb2")
    env.step(np.zeros(env.action_space.shape))

    cb1.assert_called_once()
    cb2.assert_called_once()
    # cb1 must be invoked before cb2 within the same step (core.py:494-495)
    names = [c[0] for c in parent.mock_calls]
    assert names == ["cb1", "cb2"]


def test_callbacks_close_called_on_env_close(base_env_config):
    cb1, cb2 = MagicMock(), MagicMock()
    env = FlowEnv(base_env_config(callbacks=[cb1, cb2]))
    env.close()
    cb1.close.assert_called_once()
    cb2.close.assert_called_once()


# ---------- Observation stacking ----------


@pytest.mark.parametrize(
    "obs_in, expected",
    [
        (1.0, np.array([1.0])),
        ((1.0, 2.0), np.array([1.0, 2.0])),
        ([3.0, 4.0], np.array([3.0, 4.0])),
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
    ],
)
def test_stack_observations(base_env_config, obs_in, expected):
    env = FlowEnv(base_env_config())
    out = env.stack_observations(obs_in)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, expected)


def test_stack_observations_ndarray_passes_through_unchanged(base_env_config):
    # core.py:517-518: ndarray inputs are returned as-is (no copy, no dtype cast)
    env = FlowEnv(base_env_config())
    arr = np.array([1, 2, 3])
    out = env.stack_observations(arr)
    assert out is arr


# ---------- reset options ----------


def test_reset_options_t_propagates_to_flow_reset(base_env_config):
    env = FlowEnv(base_env_config())
    pre_calls = len(env.flow.reset_calls)
    env.reset(options={"t": 1.5})
    # Most recent reset call should reflect the t=1.5 propagated via core.py:553-555
    assert env.flow.reset_calls[-1] == 1.5
    assert len(env.flow.reset_calls) == pre_calls + 1
