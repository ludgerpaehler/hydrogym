from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hydrogym.core import TransientSolver


def test_init_dt_explicit(stub_pde, make_stub_solver):
    solver = make_stub_solver(stub_pde, dt=0.25)
    assert solver.dt == 0.25


def test_init_dt_from_flow_default(stub_pde, make_stub_solver):
    # StubSolver default dt is 0.1; PDE DEFAULT_DT is also 0.1, so plain
    # construction is consistent with the flow default either way.
    solver = make_stub_solver(stub_pde)
    assert solver.dt == stub_pde.DEFAULT_DT == 0.1
    # Exercise the actual TransientSolver fallback path (dt=None ->
    # flow.DEFAULT_DT) — pins core.py:281-283.
    solver_none = make_stub_solver(stub_pde, dt=None)
    assert solver_none.dt == stub_pde.DEFAULT_DT


def test_solve_no_args_raises(stub_solver):
    # pins core.py:316-317 — XOR validation raises ValueError.
    with pytest.raises(ValueError, match="exactly one of t_span or num_steps"):
        stub_solver.solve()


def test_solve_both_args_raises(stub_solver):
    with pytest.raises(ValueError, match="exactly one of t_span or num_steps"):
        stub_solver.solve(t_span=(0.0, 1.0), num_steps=10)


def test_solve_num_steps_iterates_step(stub_solver):
    stub_solver.solve(num_steps=5)
    iters = [call[0] for call in stub_solver.step_calls]
    assert iters == [0, 1, 2, 3, 4]
    # With no controller, control is passed as None.
    assert all(call[1] is None for call in stub_solver.step_calls)


def test_solve_num_steps_zero_pins_behavior(stub_solver):
    # pins core.py:321-360 — solve(num_steps=0) leaves `flow` unbound and
    # the return statement raises UnboundLocalError. This is a latent bug;
    # we pin the current behavior so an accidental fix is noticed.
    with pytest.raises(UnboundLocalError):
        stub_solver.solve(num_steps=0)


@pytest.mark.parametrize(
    "t_span,dt,expected",
    [
        ((0.0, 0.5), 0.1, 5),  # np.arange(0, 0.5, 0.1) -> 5 elements
        ((0.0, 1.0), 0.25, 4),  # np.arange(0, 1.0, 0.25) -> 4 elements
        ((0.0, 0.3), 0.1, 3),
    ],
)
def test_solve_t_span_arange_boundary(stub_pde, make_stub_solver, t_span, dt, expected):
    # pins core.py:340 — uses np.arange(*t_span, dt), so the endpoint is
    # exclusive (typical numpy semantics).
    solver = make_stub_solver(stub_pde, dt=dt)
    solver.solve(t_span=t_span)
    assert len(solver.step_calls) == expected


def test_solve_collect_rewards_true(stub_pde, make_stub_solver):
    stub_pde.set_objective(2.0)
    solver = make_stub_solver(stub_pde, dt=0.1)
    result = solver.solve(num_steps=3, collect_rewards=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    flow, rewards = result
    assert flow is stub_pde
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (3,)
    # pins core.py:332-334 — rewards collected are the raw objective values,
    # NOT -dt * objective. The -dt scaling happens at the FlowEnv layer
    # (core.py:490, core.py:526), not in TransientSolver.solve.
    assert np.allclose(rewards, [2.0, 2.0, 2.0])


def test_solve_collect_rewards_false(stub_solver):
    result = stub_solver.solve(num_steps=3, collect_rewards=False)
    assert not isinstance(result, tuple)
    assert result is stub_solver.flow


def test_solve_with_controller(stub_solver):
    def ctrl(t, y):
        ctrl.calls.append((t, y))
        return [0.5]

    ctrl.calls = []

    stub_solver.solve(num_steps=3, controller=ctrl)

    assert len(ctrl.calls) == 3
    # StubSolver.step does not advance flow.t, so all controller invocations
    # observe t=0.0; the observations come from StubPDE.get_observations().
    for t, y in ctrl.calls:
        assert t == 0.0
        assert isinstance(y, np.ndarray)

    assert len(stub_solver.step_calls) == 3
    for _, control in stub_solver.step_calls:
        assert control == [0.5]


def test_solve_callbacks_invoked(stub_solver):
    cb1 = MagicMock()
    cb2 = MagicMock()

    stub_solver.solve(num_steps=3, callbacks=[cb1, cb2])

    # pins core.py:336-337 — callbacks invoked every step (no interval gate
    # at the solver level; CallbackBase.__call__ handles interval logic).
    # call_count tracks only __call__ invocations, not .close().
    assert cb1.call_count == 3
    assert cb2.call_count == 3
    cb1.close.assert_called_once()
    cb2.close.assert_called_once()
    # The callbacks receive (iter, t, flow).
    for i, call in enumerate(cb1.call_args_list):
        args, _ = call
        assert args[0] == i
        assert args[2] is stub_solver.flow


def test_solve_callbacks_close_called_in_order(stub_solver):
    parent = MagicMock()
    cb1 = parent.cb1
    cb2 = parent.cb2

    stub_solver.solve(num_steps=2, callbacks=[cb1, cb2])

    # Extract just the close() calls and verify cb1.close precedes cb2.close.
    close_call_names = [name for name, _, _ in parent.mock_calls if name.endswith(".close")]
    assert close_call_names == ["cb1.close", "cb2.close"]


def test_solve_returns_flow_when_collect_rewards_false(stub_solver):
    # Complement to test_solve_collect_rewards_false: same expectation via
    # t_span path to confirm both branches return raw flow.
    result = stub_solver.solve(t_span=(0.0, 0.3))
    assert result is stub_solver.flow


def test_transient_solver_step_not_implemented(stub_pde):
    # Sanity check: the base TransientSolver.step is abstract-by-convention
    # (raises NotImplementedError); pinning ensures subclasses must override.
    solver = TransientSolver(stub_pde, dt=0.1)
    with pytest.raises(NotImplementedError):
        solver.step(0)
