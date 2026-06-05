from __future__ import annotations

import numpy as np
import pytest

from hydrogym.core import ActuatorBase, PDEBase


def test_init_no_restart_default(stub_pde):
    assert stub_pde.checkpoints_loaded == []


def test_init_with_restart_string(make_stub_pde):
    pde = make_stub_pde(restart="some_path.h5")
    assert pde.checkpoints_loaded == ["some_path.h5"]


def test_init_with_restart_list(make_stub_pde):
    # pins current behavior — see core.py:59-63: PDEBase only loads restart[0]
    # (the first checkpoint); the remaining entries are deferred to FlowEnv.
    pde = make_stub_pde(restart=["a.h5", "b.h5", "c.h5"])
    assert pde.checkpoints_loaded == ["a.h5"]


def test_init_with_empty_restart_list(make_stub_pde):
    # pins asymmetry vs FlowEnv — see core.py:62: PDEBase silently no-ops on
    # empty list, while FlowEnv would raise IndexError on the same input.
    pde = make_stub_pde(restart=[])
    assert pde.checkpoints_loaded == []


def test_init_with_restart_tuple(make_stub_pde):
    # pins current behavior — see core.py:59: tuple is accepted alongside list.
    pde = make_stub_pde(restart=())
    assert pde.checkpoints_loaded == []


def test_init_with_invalid_restart_type_raises(make_stub_pde):
    with pytest.raises(ValueError, match="restart must be a string or list"):
        make_stub_pde(restart=123)


def test_set_control_none_zeros_actuators(make_stub_pde):
    pde = make_stub_pde(num_inputs=3)
    for a in pde.actuators:
        a.x = 7.5
    pde.set_control(None)
    assert [a.x for a in pde.actuators] == [0.0, 0.0, 0.0]


def test_set_control_list_correct_length(make_stub_pde):
    pde = make_stub_pde(num_inputs=3)
    pde.set_control([1.0, 2.0, 3.0])
    assert [a.x for a in pde.actuators] == [1.0, 2.0, 3.0]


def test_set_control_list_too_short_leaves_remainder(make_stub_pde):
    # pins surprising behavior — see core.py:183-184: set_control iterates
    # over the (shorter) act list, leaving trailing actuators untouched.
    pde = make_stub_pde(num_inputs=3)
    for a in pde.actuators:
        a.x = 99.0
    pde.set_control([5.0, 6.0])
    assert [a.x for a in pde.actuators] == [5.0, 6.0, 99.0]


def test_set_control_list_too_long_raises(make_stub_pde):
    # pins current behavior — see core.py:184: indexing self.actuators[i]
    # with i >= num_inputs raises IndexError.
    pde = make_stub_pde(num_inputs=3)
    with pytest.raises(IndexError):
        pde.set_control([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5.0, [5.0]),
        ([1, 2], [1, 2]),
        ((1, 2), [1, 2]),
    ],
)
def test_enlist_scalar_and_sequence(stub_pde, value, expected):
    assert stub_pde.enlist(value) == expected


def test_enlist_list_returns_list(stub_pde):
    result = stub_pde.enlist([1, 2])
    assert isinstance(result, list)
    assert result == [1, 2]


def test_enlist_ndarray(stub_pde):
    result = stub_pde.enlist(np.array([1, 2]))
    # pins current behavior — see core.py:171-173: enlist always calls
    # list(x), so an ndarray becomes a list of numpy scalars (not an ndarray).
    assert isinstance(result, list)
    np.testing.assert_array_equal(result, [1, 2])


def test_actuator_base_default_x():
    assert ActuatorBase().x == 0.0


def test_actuator_base_state_property_roundtrip():
    a = ActuatorBase()
    a.state = 3.14
    assert a.x == 3.14
    assert a.state == 3.14


def test_actuator_base_step_raises():
    with pytest.raises(NotImplementedError):
        ActuatorBase().step(0.1, 1.0)


def test_reset_controls_clobbers_custom_actuators(make_stub_pde):
    pde = make_stub_pde(num_inputs=2)
    sentinel = object()
    pde.actuators = [sentinel, sentinel]
    pde.reset_controls()
    assert len(pde.actuators) == 2
    assert all(isinstance(a, ActuatorBase) for a in pde.actuators)
    assert all(a.x == 0.0 for a in pde.actuators)


def test_advance_time_uses_control_state_when_act_none(make_stub_pde):
    # advance_time calls actuators[i].step(u, dt), not solver.step.
    # Use a recording actuator so we can capture the (u, dt) passed in.
    class RecordingActuator(ActuatorBase):
        def __init__(self, state=0.0):
            super().__init__(state=state)
            self.calls: list[tuple[float, float]] = []

        def step(self, u, dt):
            self.calls.append((u, dt))

    pde = make_stub_pde(num_inputs=1)
    rec = RecordingActuator(state=5.0)
    pde.actuators = [rec]

    pde.advance_time(0.1, act=None)

    # control_state was [5.0]; step should have been called with (5.0, 0.1).
    assert rec.calls == [(5.0, 0.1)]
    assert pde.t == pytest.approx(0.1)


def test_advance_time_advances_t(make_stub_pde):
    class NoopActuator(ActuatorBase):
        def step(self, u, dt):
            pass

    pde = make_stub_pde(num_inputs=1)
    pde.actuators = [NoopActuator()]
    pde.advance_time(0.25, act=[1.0])
    assert pde.t == pytest.approx(0.25)


def test_advance_time_wrong_length_act_raises(make_stub_pde):
    # pins current behavior — see core.py:204: a bare `assert` guards length,
    # so a mismatched act raises AssertionError (not ValueError).
    pde = make_stub_pde(num_inputs=2)
    with pytest.raises(AssertionError):
        pde.advance_time(0.1, act=[1.0, 2.0, 3.0])


def test_control_state_reflects_actuator_state(make_stub_pde):
    pde = make_stub_pde(num_inputs=3)
    pde.actuators[0].x = 1.0
    pde.actuators[1].x = 2.0
    pde.actuators[2].x = 3.0
    assert list(pde.control_state) == [1.0, 2.0, 3.0]
