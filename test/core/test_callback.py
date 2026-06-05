from __future__ import annotations

import pytest

from hydrogym.core import CallbackBase


def test_interval_three_emits_every_third_call():
    cb = CallbackBase(interval=3)
    results = [cb(iter=i, t=0.0, flow=None) for i in range(7)]
    assert results == [True, False, False, True, False, False, True]


def test_interval_one_always_emits():
    cb = CallbackBase(interval=1)
    assert all(cb(iter=i, t=0.0, flow=None) for i in range(5))


def test_interval_zero_raises_zerodivision_or_pins_behavior():
    cb = CallbackBase(interval=0)
    with pytest.raises(ZeroDivisionError):
        cb(iter=0, t=0.0, flow=None)


def test_close_is_callable_noop():
    assert CallbackBase(interval=1).close() is None


def test_subclass_override():
    # core.py:258-269 -- __call__ returns a bool gate; the caller (e.g.
    # TransientSolver.solve, core.py:336-337) invokes the callback every
    # iteration regardless of the return value. So a subclass counter
    # bumped inside __call__ increments on EVERY call, not just emissions.
    class CountingCB(CallbackBase):
        def __init__(self, interval):
            super().__init__(interval=interval)
            self.count = 0

        def __call__(self, iter, t, flow):
            self.count += 1
            return super().__call__(iter, t, flow)

    cb = CountingCB(interval=2)
    for i in range(5):
        cb(iter=i, t=0.0, flow=None)
    assert cb.count == 5
