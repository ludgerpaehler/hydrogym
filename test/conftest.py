"""Shared pytest fixtures + stubs for the pure-Python test tree."""

from __future__ import annotations

import numpy as np
import pytest

from hydrogym.core import PDEBase, TransientSolver


class StubPDE(PDEBase):
    """Minimal concrete PDEBase used by core-layer tests.

    - num_inputs / num_outputs configurable
    - records reset / checkpoint calls for assertion
    - evaluate_objective() returns a configurable scalar (default 0.0)

    NOTE on the abstract surface: PDEBase declares 11 abstract members
    (num_inputs, num_outputs, load_mesh, initialize_state, init_bcs,
    copy_state, save_checkpoint, load_checkpoint, get_observations,
    evaluate_objective, render). All are overridden here. `num_inputs`
    and `num_outputs` must be class-level descriptors (not instance
    attributes set in __init__) to satisfy ABCMeta at instantiation
    time -- so they are exposed as properties backed by `_num_inputs` /
    `_num_outputs`, which the caller may set before super().__init__().
    """

    DEFAULT_DT = 0.1

    def __init__(self, num_inputs: int = 1, num_outputs: int = 1, **kwargs):
        # IMPORTANT: PDEBase.__init__ -> reset() -> reset_controls() reads
        # self.num_inputs and calls self.init_bcs() / load_mesh() /
        # initialize_state(). Set the backing fields up front so those
        # abstract-method overrides have something to work with.
        self._num_inputs = int(num_inputs)
        self._num_outputs = int(num_outputs)
        self._objective_value = 0.0
        self.reset_calls: list[float] = []
        self.checkpoints_loaded: list[str] = []
        self.checkpoints_saved: list[str] = []
        self._initialize_state_calls = 0
        self._init_bcs_calls = 0
        super().__init__(**kwargs)

    # ---- abstract surface ----
    @property
    def num_inputs(self) -> int:  # type: ignore[override]
        return self._num_inputs

    @property
    def num_outputs(self) -> int:  # type: ignore[override]
        return self._num_outputs

    def load_mesh(self, name: str):  # type: ignore[override]
        # PDEBase.__init__ assigns the return value to self.mesh, so we
        # cannot expose `mesh` as a read-only property here. Just return
        # a sentinel string and let the base class store it.
        return f"<stub-mesh:{name}>"

    def initialize_state(self) -> None:  # type: ignore[override]
        self._initialize_state_calls += 1
        # Provide a trivial state vector so downstream code that touches
        # self.q does not blow up.
        self.q = np.zeros(self._num_outputs)

    def init_bcs(self) -> None:  # type: ignore[override]
        self._init_bcs_calls += 1

    def copy_state(self, deepcopy: bool = True):  # type: ignore[override]
        # Return a copy of self.q (set in initialize_state) so FlowEnv-style
        # consumers that snapshot initial states behave sensibly.
        q = getattr(self, "q", None)
        if q is None:
            return None
        return np.array(q, copy=True) if deepcopy else q

    def get_observations(self) -> np.ndarray:  # type: ignore[override]
        return np.zeros(self._num_outputs)

    def evaluate_objective(self, q=None) -> float:  # type: ignore[override]
        return self._objective_value

    def reset(self, q0=None, t: float = 0.0) -> None:  # type: ignore[override]
        self.reset_calls.append(float(t))
        # Defer to the base implementation so self.t, self.actuators and
        # the BCs are wired up the same way the production solvers expect.
        super().reset(q0=q0, t=t)

    def load_checkpoint(self, filename) -> None:  # type: ignore[override]
        self.checkpoints_loaded.append(str(filename))

    def save_checkpoint(self, filename) -> None:  # type: ignore[override]
        self.checkpoints_saved.append(str(filename))

    def render(self, **kwargs):  # type: ignore[override]
        return None

    # ---- test helpers ----
    def set_objective(self, value: float) -> None:
        self._objective_value = float(value)


class StubSolver(TransientSolver):
    """Minimal TransientSolver: records each .step() call.

    `.solve(...)` is the real implementation under test (inherited from
    TransientSolver). Only .step() is overridden.
    """

    def __init__(self, flow: StubPDE, dt: float = 0.1):
        self.step_calls: list[tuple[int, object]] = []
        super().__init__(flow, dt=dt)

    def step(self, iter: int, control=None):  # type: ignore[override]
        self.step_calls.append((iter, control))
        return self.flow


@pytest.fixture
def stub_pde() -> StubPDE:
    return StubPDE(num_inputs=1)


@pytest.fixture
def make_stub_pde():
    def factory(num_inputs: int = 1, **kwargs) -> StubPDE:
        return StubPDE(num_inputs=num_inputs, **kwargs)

    return factory


@pytest.fixture
def stub_solver(stub_pde) -> StubSolver:
    return StubSolver(stub_pde, dt=0.1)


@pytest.fixture
def make_stub_solver():
    def factory(pde: StubPDE, dt: float = 0.1) -> StubSolver:
        return StubSolver(pde, dt=dt)

    return factory


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Chdir to a tmp dir for tests that write into the cwd (legacy test_io)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
