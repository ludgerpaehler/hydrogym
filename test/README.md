# HydroGym test suite

The test suite has two parts:

## Pure-Python tests (CI-runnable)

These tests live under:

- `test/test_public_api_surface.py` — top-level package surface
- `test/lazy_import/` — lazy-import discipline regression guards
- `test/core/` — PDEBase / FlowEnv / TransientSolver / CallbackBase contracts
- `test/data_manager/` — HFDataManager profile detection + workspace prep
- `test/jax/` — JAX backend (needs the `[jax]` extra)
- `test/maia/` — MAIA Python wrappers (needs mpi4py importable; no mpirun)
- `test/nek/` — NEK Python wrappers (needs mpi4py importable; no mpirun)
- `test/jaxfluids/` — JAX-Fluids cross-backend resolver drift guard

They run on every push / PR via `.github/workflows/pytest.yml` in three parallel jobs (core, jax, mpi-pure-python).

To run locally:

```bash
pip install -e .
pip install pytest pytest-cov pytest-mock
# Pure-python (always works):
python -m pytest test/test_public_api_surface.py test/core test/data_manager test/lazy_import
# JAX (needs the extra):
pip install -e ".[jax]"
python -m pytest test/jax -m requires_jax
# MAIA / NEK pure-python (needs mpi4py importable):
pip install mpi4py
python -m pytest test/maia test/nek -m "requires_maia or requires_nek"
```

## Firedrake suite (Docker-only)

The original test files (`test_cyl.py`, `test_cavity.py`, `test_pinball.py`, `test_step.py`, `test_io.py` and their `*_grad.py` siblings) require the Firedrake stack. Run them inside the `lpaehler/hydrogym-env:stable` Docker container (the VSCode devcontainer is the easiest entry):

```bash
source /home/firedrake/firedrake/bin/activate
pip install -e .
cd test
python -m pytest -m requires_firedrake
```

The `*_grad.py` tests are additionally marked `slow` and are skipped by default — invoke with `-m "requires_firedrake and slow"` to run them. They are *currently expected to be run at your own risk*: the adjoint stack is fragile.

## Markers reference

- `requires_firedrake` — needs the Firedrake docker image
- `requires_maia` — needs the maia native binary + MPMD mpirun (NOT exercised in pure-python tests)
- `requires_nek` — needs the nek5000 native binary + MPMD mpirun (NOT exercised in pure-python tests)
- `requires_jaxfluids` — needs the jaxfluids dependency stack
- `requires_jax` — needs the `[jax]` extra installed
- `slow` — long-running gradient / training test

Defaults (set in `pyproject.toml [tool.pytest.ini_options].addopts`):
deselects `requires_firedrake`, `requires_maia`, `requires_nek`, `requires_jaxfluids`.
That means a bare `pytest` from a fresh checkout runs the pure-Python tests + (if installed) the JAX tests.
