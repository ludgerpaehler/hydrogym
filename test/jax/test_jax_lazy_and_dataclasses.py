"""Tests for the JAX backend's pure-Python surface.

These tests intentionally avoid running any real PDE rollout — they exercise
the lazy-import contract, dataclass defaults, mesh constructors and the
small pure helper functions in ``hydrogym/jax/envs/channel.py``.

Per ``CLAUDE.md``: importing ``hydrogym.jax`` must NOT eagerly pull in any
Firedrake / MAIA / NEK / mpi4py code paths, even transitively.

The whole module is marked ``requires_jax`` so the default CI invocation
(``-m 'not requires_*'``) skips it cleanly when the ``[jax]`` extra is not
installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

# Bail out cleanly when the JAX wheel is not installed -- this makes the
# file safe to load under a bare CI even if the marker is overridden.
pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.requires_jax


# ---------------------------------------------------------------------------
# 1. Lazy-import contract
# ---------------------------------------------------------------------------


def test_import_jax_module_does_not_load_firedrake_or_mpi4py():
    """Importing ``hydrogym.jax`` must stay isolated from the heavy stacks.

    Runs in a subprocess so we get a clean ``sys.modules`` slate -- the
    current test process may already have those modules loaded by
    unrelated machinery.
    """
    script = textwrap.dedent(
        """
        import sys
        import hydrogym.jax  # noqa: F401
        print(
            "firedrake" in sys.modules,
            "mpi4py" in sys.modules,
            "hydrogym.maia" in sys.modules,
            "hydrogym.nek" in sys.modules,
        )
        """
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip().splitlines()[-1]
    assert line == "False False False False", (
        f"hydrogym.jax pulled in a forbidden module: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# 2. ChannelEnvParams defaults
# ---------------------------------------------------------------------------


def _make_channel_params():
    """Build a default ``ChannelEnvParams``.

    The base ``EnvParams`` in ``hydrogym/jax/env_core.py`` declares
    ``config: dict`` without a default. If gymnax's ``EnvParams`` is also
    a ``@struct.dataclass`` that propagates that into ChannelEnvParams,
    a bare ``ChannelEnvParams()`` will fail and we fall back to passing
    ``config={}``.
    """
    channel = pytest.importorskip("hydrogym.jax.envs.channel")
    try:
        return channel.ChannelEnvParams()
    except TypeError:
        return channel.ChannelEnvParams(config={})


def test_channel_env_params_defaults():
    params = _make_channel_params()

    # These values are pinned against
    # ``hydrogym/jax/envs/channel.py:ChannelEnvParams``.
    assert params.action_dim == 24
    assert params.obs_subsample == 8
    assert params.obs_include_components == 2
    assert params.k_det == 9

    assert params.Nx == 72
    assert params.Ny == 72
    assert params.Nz == 72

    assert params.nsteps == 50
    assert params.dt == pytest.approx(2e-4)

    assert params.max_steps_in_episode == 5000
    assert params.min_action == -1
    assert params.max_action == 1


# ---------------------------------------------------------------------------
# 3-5. KolmogorovFlow / FlowConfig
# ---------------------------------------------------------------------------


def _make_kolmogorov_flow():
    """Construct a default ``hydrogym.jax.envs.kolmogorov.FlowConfig``.

    ``FlowConfig.__init__`` calls into ``PDEBase.__init__`` which in turn
    invokes ``initialize_state`` -> ``jax.grad``. That works on a plain
    jax-cpu install with no GPU.
    """
    kolmogorov = pytest.importorskip("hydrogym.jax.envs.kolmogorov")
    return kolmogorov.FlowConfig()


def test_kolmogorov_flow_num_inputs_is_two():
    flow = _make_kolmogorov_flow()
    assert flow.num_inputs == 2


def test_load_mesh_shape():
    flow = _make_kolmogorov_flow()
    X, Y = flow.load_mesh("default")

    # Defaults: grid_size=(64, 64), domain_x=domain_y=(0, 2*pi).
    # NOTE the swapped indices in the source: ``nx = grid_size[1]``,
    # ``ny = grid_size[0]``. With grid_size=(64, 64) this is symmetric,
    # but the swap is preserved here intentionally as a pin.
    assert X.shape == (64, 64)
    assert Y.shape == (64, 64)

    x_ref = jnp.linspace(0.0, 2 * jnp.pi, 64)
    y_ref = jnp.linspace(0.0, 2 * jnp.pi, 64)
    X_ref, Y_ref = jnp.meshgrid(x_ref, y_ref, indexing="ij")
    np.testing.assert_array_equal(np.asarray(X), np.asarray(X_ref))
    np.testing.assert_array_equal(np.asarray(Y), np.asarray(Y_ref))


def test_load_fft_mesh_shape():
    flow = _make_kolmogorov_flow()
    kx, ky = flow.load_fft_mesh()

    # ``ky`` comes from ``rfftfreq`` so its length is N//2 + 1 = 33.
    assert kx.shape == (64, 33)
    assert ky.shape == (64, 33)

    # Zero-frequency entry sits at the [0, 0] corner for both axes
    # because both fftfreq and rfftfreq start at 0.
    assert float(kx[0, 0]) == 0.0
    assert float(ky[0, 0]) == 0.0


# ---------------------------------------------------------------------------
# 6. make_obs_grid_indices
# ---------------------------------------------------------------------------


def test_make_obs_grid_indices():
    channel = pytest.importorskip("hydrogym.jax.envs.channel")

    # Use Nx=Ny=16 with obs_subsample=4. The function returns two index
    # arrays Xi, Yi (NOT a flattened (n*n, 2) tensor as one might guess).
    Xi, Yi = channel.make_obs_grid_indices(Nx=16, Ny=16, n=4)
    assert Xi.shape == (4, 4)
    assert Yi.shape == (4, 4)

    # linspace(0, 15, 4) cast to int64 -> [0, 5, 10, 15] (floor of
    # 0, 5.0, 10.0, 15.0 respectively). Both axes use the same stride.
    expected_axis = jnp.array([0, 5, 10, 15], dtype=jnp.int64)
    Xi_ref, Yi_ref = jnp.meshgrid(expected_axis, expected_axis, indexing="ij")
    np.testing.assert_array_equal(np.asarray(Xi), np.asarray(Xi_ref))
    np.testing.assert_array_equal(np.asarray(Yi), np.asarray(Yi_ref))


# ---------------------------------------------------------------------------
# 7. get_obs_spectral_channel
# ---------------------------------------------------------------------------


def test_get_obs_spectral_channel():
    channel = pytest.importorskip("hydrogym.jax.envs.channel")

    # Build a small ChannelEnvState by hand. We override Nx/Ny/Nz so the
    # test doesn't allocate 72^3 floats, and pin k_det to a slot that
    # actually exists inside our small box.
    Nx, Ny, Nz, n = 8, 8, 4, 4
    try:
        params = channel.ChannelEnvParams(Nx=Nx, Ny=Ny, Nz=Nz, obs_subsample=n, k_det=1)
    except TypeError:
        params = channel.ChannelEnvParams(Nx=Nx, Ny=Ny, Nz=Nz, obs_subsample=n, k_det=1, config={})

    # Deterministic field so the output of the gather is predictable.
    U = jnp.arange(Nx * Ny * Nz, dtype=jnp.float32).reshape((Nx, Ny, Nz))
    V = jnp.zeros_like(U)
    W = U + 0.5  # so U-slice and W-slice are distinct

    state = channel.ChannelEnvState(
        time=jnp.int32(0),
        U=U,
        V=V,
        W=W,
        dt=jnp.float32(1e-3),
        terminal=jnp.bool_(False),
    )

    Xi, Yi = channel.make_obs_grid_indices(Nx, Ny, n)
    obs = channel.get_obs_spectral_channel(state, params, Xi, Yi)

    # Shape: stack of 2 (U, W) channels of size n*n, then flatten.
    assert obs.shape == (2 * n * n,)

    # Reconstruct the expected gather to PIN ordering: first n*n entries
    # are U[:,:,k_det][Xi, Yi], next n*n are W[:,:,k_det][Xi, Yi].
    expected = jnp.concatenate(
        [
            U[:, :, params.k_det][Xi, Yi].reshape(-1),
            W[:, :, params.k_det][Xi, Yi].reshape(-1),
        ]
    )
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(expected))


# ---------------------------------------------------------------------------
# 8. wss_compute hand value
# ---------------------------------------------------------------------------


def test_wss_compute_hand_value():
    channel = pytest.importorskip("hydrogym.jax.envs.channel")

    # Tiny field: Nx=Ny=2, Nz=4. Pick k=2 and z = [0, 0.25, 0.5, 1.0]
    # so dz = z[2] - z[0] = 0.5. Set U[:,:,0] = 0 and U[:,:,2] = 1 so
    # du_dz_wall = (1 - 0) / 0.5 = 2.0 at every (x, y).
    # With nu=1.9e-3, tau_w = 1.9e-3 * 2.0 = 3.8e-3 everywhere -> mean
    # is 3.8e-3.
    Nx, Ny, Nz = 2, 2, 4
    k = 2
    z = jnp.array([0.0, 0.25, 0.5, 1.0])

    U = jnp.zeros((Nx, Ny, Nz))
    U = U.at[:, :, k].set(1.0)

    wss = channel.wss_compute(k, U, nu=1.9e-3, z=z)
    assert wss.shape == ()  # scalar
    assert float(wss) == pytest.approx(3.8e-3, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. Duplicate FlowConfig drift detector
# ---------------------------------------------------------------------------


def test_duplicate_flow_config_classes_match():
    """Pin the dead-code drift between two FlowConfig copies.

    ``hydrogym.jax.flow.FlowConfig`` and
    ``hydrogym.jax.envs.kolmogorov.FlowConfig`` are *near* duplicates --
    same dataclass-style defaults, same public API, but the
    implementations diverged (kolmogorov.py uses ``jax.grad`` directly
    and imports helpers individually; flow.py routes everything through
    ``utils.compute_*`` and uses ``from jax import grad``).

    The classes are NOT byte-for-byte identical, but they MUST agree on
    the public surface so that downstream code can swap one for the
    other. This test pins that surface; if either copy drifts the test
    will fail and force a deliberate decision (delete the duplicate or
    update the assertion).
    """
    flow_mod = pytest.importorskip("hydrogym.jax.flow")
    kolmogorov_mod = pytest.importorskip("hydrogym.jax.envs.kolmogorov")

    FlowConfigA = flow_mod.FlowConfig
    FlowConfigB = kolmogorov_mod.FlowConfig

    # 1. Same DEFAULT_* class constants.
    for name in (
        "DEFAULT_REYNOLDS",
        "DEFAULT_WAVENUMBER",
        "DEFAULT_GRID_SIZE",
        "DEFAULT_OBS_SIZE",
    ):
        assert getattr(FlowConfigA, name) == getattr(FlowConfigB, name), f"DEFAULT_* drift on {name}"

    # DEFAULT_DOMAIN_X / Y are tuples of jnp scalars -- compare value-wise.
    for name in ("DEFAULT_DOMAIN_X", "DEFAULT_DOMAIN_Y"):
        a = getattr(FlowConfigA, name)
        b = getattr(FlowConfigB, name)
        assert len(a) == len(b) == 2
        assert float(a[0]) == float(b[0])
        assert float(a[1]) == float(b[1])

    # 2. Same public attribute / method set (ignoring dunders other than
    # __init__).
    def public_surface(cls):
        return {name for name in vars(cls) if not name.startswith("_") or name == "__init__"}

    surface_a = public_surface(FlowConfigA)
    surface_b = public_surface(FlowConfigB)
    assert surface_a == surface_b, (
        f"FlowConfig surface drift: only-in-A={surface_a - surface_b} only-in-B={surface_b - surface_a}"
    )

    # 3. Construct both and check the headline scalar invariants match.
    a = FlowConfigA()
    b = FlowConfigB()
    assert a.num_inputs == b.num_inputs == 2
    assert a.Re == b.Re
    assert a.k == b.k
    assert a.grid_size == b.grid_size
    assert a.obs_size == b.obs_size
