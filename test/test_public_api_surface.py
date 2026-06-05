from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import hydrogym
import hydrogym.core as hydrogym_core


EXPECTED_ALL = {
    "CallbackBase",
    "FlowEnv",
    "PDEBase",
    "TransientSolver",
    "distributed",
    "firedrake",
    "maia",
    "nek",
}


def test_dunder_all_is_expected_set():
    assert set(hydrogym.__all__) == EXPECTED_ALL


@pytest.mark.parametrize(
    "attr",
    ["CallbackBase", "FlowEnv", "PDEBase", "TransientSolver"],
)
def test_core_reexports_are_identity_with_core_module(attr):
    assert getattr(hydrogym, attr) is getattr(hydrogym_core, attr)


def test_distributed_subpackage_is_importable():
    import hydrogym.distributed as distributed

    assert distributed is hydrogym.distributed


def test_unknown_attribute_raises_useful_attribute_error():
    with pytest.raises(AttributeError) as excinfo:
        hydrogym.does_not_exist_xyz
    message = str(excinfo.value)
    assert "hydrogym" in message
    assert "does_not_exist_xyz" in message


def _firedrake_importable() -> bool:
    try:
        import firedrake  # noqa: F401
    except ImportError:
        return False
    return True


def test_lazy_submodule_access_is_cached_distributed():
    # `distributed` exercises the same caching branch in
    # hydrogym/__init__.py:__getattr__ (the `globals()[name] = module`
    # line) without requiring the firedrake binary stack. Pinning this
    # against distributed keeps the caching guard meaningful in the
    # pure-Python CI environment.
    first = hydrogym.distributed
    second = hydrogym.distributed
    assert first is second


@pytest.mark.skipif(
    not _firedrake_importable(),
    reason="firedrake binary stack not available in pure-Python test env",
)
def test_lazy_submodule_access_is_cached_firedrake():
    first = hydrogym.firedrake
    second = hydrogym.firedrake
    assert first is second


def test_lazy_submodules_not_loaded_before_access():
    # Pins hydrogym/__init__.py lazy-loading via module __getattr__:
    # the three MPI-touching submodules must NOT appear in the package
    # globals until something explicitly references them. This must run
    # in a subprocess because earlier tests in this session (or even
    # earlier assertions in this file) may have triggered loading.
    code = textwrap.dedent(
        """
        import hydrogym
        names = vars(hydrogym)
        for sub in ("firedrake", "maia", "nek"):
            assert sub not in names, f"{sub} was eagerly loaded into hydrogym package globals"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stdout.strip().endswith("OK")
