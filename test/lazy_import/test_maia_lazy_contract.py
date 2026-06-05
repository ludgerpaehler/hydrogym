from __future__ import annotations

import inspect
import re
import subprocess
import sys
import textwrap

import pytest

try:
    import mpi4py  # noqa: F401

    SKIP_NO_MPI4PY = False
except ImportError:
    SKIP_NO_MPI4PY = True


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_import_maia_no_mpi4py():
    result = _run(
        """
        import hydrogym.maia
        import sys
        print("mpi4py" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


@pytest.mark.skipif(SKIP_NO_MPI4PY, reason="mpi4py not installed")
def test_attribute_access_loads_mpi4py():
    result = _run(
        """
        import hydrogym.maia
        _ = hydrogym.maia.MaiaFlowEnv
        import sys
        print("mpi4py" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "True\n"


def test_eager_symbols_available_without_mpi():
    result = _run(
        """
        import hydrogym.maia
        print(
            hasattr(hydrogym.maia, "HFDataManager"),
            hasattr(hydrogym.maia, "MaiaWorkspace"),
            hasattr(hydrogym.maia, "prepare_maia_workspace"),
            "mpi4py" in __import__("sys").modules,
        )
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "True True True False\n"


def test_mpi_attrs_set_matches_lazy_table():
    import hydrogym.maia as maia_mod

    lazy_table = set(maia_mod._MPI_ATTRS)
    all_set = set(maia_mod.__all__)
    eager_set = all_set - lazy_table

    assert lazy_table.issubset(all_set), (
        f"lazy names not in __all__: {lazy_table - all_set}"
    )
    assert eager_set, "expected at least one eager symbol in __all__"

    src = inspect.getsource(maia_mod._load_mpi_deps)
    for name in lazy_table:
        # Each lazy name must be wired in via a (name, obj) tuple inside the
        # setattr loop in hydrogym/maia/__init__.py:_load_mpi_deps. We accept
        # either a literal setattr(_mod, "<name>", ...) call OR a "<name>"
        # string literal that the loop iterates over.
        pattern = rf'["\']{re.escape(name)}["\']'
        assert re.search(pattern, src), (
            f"lazy name {name!r} missing from _load_mpi_deps body"
        )


def test_unknown_attribute_raises():
    result = _run(
        """
        import hydrogym.maia
        hydrogym.maia.NotAClass
        """
    )
    assert result.returncode != 0
    assert "AttributeError" in result.stderr
