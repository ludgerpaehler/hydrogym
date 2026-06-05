from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_import_hydrogym_does_not_import_mpi4py() -> None:
    result = _run("import hydrogym; import sys; print('mpi4py' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


def test_import_hydrogym_does_not_import_firedrake() -> None:
    result = _run("import hydrogym; import sys; print('firedrake' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


def test_import_hydrogym_does_not_import_jax() -> None:
    result = _run("import hydrogym; import sys; print('jax' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


def test_import_hydrogym_distributed_does_not_import_mpi4py() -> None:
    result = _run("import hydrogym.distributed; import sys; print('mpi4py' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


def test_attribute_access_caches() -> None:
    result = _run("import hydrogym; a = hydrogym.distributed; b = hydrogym.distributed; print(a is b)")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "True\n"


def test_unknown_attribute_raises() -> None:
    result = _run("import hydrogym; hydrogym.does_not_exist")
    assert result.returncode != 0
    assert "AttributeError" in result.stderr
