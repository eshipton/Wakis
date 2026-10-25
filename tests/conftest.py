import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", help="Enable GPU support for SolverFIT3D"
    )
    parser.addoption(
        "--interactive",
        action="store_true",
        help="Enable interactive (onscreen) PyVista rendering",
    )
    parser.addoption(
        "--debug-plots",
        action="store_true",
        help="Show simulation versus static reference comparison plots",
    )


@pytest.fixture(scope="session")
def use_gpu(request):
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def flag_offscreen(request):
    # If --interactive is set, offscreen should be False
    return not request.config.getoption("--interactive")


@pytest.fixture
def plot_comparison(request):
    """Show the values used by a regression assertion when requested."""
    if not request.config.getoption("--debug-plots"):
        return lambda actual, expected, title: None

    import matplotlib.pyplot as plt
    import numpy as np

    def plot(actual, expected, title):
        actual = np.atleast_1d(actual)
        expected = np.atleast_1d(expected)
        fig, ax = plt.subplots()
        ax.plot(np.arange(actual.size), actual, label="Simulation")
        ax.plot(
            np.arange(expected.size),
            expected,
            "o",
            label="Static reference",
            markersize=3,
        )
        ax.set(title=f"{request.node.name}: {title}", xlabel="Sample index")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return plot


# --- Tutorial-local modules -------------------------------------------------------------------------------------
# Every tutorial folder has its own `_helpers.py`. When several tutorial tests run in one pytest process (e.g. a
# Batch shard), the name `_helpers` would resolve to whichever tutorial was imported first. Before each tutorial
# test module is imported and before each of its tests runs, point `_helpers` (and the import path) at that test's
# own tutorial folder, so module-level and in-function imports both get the right module.
import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"
_LOCAL_MODULES = ("_helpers",)
_LOADED = {}


# Tutorial tests that do not follow the test_tut_NN_MM_ naming: file name -> "NN-MM".
_EXTRA_TUTORIAL_TESTS = {
    "test_066_maxwell_perspective_acceleration.py": "08-03",
    "test_067_maxwellian_laser_acceleration.py": "08-03",
}


def _tutorial_dir(path):
    """Tutorial folder of a test file named test_tut_NN_MM_<slug>.py (or listed above), or None."""
    name = Path(str(path)).name
    m = re.match(r"test_tut_(\d\d)_(\d\d)_", name)
    if m:
        ch, num = m.groups()
    elif name in _EXTRA_TUTORIAL_TESTS:
        ch, num = _EXTRA_TUTORIAL_TESTS[name].split("-")
    else:
        return None
    hits = sorted(_TUTORIALS.glob(f"{ch}_*/{ch}-{num}_*"))
    return hits[0] if hits else None


def _activate_tutorial(tdir):
    d = str(tdir)
    if d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)
    mods = _LOADED.setdefault(d, {})
    for name in _LOCAL_MODULES:
        if name not in mods:
            src = tdir / f"{name}.py"
            if not src.exists():
                sys.modules.pop(name, None)
                continue
            spec = importlib.util.spec_from_file_location(name, src)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mods[name] = mod
        sys.modules[name] = mods[name]


def pytest_collectstart(collector):
    # Module objects for a whole directory are created first and imported later, when each one is collected:
    # activate the tutorial right before its test module is imported.
    if isinstance(collector, pytest.Module):
        tdir = _tutorial_dir(collector.path)
        if tdir is not None:
            _activate_tutorial(tdir)


def pytest_runtest_setup(item):
    tdir = _tutorial_dir(item.path)
    if tdir is not None:
        _activate_tutorial(tdir)
