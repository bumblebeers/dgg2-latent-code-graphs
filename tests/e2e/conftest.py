"""
conftest.py - Pytest fixtures and infrastructure for DGG2 Latent Code Graphs E2E Test Suite.

Provides:
- Milestone & Tier marker registrations.
- Progressive testability helpers (graceful verification of current vs pending milestone artifacts).
- Ephemeral module and data loaders supporting both extracted workspace and fallback handoff packet v3.
- PREREG.json specification loader and schema validators.
"""

import atexit
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_ZIP = REPO_ROOT / "codegraph_handoff_packet_v3.zip"
PARENT_REPO = Path(os.path.expanduser("~/workspaces/arc-agi-3"))

# Ephemeral cache directory under /tmp/dgg2_codegraph/ (Rule 5 compliant)
_SESSION_CACHE = Path("/tmp/dgg2_codegraph/pytest_session_cache")


def pytest_configure(config):
    """Register custom markers to prevent unknown marker warnings."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Interaction tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenario tests")
    for m in range(1, 7):
        config.addinivalue_line("markers", f"m{m}: Milestone M{m} targeted tests")
    config.addinivalue_line("markers", "contract: Interface contract tests")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def parent_repo() -> Path:
    return PARENT_REPO


@pytest.fixture(scope="session")
def prereg_data() -> dict:
    """Load PREREG.json from repo root or extract from packet v3."""
    root_prereg = REPO_ROOT / "PREREG.json"
    if root_prereg.exists():
        with open(root_prereg, "r", encoding="utf-8") as f:
            return json.load(f)
    elif PACKET_ZIP.exists():
        with zipfile.ZipFile(PACKET_ZIP, "r") as z:
            data = z.read("codegraph_handoff/PREREG.json")
            return json.loads(data.decode("utf-8"))
    raise FileNotFoundError("PREREG.json not found in repository root or codegraph_handoff_packet_v3.zip")


def load_cg_module(module_name: str):
    """
    Import a code graph module (e.g. 'cg_ops', 'cg_sec27_spec').
    Looks in REPO_ROOT/cg first, falling back to codegraph_handoff_packet_v3.zip.
    """
    cg_path = REPO_ROOT / "cg" / f"{module_name}.py"
    if cg_path.exists():
        if str(REPO_ROOT / "cg") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "cg"))
        return importlib.import_module(module_name)

    if not PACKET_ZIP.exists():
        raise FileNotFoundError(f"Cannot load {module_name}: neither cg/ directory nor packet zip exists.")

    _SESSION_CACHE.mkdir(parents=True, exist_ok=True)
    # Unpack all cg/*.py files once to ensure inter-module imports resolve
    with zipfile.ZipFile(PACKET_ZIP, "r") as z:
        for member in z.namelist():
            if member.startswith("codegraph_handoff/cg/") and member.endswith(".py"):
                fname = os.path.basename(member)
                dst = _SESSION_CACHE / fname
                if not dst.exists():
                    with open(dst, "wb") as f:
                        f.write(z.read(member))

    if str(_SESSION_CACHE) not in sys.path:
        sys.path.insert(0, str(_SESSION_CACHE))

    target_py = _SESSION_CACHE / f"{module_name}.py"
    if not target_py.exists():
        raise FileNotFoundError(f"Module {module_name} not found in packet zip cg/")

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(target_py))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_data_filepath(filename: str) -> Path:
    """
    Get Path to a data pack file (e.g. 'gridA_train1608.npz').
    Returns path in REPO_ROOT/data if present, otherwise unpacks to session cache.
    """
    root_data = REPO_ROOT / "data" / filename
    if root_data.exists():
        return root_data

    if not PACKET_ZIP.exists():
        raise FileNotFoundError(f"Data file {filename} not found at {root_data} and zip is missing.")

    _SESSION_CACHE.mkdir(parents=True, exist_ok=True)
    target_data = _SESSION_CACHE / filename
    if not target_data.exists():
        with zipfile.ZipFile(PACKET_ZIP, "r") as z:
            zip_member = f"codegraph_handoff/data/{filename}"
            if zip_member not in z.namelist():
                raise FileNotFoundError(f"Member {zip_member} not in {PACKET_ZIP}")
            with z.open(zip_member) as src, open(target_data, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return target_data


def get_graph_filepath(graph_name: str) -> Path:
    """
    Get Path to a raw call graph file (e.g. 'httpx.json.gz').
    """
    root_graph = REPO_ROOT / "data" / "graphs" / graph_name
    if root_graph.exists():
        return root_graph

    if not PACKET_ZIP.exists():
        raise FileNotFoundError(f"Graph file {graph_name} not found at {root_graph} and zip missing.")

    _SESSION_CACHE.mkdir(parents=True, exist_ok=True)
    target_graph = _SESSION_CACHE / graph_name
    if not target_graph.exists():
        with zipfile.ZipFile(PACKET_ZIP, "r") as z:
            zip_member = f"codegraph_handoff/data/graphs/{graph_name}"
            if zip_member not in z.namelist():
                raise FileNotFoundError(f"Member {zip_member} not in {PACKET_ZIP}")
            with z.open(zip_member) as src, open(target_graph, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return target_graph


@pytest.fixture(scope="session")
def cg_loader():
    return load_cg_module


@pytest.fixture(scope="session")
def data_locator():
    return get_data_filepath


@pytest.fixture(scope="session")
def graph_locator():
    return get_graph_filepath


def _cleanup_session():
    if _SESSION_CACHE.exists():
        try:
            shutil.rmtree(_SESSION_CACHE)
        except Exception:
            pass


atexit.register(_cleanup_session)
