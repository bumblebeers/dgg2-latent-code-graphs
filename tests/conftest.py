"""Global pytest configuration for tests/."""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    """Register custom markers to prevent unknown marker warnings."""
    for m in range(1, 7):
        config.addinivalue_line("markers", f"m{m}: Milestone M{m} targeted tests")
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Interaction tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenario tests")
    config.addinivalue_line("markers", "contract: Interface contract tests")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
