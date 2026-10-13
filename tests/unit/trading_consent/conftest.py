"""Loaders for the trading-consent Hermes plugin.

The plugin lives at ``hermes_plugins/trading-consent/`` -- hyphenated, matching the real,
already-installed Hermes plugin convention on this host (``~/.hermes/plugins/tradingview-mcp``
is hyphenated too), which is why it can't be `import`ed with a normal dotted statement. These
fixtures load it the way a custom plugin loader would: via importlib against the file paths
directly, constructing a proper package for ``__init__.py`` so its ``from . import categories``
relative imports resolve exactly as they do under Hermes's real loader (confirmed against
plugins/security-guidance/__init__.py, which uses the identical `from . import patterns` pattern).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "hermes_plugins" / "trading-consent"


def _load_flat_module(name: str, path: Path):
    """Load a standalone module (no relative imports of its own) by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plugin_dir() -> Path:
    return PLUGIN_DIR


@pytest.fixture(scope="session")
def categories_mod():
    return _load_flat_module("trading_consent_categories", PLUGIN_DIR / "categories.py")


@pytest.fixture(scope="session")
def rule_key_mod():
    return _load_flat_module("trading_consent_rule_key", PLUGIN_DIR / "rule_key.py")


@pytest.fixture(scope="session")
def marker_mod():
    return _load_flat_module("trading_consent_marker", PLUGIN_DIR / "marker.py")


@pytest.fixture(scope="session")
def launcher_mod():
    return _load_flat_module("trading_consent_launcher", PLUGIN_DIR / "guarded_mcp_launcher.py")


@pytest.fixture(scope="session")
def trading_consent_pkg():
    """Load hermes_plugins/trading-consent/__init__.py as a real package named
    ``trading_consent_pkg`` so its `from . import categories` etc. resolve via
    submodule_search_locations, the same mechanism Hermes's own loader relies on.
    """
    pkg_name = "trading_consent_pkg"
    spec = importlib.util.spec_from_file_location(
        pkg_name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fresh_profile_home(tmp_path: Path) -> Path:
    """An empty directory standing in for a profile home the marker can be written under,
    isolated per test (no interaction with any real ~/.hermes profile)."""
    home = tmp_path / "fake-profile-home"
    home.mkdir()
    return home
