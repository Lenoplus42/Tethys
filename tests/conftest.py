"""Pytest config: @pytest.mark.live tests hit the live LLM and are SKIPPED in the
normal sweep. Run them explicitly with: python -m pytest --run-live
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: hits the live LLM; skipped unless --run-live")


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="run @pytest.mark.live tests (hit the LLM)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live (hits the LLM)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
