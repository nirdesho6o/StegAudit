"""Shared fixtures. `qwen` loads the real local model once per test session."""

import pytest


@pytest.fixture(scope="session")
def qwen():
    from stegaudit.lm import LocalLM
    return LocalLM()
