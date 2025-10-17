from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Return the project README content for documentation smoke tests."""

    return Path("README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "snippet",
    [
        "Copula Lab v1.0",
        "Key capabilities",
        "Running the Streamlit app",
        "Testing and quality gates",
        "Release checklist",
    ],
)
def test_readme_includes_required_sections(
    readme_text: str,
    snippet: str,
) -> None:
    """README must retain the release-critical sections."""

    assert snippet in readme_text


def test_readme_mentions_supported_families(readme_text: str) -> None:
    """Ensure the README summarises the copula families shipped in v1.0."""

    families = (
        "Gaussian",
        "Student t",
        "Clayton",
        "Gumbel",
        "Frank",
        "Joe",
        "AMH",
    )
    for family in families:
        assert family in readme_text
