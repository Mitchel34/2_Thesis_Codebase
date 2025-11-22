"""Basic smoke tests to ensure repository metadata is accessible."""

from pathlib import Path


def test_readme_mentions_project_name() -> None:
    """README should exist and reference the Watauga project."""

    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert readme.exists(), "README.md missing at repository root"
    contents = readme.read_text().lower()
    assert "watauga" in contents, "README should mention the Watauga study"
