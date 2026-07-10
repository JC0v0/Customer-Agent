"""Non-invasive project metadata checks.

These tests avoid importing application modules so they do not create config files,
start the Qt UI, connect to Pinduoduo, or call LLM services.
"""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pyproject_declares_expected_project_metadata():
    data = tomllib.loads(read_text("pyproject.toml"))

    project = data["project"]
    assert project["name"] == "agent-customer"
    assert project["requires-python"] == ">=3.11"
    assert project["readme"] == "README.md"


def test_pyproject_declares_non_runtime_quality_tooling():
    data = tomllib.loads(read_text("pyproject.toml"))

    optional_dependencies = data["project"]["optional-dependencies"]
    assert "build" in optional_dependencies
    assert "dev" in optional_dependencies
    assert any(dep.startswith("pytest") for dep in optional_dependencies["dev"])
    assert any(dep.startswith("ruff") for dep in optional_dependencies["dev"])

    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert data["tool"]["ruff"]["target-version"] == "py311"


def test_gitignore_excludes_runtime_and_quality_artifacts():
    gitignore = read_text(".gitignore")

    for pattern in [
        "config.json",
        "config.tmp",
        "temp/",
        "logs/",
        ".pytest_cache/",
        ".ruff_cache/",
        "htmlcov/",
    ]:
        assert pattern in gitignore


def test_readme_documents_required_local_configuration():
    readme = read_text("README.md")

    for text in [
        "uv sync",
        "uv run python app.py",
        "scripts/install_playwright.py",
        "llm.model_name",
        "llm.api_key",
        "llm.api_base",
        "business_hours.start",
        "business_hours.end",
    ]:
        assert text in readme
