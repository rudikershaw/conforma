"""Nox sessions for testing, linting, and formatting."""

import re
from pathlib import Path

from nox import Session, options
from nox_uv import session

options.default_venv_backend = "uv"
options.reuse_existing_virtualenvs = True
options.sessions = ["lint", "security", "tests", "integration", "documentation"]


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["lint", "dev"])
def lint(session: Session) -> None:
    """Run linters."""
    session.run("ruff", "check", "src", "tests", "noxfile.py")
    session.run("ruff", "format", "--check", "--diff", "src", "tests", "noxfile.py")
    session.run("mypy", "src", "noxfile.py")


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["lint"])
def format_apply(session: Session) -> None:
    """Format code with ruff."""
    session.run("ruff", "check", "--fix", "src", "tests", "noxfile.py")
    session.run("ruff", "format", "src", "tests", "noxfile.py")


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["security"])
def security(session: Session) -> None:
    """Audit dependencies for security vulnerabilities."""
    session.run("bandit", "-r", "src")
    session.run("uv", "audit", external=True)


@session(python=["3.12", "3.13"], uv_quiet=True, uv_groups=["test"])
def tests(session: Session) -> None:
    """Run tests with appropriate numpy version for each Python version.

    - Python 3.12: numpy 1.24+ (oldest supported)
    - Python 3.13: numpy 2.0+ (only version with 3.13 wheels)
    """
    is3_12 = session.python == "3.12"
    session.install("numpy>=1.24.0,<2.0.0" if is3_12 else "numpy>=2.0.0,<3.0.0")

    # Validate that we have the expected numpy version for this Python version
    numpy_version = session.run("python", "-c", "import numpy; print(numpy.__version__)", silent=True)
    if not numpy_version:
        session.error("Failed to detect numpy version")
    numpy_version = numpy_version.strip()
    expected_major = 1 if is3_12 else 2
    if not numpy_version.startswith(f"{expected_major}."):
        session.error(f"Expected numpy {expected_major}.x but got {numpy_version}")

    session.run(
        "pytest",
        "tests/unit",
        "--doctest-modules",
        "--pyargs",
        "conformal",
        "--cov=conformal",
        "--cov-report=term-missing",
        "--cov-report=html",
    )


@session(python=["3.12", "3.13"], uv_quiet=True, uv_groups=["integration"])
def integration(session: Session) -> None:
    """Run integration tests against real models and datasets."""
    session.run("pytest", "tests/integration", "-o", "addopts=")


@session(uv_no_install_project=True, uv_quiet=True)
def documentation(session: Session) -> None:
    """Inject code examples from source files into markdown documentation."""
    docs_dir = Path("docs")
    insert_pattern = re.compile(r"(<!-- INSERT_CODE:(?P<filepath>[^\s>]+) -->).*?(<!--CODE_END -->)", re.DOTALL)

    for md_file in docs_dir.rglob("*.md"):
        original = md_file.read_text(encoding="utf-8")

        def replace_block(match: re.Match[str], _md_file: Path = md_file) -> str:
            filepath = match.group("filepath").strip()
            source = Path(filepath)
            if not source.exists():
                session.warn(f"Referenced file not found: {filepath} (in {_md_file})")
                return match.group(0)
            code = source.read_text(encoding="utf-8").rstrip()
            lang = source.suffix.lstrip(".")
            return f"<!-- INSERT_CODE:{filepath} -->\n```{lang}\n{code}\n```\n<!--CODE_END -->"

        updated = insert_pattern.sub(replace_block, original)

        if updated != original:
            md_file.write_text(updated, encoding="utf-8")
            session.log(f"Updated: {md_file}")
