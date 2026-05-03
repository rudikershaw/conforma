"""Nox sessions for testing, linting, and formatting."""

from nox import Session, options
from nox_uv import session

options.default_venv_backend = "uv"
options.reuse_existing_virtualenvs = True
options.sessions = ["lint", "security", "tests"]


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["lint", "dev"])
def lint(session: Session) -> None:
    """Run ruff linter."""
    session.run("ruff", "check", "src", "noxfile.py")
    session.run("ruff", "format", "--check", "--diff", "src", "noxfile.py")
    session.run("mypy", "src", "noxfile.py")


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["lint"])
def format_apply(session: Session) -> None:
    """Format code with ruff."""
    session.run("ruff", "format", "src", "noxfile.py")


@session(uv_no_install_project=True, uv_quiet=True, uv_groups=["lint"])
def security(session: Session) -> None:
    """Audit dependencies for security vulnerabilities."""
    session.run("pip-audit")


@session(python=["3.12", "3.13"], uv_quiet=True, uv_groups=["test"])
def tests(session: Session) -> None:
    """Run tests with appropriate numpy version for each Python version.

    - Python 3.12: numpy 1.24+ (oldest supported)
    - Python 3.13: numpy 2.0+ (only version with 3.13 wheels)
    """
    is3_12 = session.python == "3.12"
    session.install("pytest", "numpy>=1.24.0,<2.0.0" if is3_12 else "numpy>=2.0.0,<3.0.0")

    # Validate that we have the expected numpy version for this Python version
    numpy_version = session.run("python", "-c", "import numpy; print(numpy.__version__)", silent=True)
    if not numpy_version:
        session.error("Failed to detect numpy version")
    numpy_version = numpy_version.strip()
    expected_major = 1 if is3_12 else 2
    if not numpy_version.startswith(f"{expected_major}."):
        session.error(f"Expected numpy {expected_major}.x but got {numpy_version}")

    session.run("pytest", "tests", "-v", "--cov=conformal", "--cov-report=term-missing", "--cov-report=html")
