"""Nox sessions for testing, linting, and formatting."""

import nox

# Default sessions to run when just typing 'nox'
nox.options.sessions = ["lint", "format_check", "type_check"]


@nox.session
def lint(session):
    """Run ruff linter."""
    session.install("ruff")
    session.run("ruff", "check", "src", "noxfile.py")


@nox.session
def format_check(session):
    """Check code formatting with ruff."""
    session.install("ruff")
    session.run("ruff", "format", "--check", "src", "noxfile.py")


@nox.session
def format_apply(session):
    """Format code with ruff."""
    session.install("ruff")
    session.run("ruff", "format", "src", "noxfile.py")


@nox.session
def type_check(session):
    """Run mypy type checker."""
    session.install("mypy")
    session.install(".")
    session.run("mypy", "src")


@nox.session(python=["3.12", "3.13"])
def tests(session):
    """Run tests (placeholder - pytest not yet configured)."""
    session.install(".")
    # Will add pytest here later
    session.run("python", "-c", "import conformal; print(f'conformal v{conformal.__version__}')")
