import tempfile

import nox

nox.options.sessions = "lint", "tests", "mypy", "pytype"

_LOCATIONS = ["libspn_keras", "tests", "noxfile.py"]


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=["3.8", "3.7", "3.6"])
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mock"
    )
    session.run("pytest", *args)


@nox.session(python=["3.8", "3.7", "3.6"])
def lint(session):
    args = session.posargs or _LOCATIONS
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args, external=True)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or _LOCATIONS
    install_with_constraints(session, "black")
    session.run("black", *args, external=True)


@nox.session(python=["3.8", "3.7", "3.6"])
def mypy(session):
    args = session.posargs or _LOCATIONS
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python="3.7")
def pytype(session):
    """Run the static type checker."""
    args = session.posargs or ["--disable=import-error", *_LOCATIONS]
    install_with_constraints(session, "pytype")
    session.run("pytype", *args)


@nox.session(python=["3.8", "3.7", "3.6"])
def typeguard(session):
    args = session.posargs or ["-m", "not e2e"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "pytest", "pytest-mock", "typeguard")
    session.run("pytest", "--typeguard-packages=libspn_keras", *args)


@nox.session(python="3.8")
def docs(session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session,
        "sphinx",
        "sphinx-autodoc-typehints",
        "recommonmark",
        "sphinx_rtd_theme",
    )
    session.run("sphinx-build", "docs", "docs/_build")
