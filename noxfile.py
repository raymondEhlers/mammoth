from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import nox  # pyright: ignore [reportMissingImports]

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "pylint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run PyLint.
    """
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install(".", "pylint")
    session.run("pylint", "mammoth", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-e.[docs]", *extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install("sphinx")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/mammoth",
    )


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")


@nox.session(venv_backend="none")
def dev_cpp_vscode(session: nox.Session) -> None:
    """
    Prepare a .venv folder and create a build dir for vscode to read from.

    This should make the cmake extension work more nicely.
    Adapted from: https://github.com/scikit-build/scikit-build-core/issues/635

    NOTE:
        This can mix python versions (especially if executed with eg. pipx), so be
        very careful with this! It's better to use my other workarounds documented in
        the README
    """
    # Setup
    venv_name = ".venv-cpp-vscode"
    session.run(sys.executable, "-m", "venv", venv_name)
    # Dependencies
    session.run(f"{venv_name}/bin/pip", "install", "scikit-build-core[pyproject]", "pybind11")
    # We want to trigger the mammoth-cpp
    session.run(
        f"{venv_name}/bin/pip",
        "install",
        "--no-build-isolation",
        "--check-build-dependencies",
        "-ve",
        "mammoth-cpp",
        "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=1",
        "-Cbuild-dir=build",
    )
