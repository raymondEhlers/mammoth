"""Utilities for using additional HEP software, such as RooUnfold

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import attrs

from mammoth.framework import production


@attrs.define(frozen=True)
class Dependency:
    name: str
    tasks: Sequence[str]
    depends_on_ROOT: bool


def _validate_productions_and_tasks_to_run(
    productions: Sequence[production.ProductionSettings] | None = None,
    tasks_to_run: Sequence[Sequence[str]] | None = None,
) -> list[list[str]]:
    if productions is None and tasks_to_run is None:
        _msg = "Need to provide either productions or tasks to run."
        raise ValueError(_msg)
    _tasks_to_run = [list(tasks) for tasks in tasks_to_run] if tasks_to_run is not None else []
    if productions is not None:
        _tasks_to_run.extend([list(prod.tasks_to_execute) for prod in productions])
    return _tasks_to_run


def determine_additional_worker_init_alibuild(
    productions: Sequence[production.ProductionSettings] | None = None,
    tasks_to_run: Sequence[Sequence[str]] | None = None,
    tasks_requiring_root: Sequence[str] | None = None,
    tasks_requiring_aliphysics: Sequence[str] | None = None,
    tasks_requiring_roounfold: Sequence[str] | None = None,
    software_version_tag: str = "latest",
) -> str:
    # Validation
    tasks_to_run = _validate_productions_and_tasks_to_run(productions=productions, tasks_to_run=tasks_to_run)
    if tasks_requiring_root is None:
        tasks_requiring_root = []
    if tasks_requiring_aliphysics is None:
        tasks_requiring_aliphysics = []
    if tasks_requiring_roounfold is None:
        tasks_requiring_roounfold = []

    _software_to_load: list[str] = []
    _additional_worker_init_script = ""
    _software_options = [
        # NOTE: It's important that ROOT is handled first so that we can remove it later if we need to do so. See below.
        Dependency("ROOT", tasks_requiring_root, False),
        Dependency("AliPhysics", tasks_requiring_aliphysics, True),
        Dependency("RooUnfold", tasks_requiring_roounfold, True),
    ]
    for _software in _software_options:
        # fmt: off
        if any(
            task in tasks_requiring_root
            for tasks in tasks_to_run
            for task in tasks
        ):
            if _software.depends_on_ROOT:
                # NOTE: This is a little unconventional to redefine the list here, but ROOT is already
                #       a dependency of this software, so we redefine the list to remove ROOT. (Otherwise, AliBuild
                #       may have issues loading the software since we're specifying versions for all software)
                _software_to_load = [s for s in _software_to_load if s != f"ROOT/{software_version_tag}"]
            _software_to_load.append(f"{_software.name}/{software_version_tag}")
        # fmt: on

    # If there is anything to load, add the initialization
    if _software_to_load:
        _additional_worker_init_script = (
            f"eval `/usr/bin/alienv -w /software/rehlers/alice/sw --no-refresh printenv {','.join(_software_to_load)}`"
        )

    return _additional_worker_init_script


def determine_additional_worker_init_conda(
    environment_name: str,
    productions: Sequence[production.ProductionSettings] | None = None,
    tasks_to_run: Sequence[Sequence[str]] | None = None,
    tasks_requiring_root: Sequence[str] | None = None,
    tasks_requiring_aliphysics: Sequence[str] | None = None,
    tasks_requiring_roounfold: Sequence[str] | None = None,
) -> str:
    # Validation
    tasks_to_run = _validate_productions_and_tasks_to_run(productions=productions, tasks_to_run=tasks_to_run)
    if tasks_requiring_root is None:
        tasks_requiring_root = []
    if tasks_requiring_aliphysics is None:
        tasks_requiring_aliphysics = []
    if tasks_requiring_roounfold is None:
        tasks_requiring_roounfold = []

    _software_to_load = []
    _additional_worker_init_script = ""
    _load_conda_env = False
    # NOTE: Each if statement here is required to write the full set of commands required to load the piece of software.
    # fmt: off
    if any(
        task in tasks_requiring_root
        for tasks in tasks_to_run
        for task in tasks
    ):
        # Nothing to be done here - just note that it needs to be done
        _load_conda_env = True
    if any(
        task in tasks_requiring_aliphysics
        for tasks in tasks_to_run
        for task in tasks
    ):
        # Additional validation
        # We probably won't support AliPhysics, so tell the user
        _msg = "AliPhysics isn't currently supported via conda, sorry!"
        raise ValueError(_msg)
    if any(
        task in tasks_requiring_roounfold
        for tasks in tasks_to_run
        for task in tasks
    ):
        _load_conda_env = True
        # Assume that we're using a local install of mammoth. As of Nov 2022, I don't foresee any case
        # where I won't do this so I think this is a reasonable assumption.
        _here = Path(__file__).parent
        _software_to_load.append(f"source {_here.parent.parent.parent / 'external' / 'setup_roounfold.sh'}")
    # fmt: on

    # If there is anything to load, add the initialization
    if _load_conda_env or _software_to_load:
        # Load conda env via our software list. It needs to go first, since we depend on it for
        # loading the rest of the software
        _software_to_load.insert(0, f"mamba activate {environment_name}")
        _additional_worker_init_script = "; ".join(_software_to_load)

    return _additional_worker_init_script


def determine_additional_worker_init(
    productions: Sequence[production.ProductionSettings] | None = None,
    tasks_to_run: Sequence[Sequence[str]] | None = None,
    tasks_requiring_root: Sequence[str] | None = None,
    tasks_requiring_aliphysics: Sequence[str] | None = None,
    tasks_requiring_roounfold: Sequence[str] | None = None,
    conda_environment_name: str | None = None,
) -> str:
    """Wrapper for convenience"""
    if conda_environment_name:
        return determine_additional_worker_init_conda(
            environment_name=conda_environment_name,
            productions=productions,
            tasks_to_run=tasks_to_run,
            tasks_requiring_root=tasks_requiring_root,
            tasks_requiring_aliphysics=tasks_requiring_aliphysics,
            tasks_requiring_roounfold=tasks_requiring_roounfold,
        )
    else:  # noqa: RET505
        return determine_additional_worker_init_alibuild(
            productions=productions,
            tasks_to_run=tasks_to_run,
            tasks_requiring_root=tasks_requiring_root,
            tasks_requiring_aliphysics=tasks_requiring_aliphysics,
            tasks_requiring_roounfold=tasks_requiring_roounfold,
        )
