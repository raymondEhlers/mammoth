"""Utilities for using additional HEP software, such as RooUnfold

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from typing import List, Optional, Sequence
from pathlib import Path

import attr

from mammoth.framework import production


@attr.define
class Dependency:
    name: str
    tasks: List[str]
    depends_on_ROOT: bool


def determine_additional_worker_init_alibuild(
    productions: Sequence[production.ProductionSettings],
    tasks_requiring_root: Optional[Sequence[str]] = None,
    tasks_requiring_aliphysics: Optional[Sequence[str]] = None,
    tasks_requiring_roounfold: Optional[Sequence[str]] = None,
    software_version_tag: str = "latest",
) -> str:
    # Validation
    if tasks_requiring_root is None:
        tasks_requiring_root = []
    if tasks_requiring_aliphysics is None:
        tasks_requiring_aliphysics = []
    if tasks_requiring_roounfold is None:
        tasks_requiring_roounfold = []

    _software_to_load = []
    _additional_worker_init_script = ""
    _software_options = {
        # NOTE: It's important that ROOT is handled first so that we can remove it later if we need to do so. See below.
        Dependency("ROOT", tasks_requiring_root, False),
        Dependency("AliPhysics", tasks_requiring_aliphysics, True),
        Dependency("RooUnfold", tasks_requiring_roounfold, True),
    }
    for _software in _software_options:
        # fmt: off
        if any(
            (
                task in tasks_requiring_root
                for prod in productions
                for task in prod.tasks_to_execute
            )
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
    productions: Sequence[production.ProductionSettings],
    environment_name: str,
    tasks_requiring_root: Optional[Sequence[str]] = None,
    tasks_requiring_aliphysics: Optional[Sequence[str]] = None,
    tasks_requiring_roounfold: Optional[Sequence[str]] = None,
) -> str:
    # Validation
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
        (
            task in tasks_requiring_root
            for prod in productions
            for task in prod.tasks_to_execute
        )
    ):
        # Nothing to be done here - just note that it needs to be done
        _load_conda_env = True
    if any(
        (
            task in tasks_requiring_aliphysics
            for prod in productions
            for task in prod.tasks_to_execute
        )
    ):
        # Additional validation
        # We probably won't support AliPhysics, so tell the user
        raise ValueError("AliPhysics isn't currently supported via conda, sorry!")
    if any(
        (
            task in tasks_requiring_roounfold
            for prod in productions
            for task in prod.tasks_to_execute
        )
    ):
        _load_conda_env = True
        # Assume that we're using a local install of mammoth. As of Nov 2022, I don't forsee any case
        # where I won't do this so I think this is a reasonable assumption.
        _here = Path(__file__).parent
        _software_to_load.append(f"source {_here.parent.parent / 'external' / 'setup_roounfold.sh'}")
    # fmt: on

    # If there is anything to load, add the initialization
    if _load_conda_env or _software_to_load:
        _additional_worker_init_script = (
            f"mamba activate {environment_name}" + "; ".join(_software_to_load)
        )

    return _additional_worker_init_script
