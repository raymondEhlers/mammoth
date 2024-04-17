"""Utilities for working with track skims.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path

import uproot

import mammoth.helpers

logger = logging.getLogger(__name__)


def count_events_in_track_skim(
    trains_directory: Path, collision_system: str, train_number: int, pattern_for_finding_files: str | None = None
) -> tuple[int, int]:
    """Count events in a track skim.

    This is for calculating luminosity. Also as a cross check.

    Args:
        trains_directory: Path to the trains directory
        collision_system: The collision system. Here, this really just serves as
            the path to the input.
        train_number: The train number to count the events for.
        pattern_for_finding_files: Pattern for finding the files. Default: "run_by_run/*/*/*/*.root",
            which was designed for pp_MC.
    Returns:
        tuple: Number of events and number of files.
    """
    # Validation
    trains_directory = Path(trains_directory)
    pattern_for_finding_files = _validate_pattern_for_finding_files(
        pattern=pattern_for_finding_files, collision_system=collision_system
    )

    # Setup
    found_tree_name = False
    tree_name = "AliAnalysisTaskTrackSkim_*_tree"

    # Example: pythia/641/run_by_run/LHC20g4/295612/1/
    filenames = list((trains_directory / collision_system / str(train_number)).glob(pattern_for_finding_files))
    logger.info(f"Found {len(filenames)} files")
    logger.debug(filenames)
    # For testing
    # filenames = Path(str(train_number)).glob("run_by_run/*/295612/1/*")
    # print(list(filenames))

    num_entries_per_file = []
    with mammoth.helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(filenames), description="Processing filenames...")
        for filename in filenames:
            with uproot.open(filename) as f:
                if not found_tree_name:
                    # (Basically pulled this from UprootSource, but wasn't worth refactoring)
                    _possible_tree_names = f.keys(cycle=False, filter_name=tree_name, filter_classname="TTree")
                    if len(_possible_tree_names) != 1:
                        if len(_possible_tree_names) == 0:
                            _msg = f"Missing tree name '{tree_name}'. Please check the file. Filename: {filename}"
                            raise RuntimeError(_msg)
                        else:  # noqa: RET506
                            _msg = f"Ambiguous tree name '{tree_name}'. Please revise it as needed. Possible tree names: {_possible_tree_names}. Filename: {filename}"
                            raise ValueError(_msg)

                    found_tree_name = True
                    tree_name = _possible_tree_names[0]
                try:
                    num_entries_per_file.append(len(f[tree_name].arrays("run_number")["run_number"]))
                except uproot.exceptions.KeyInFileError as e:
                    # Just log and continue
                    logger.exception(e)

            # And update the process
            progress.update(track_results, advance=1)

    return sum(num_entries_per_file), len(num_entries_per_file)


def _validate_pattern_for_finding_files(pattern: str | None, collision_system: str) -> str:
    """Validate + determine the pattern for the finding files.

    Basically, if set, we'll just check and return it. If not set, we'll return the right default value.

    Args:
        pattern: Pattern to validate.
        collision_system: Collision system, used for determining the right default.

    Returns:
        Validated pattern
    """
    if pattern is None:
        if "mc" in collision_system.lower() or collision_system == "pythia":
            # Need an extra directory to account for pt hat bins
            pattern = "run_by_run/*/*/*/*.root"
        else:
            pattern = "run_by_run/*/*/*.root"
    elif "*" not in pattern:
        msg = f"We need a pattern - i.e. including a '*' - but it doesn't appear to be one. Provided: {pattern}"
        raise ValueError(msg)
    return pattern


def count_events_in_track_skim_entry_point() -> None:
    """Entry point for counting events in a track skim.

    Args:
        None. This is the entry point
    Returns:
        None
    """
    # Delayed import since this is self contained
    import argparse

    from rich_argparse import RichHelpFormatter

    # Setup
    parser = argparse.ArgumentParser(description="Sync files from filed jobs.", formatter_class=RichHelpFormatter)

    parser.add_argument(
        "-t", "--trains-directory", type=Path, default=Path("trains"), help="Base path to the trains directory"
    )
    parser.add_argument("-c", "--collision-system", required=True, type=str, help="Collision system")
    parser.add_argument("-n", "--train-number", required=True, type=int, help="Train number")
    parser.add_argument(
        "-p",
        "--pattern-for-finding-files",
        type=str,
        default=None,
        help="Pattern for finding files. Default: 'run_by_run/*/*/*/*.root' for MC, 'run_by_run/*/*/*/*.root' for data. Remember to pass the arguments in single quotes so the shell doesn't expand the *.",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug level messages")

    args = parser.parse_args()
    trains_directory: Path = args.trains_directory
    collision_system: str = args.collision_system
    train_number: int = args.train_number
    pattern_for_finding_files: str = _validate_pattern_for_finding_files(
        pattern=args.pattern_for_finding_files, collision_system=collision_system
    )
    debug: bool = args.debug
    logging_level = logging.INFO
    if debug:
        logging_level = logging.DEBUG
    mammoth.helpers.setup_logging(level=logging_level)
    if debug:
        # Quiet down fsspec
        logging.getLogger("fsspec").setLevel(logging.INFO)

    num_events, num_files = count_events_in_track_skim(
        trains_directory=trains_directory,
        collision_system=collision_system,
        train_number=train_number,
        pattern_for_finding_files=pattern_for_finding_files,
    )
    logger.info(f"num_entries: {num_events}, num_files: {num_files}")
