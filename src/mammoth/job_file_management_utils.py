"""Job file management related utilities

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import uproot

import mammoth.helpers
from mammoth import job_file_management

logger = logging.getLogger(__name__)


def check_if_file_is_valid(p: Path, settings: job_file_management.FileStagingSettings) -> bool:
    """Check if a file is valid.

    It must:
    - Exist on the node
    - Not exist at the permanent path
    - And the contents must be correct in the case:
        - ".empty" file
        - parquet file
        - root file with entries

    Args:
        p: The file to check.
    Returns:
        True if the file is valid, False otherwise.
    """
    logger.debug(f"Checking {p}...")
    # Validation. This shouldn't be an issue here, but best to check
    if not p.exists():
        logger.debug("Node file doesn't exist. Wat?")
        return False

    permanent_path = settings.translate_output_node_to_permanent_path(p)
    if permanent_path.exists():
        logger.warning("Permanent path already exists")
        return False

    match p.suffix:
        case ".empty":
            # If empty as defined by the extension, it is by definition valid
            logger.debug("Empty extension")
            return True
        case ".parquet":
            # Check if file can be opened.
            try:
                # We don't want to get into what's provided - just that it's something.
                _ = ak.from_parquet(p)
                logger.debug("Successful read by parquet")
                return True
            except Exception:
                logger.debug("Failed reading by parquet")
                return False
        case ".root":
            # Check if the file can be opened.
            try:
                with uproot.open(p) as f:
                    # Take whatever keys are available
                    keys = f.keys()
                    logger.debug(f"Read by uproot with keys: {list(keys)}")

                    # If there are no entries, then there may be an issue with the file
                    if len(list(keys)) == 0:
                        return False

                    # Finally, if there are keys, check for a non-zero number of entries
                    num_entries: int = f[next(iter(keys))].num_entries
                    has_entries = num_entries > 0
                    logger.debug(f"Read array by uproot with {num_entries=}")
                    return has_entries
            except Exception:
                logger.debug("Failed reading by uproot.")
                return False
        case _:
            logger.warning("Unrecognized output file")
            return False


def steer_handle_files_from_failed_jobs(node_work_dir: Path, permanent_work_dir: Path) -> None:
    """Steer the handling of files from failed jobs.

    For example, a job may fail due to running out of memory or a job manager closing,
    but that doesn't mean that all of the chunks failed. Some valid outputs may still
    be sitting there. So we:
    1. Check if they are valid by opening them.
    2. If they are, we move them to the permanent directory.
    3. And then clean up the work directory.

    Args:
        node_work_dir: The work directory on the node where the files are stored.
        permanent_work_dir: The permanent work directory where the files should be stored.
            This is the base path for the permanent directory.
    Returns:
        None
    """
    # Setup
    invalid_output_files_per_work_dir = {}

    # Find the work directories.
    work_directories_on_node = [v for v in node_work_dir.glob("*") if v.is_dir()]

    with mammoth.helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(work_directories_on_node), description="Processing work_dir...")
        for work_dir in work_directories_on_node:
            logger.info(f'Processing work_dir="{work_dir.name}"')
            # Setup
            settings = job_file_management.FileStagingSettings(
                permanent_work_dir=permanent_work_dir,
                node_work_dir=node_work_dir,
                unique_id=work_dir.name,
            )
            # NOTE: We don't want to create it via `from_settings` because we want it to use
            #       the existing FileStagingSettings object that we just created. That ensures
            #       that the UUID is correct.
            manager = job_file_management.FileStagingManager(
                file_stager=job_file_management.FileStager(settings=settings), input_files=[]
            )
            # Help out mypy...
            assert manager.file_stager is not None
            input_files = sorted(
                [v for v in manager.file_stager.settings.node_work_dir_input.rglob("*") if v.is_file()]
            )
            output_files = sorted(
                [v for v in manager.file_stager.settings.node_work_dir_output.rglob("*") if v.is_file()]
            )
            # logger.info(f"{input_files=}")
            # logger.info(f"{output_files=}")

            # Clean up the input files
            # NOTE: Yes, I'm breaking the API. I don't really want to expose this in most
            #       cases, but it makes sense here.
            manager._clean_up_staged_in_files_on_node(files_on_node_to_clean_up=input_files)

            # Check that the output files are valid
            valid_output_files = [v for v in output_files if check_if_file_is_valid(v, settings=settings)]
            logger.debug(f"Valid output_files: {output_files=}")
            invalid_output_files_per_work_dir[work_dir] = set(output_files) - set(valid_output_files)
            # Warn if there are output files that do not appear to be valid.
            if len(invalid_output_files_per_work_dir[work_dir]) > 0:
                msg = (
                    f"There are output files that are invalid to be transferred to permanent storage!"
                    f"\nFiles: {invalid_output_files_per_work_dir[work_dir]}"
                )
                logger.warning(msg)

            # Transfer the output files
            manager.file_stager.stage_files_out(files_to_stage_out=valid_output_files)
            # And then clean up the work dir
            manager.file_stager.settings.node_work_dir.rmdir()
            # And update the process
            progress.update(track_results, advance=1)

    # Print for the user to see the problem files:
    if any(v for v in invalid_output_files_per_work_dir.values()):
        logger.warning("Found invalid output files - check on these!")
        for work_dir, invalid_files in invalid_output_files_per_work_dir.items():
            logger.warning(f"{work_dir}: {invalid_files}")
    else:
        logger.info("Successfully moved all files!")


def steer_handle_files_from_failed_jobs_entry_point() -> None:
    """Entry point for handling files from failed jobs.

    Args:
        None. This is the entry point
    Returns:
        None
    """
    # Delayed import since this is self contained
    import argparse

    # Setup
    parser = argparse.ArgumentParser(
        description="Sync files from filed jobs.", formatter_class=mammoth.helpers.RichHelpFormatter
    )

    parser.add_argument("-n", "--node-work-dir", required=True, type=Path, help="Node work directory")
    parser.add_argument("-p", "--permanent-work-dir", required=True, type=Path, help="Permanent work directory")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug level messages")

    args = parser.parse_args()
    node_work_dir: Path = args.node_work_dir
    permanent_work_dir: Path = args.permanent_work_dir
    debug: bool = args.debug
    logging_level = logging.INFO
    if debug:
        logging_level = logging.DEBUG
    mammoth.helpers.setup_logging(level=logging_level)
    if debug:
        # Quiet down fsspec
        logging.getLogger("fsspec").setLevel(logging.INFO)

    steer_handle_files_from_failed_jobs(
        node_work_dir=node_work_dir,
        permanent_work_dir=permanent_work_dir,
    )


if __name__ == "__main__":
    steer_handle_files_from_failed_jobs_entry_point()
