"""Job file management related utilities

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import uproot

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
    # Validation. This shouldn't be an issue here, but best to check
    if not p.exists():
        logger.info(f"doesn't exist: {p}")
        return False

    permanent_path = settings.translate_output_node_to_permanent_path(p)
    if permanent_path.exists():
        logger.warning(f"Permanent path already exists. {p}")
        return False

    match p.suffix:
        case ".empty":
            # If empty, it is by definition valid
            logger.info(f"Empty: {p}")
            return True
        case ".parquet":
            # Check if file can be opened.
            try:
                arrays = ak.from_parquet(p)
                # If so, cleanup and return
                del arrays
                logger.info(f"Read by parquet: {p}")
                return True
            except Exception:
                logger.info(f"Failed reading by parquet: {p}")
                return False
        case ".root":
            # Check if the file can be opened.
            #try:
            with uproot.open(p) as f:
                keys = f.keys()
                has_keys = len(list(keys)) > 0
                logger.info(f"Read by uproot. has keys: {has_keys}, keys: {list(f.keys())} {p}")
                if has_keys:
                    num_entries = f[next(iter(keys))].num_entries
                    has_entries = num_entries > 0
                    #arrays = f[next(iter(keys))].arrays()
                    logger.info(f"Read array by uproot. {has_entries=} {p}")
                    #del arrays
                    return has_entries
                return has_keys
            #except Exception:
            #    logger.info(f"Failed reading by uproot. {p}")
            #    return False
        case _:
            logger.info(f"Unrecognized output file: {p}")
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
    # Find the work directories.
    work_directories_on_node = [v for v in node_work_dir.glob("*") if v.is_dir()]

    invalid_output_files_per_work_dir = {}
    for work_dir in work_directories_on_node:
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
        input_files = sorted([
            v for v in manager.file_stager.settings.node_work_dir_input.rglob("*") if v.is_file()
        ])
        output_files = sorted([
            v for v in manager.file_stager.settings.node_work_dir_output.rglob("*") if v.is_file()
        ])
        logger.info(f"{input_files=}")
        logger.info(f"{output_files=}")

        # Clean up the input files
        # NOTE: Yes, I'm breaking the API. I don't really want to expose this in most
        #       cases, but it makes sense here.
        #manager._clean_up_staged_in_files_on_node(files_on_node_to_clean_up=input_files)

        # Check that the output files are valid
        valid_output_files = [v for v in output_files if check_if_file_is_valid(v, settings=settings)]
        logger.info(f"Valid output_files: {output_files=}")
        invalid_output_files_per_work_dir[work_dir] = set(output_files) - set(valid_output_files)
        # Warn if there are output files that do not appear to be valid.
        if len(invalid_output_files_per_work_dir[work_dir]) > 0:
            msg = (
                f"There are output files that are invalid to be transfered to permanent storage!"
                f"\nFiles: {invalid_output_files_per_work_dir[work_dir]}"
            )
            logger.warning(msg)

        ## Transfer the output files
        #manager.file_stager.stage_files_out(files_to_stage_out=valid_output_files)
        ## And then the work dir
        #manager.file_stager.settings.node_work_dir.rmdir()

        ## Find the files to move
        # files_to_move = [
        #    v for v in work_dir.glob("*") if v.is_file()
        # ]

        ## Check if the files are valid
        # valid_files = [v for v in files_to_move if check_if_file_is_valid(v)]

        ## Move the valid files
        # for valid_file in valid_files:
        #    sync_files_back_from_worker(
        #        node_work_dir=node_work_dir,
        #        permanent_work_dir=permanent_work_dir,
        #        files_to_move=valid_file,
        #    )

        ## Cleanup the work directory
        # for file_to_remove in files_to_move:
        #    file_to_remove.unlink()



def steer_handle_files_from_failed_jobs_entry_point() -> None:
    """Entry point for handling files from failed jobs.

    Args:
        None. This is the entry point
    Returns:
        None
    """
    # Delayed import since this is self contained
    import argparse

    from rich_argparse import RichHelpFormatter

    import mammoth.helpers

    # Setup
    mammoth.helpers.setup_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sync files from filed jobs.", formatter_class=RichHelpFormatter)

    parser.add_argument("-n", "--node-work-dir", required=True, type=Path, help="Node work directory")
    parser.add_argument("-p", "--permanent-work-dir", required=True, type=Path, help="Permanent work directory")

    args = parser.parse_args()
    node_work_dir: Path = args.node_work_dir
    permanent_work_dir: Path = args.permanent_work_dir

    steer_handle_files_from_failed_jobs(
        node_work_dir=node_work_dir,
        permanent_work_dir=permanent_work_dir,
    )


if __name__ == "__main__":
    steer_handle_files_from_failed_jobs_entry_point()
