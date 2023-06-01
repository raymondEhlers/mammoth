""" Output utilities

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import awkward as ak
import uproot

logger = logging.getLogger(__name__)


def task_output_path_hist(output_filename: Path) -> Path:
    """Get the path of the hist output file for a task."""
    return output_filename.parent / "hists" / output_filename.with_suffix(".root").name


def task_output_path_skim(output_filename: Path, skim_name: str) -> Path:
    """Get the path of the skim output file for a task."""
    return output_filename.parent / skim_name / output_filename.name


def check_for_task_root_skim_output_file(output_filename: Path, reference_tree_name: str = "tree") -> tuple[bool, str]:
    """Check if the ROOT skim output file for a task has already been processed.

    Note:
        We also check for an empty file, which indicates that we've run the analysis, but there's nothing
        to output.

    Args:
        output_filename: Output filename to be checked.
        reference_array_name: Name of the reference array to check for. If it exists and has
            entries, then we known the file has been processed. Default: "tree".
    Returns:
        Tuple of (bool, str) indicating if the file has been processed and a message indicating
            what has been found.
    """
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no tree outputs after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_tree_name:
            try:
                with uproot.open(output_filename) as f:
                    # If the tree exists, can be read, and has more than 0 entries, we should be good
                    if f[reference_tree_name].num_entries > 0:
                        # Return immediately to indicate that we're done.
                        return (True, f"already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try running the analysis
                pass
        else:
            return (True, "already processed (no reference tree name provided, but file exists)")

    return (False, "")


def check_for_task_parquet_skim_output_file(output_filename: Path, reference_array_name: str = "") -> tuple[bool, str]:
    """Check if the parquet skim output file for a task has already been processed.

    Note:
        We also check for an empty file, which indicates that we've run the analysis, but there's nothing
        to output.

    Args:
        output_filename: Output filename to be checked.
        reference_array_name: Name of the reference array to check for. If it exists and has
            entries, then we known the file has been processed. Default: "", which means that
            we will just check for the existence of the file.
    Returns:
        Tuple of (bool, str) indicating if the file has been processed and a message indicating
            what has been found.
    """
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no array outputs after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_array_name:
            try:
                arrays = ak.from_parquet(output_filename)
                # If the reference array exists, can be read, and has more than 0 entries, we have analyzed successfully
                # and don't need to do it again
                if ak.num(arrays[reference_array_name], axis=0) > 0:
                    # Return immediately to indicate that we're done.
                    return (True, "already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try running the analysis
                pass
        else:
            return (True, "already processed (no reference array name provided, but file exists)")

    return (False, "")


def check_for_task_hist_output_file(output_filename: Path, reference_hist_name: str = "") -> tuple[bool, str]:
    """Check if the hist output file for a task has already been processed.

    Note:
        We also check for an empty file, which indicates that we've run the analysis, but there's nothing
        to output.

    Args:
        output_filename: Output filename to be checked.
        reference_hist_name: Name of the reference histogram to check for. If it exists and has
            entries, then we known the file has been processed. Default: "", which means that
            we will just check for the existence of the file.
    Returns:
        Tuple of (bool, str) indicating if the file has been processed and a message indicating
            what has been found.
    """
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no hists after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_hist_name:
            try:
                with uproot.open(output_filename) as f:
                    # If the hist exists, can be read, and has more than 0 entries in any bin, we should be good
                    if ak.any(f[reference_hist_name].values() > 0):
                        # Return immediately to indicate that we're done.
                        return (True, "already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try running the analysis
                pass
        else:
            return (True, "already processed (no reference hist name provided, but file exists)")

    return (False, "")


def merge_results(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """Merge job results together.

    By convention, we merge into the first dict to try to avoid unnecessary copying.

    Although this should generically work for any object which implements `__add__`,
    it's geared towards histograms.

    Note:
        For the first result, it's often convenient to start with a variable containing an
        empty dict as the argument to a. That way, the merged results will be stored in
        a persistent variable.

    Args:
        a: Job result to be merged into.
        b: Result to be merged with.

    Returns:
        Merged histograms
    """
    # Short circuit if nothing to be done
    if not b and a:
        logger.debug("Returning a since b is None")
        return a
    if not a and b:
        logger.debug("Returning b since a is None")
        return b

    # Ensure we don't miss anything in either dict
    all_keys = set(a) | set(b)

    for k in all_keys:
        a_value = a.get(k)
        b_value = b.get(k)
        # Nothing to be done
        if a_value and b_value is None:
            logger.debug(f"b_value is None for {k}. Skipping")
            continue
        # Just take the b value and move on
        if a_value is None and b_value:
            logger.debug(f"a_value is None for {k}. Assigning")
            a[k] = b_value
            continue
        # At this point, both a_value and b_value should be not None
        assert a_value is not None
        assert b_value is not None

        # Recursive on dict
        if isinstance(a_value, dict):
            logger.debug(f"Recursing on dict for {k}")
            a[k] = merge_results(a_value, b_value)
        else:
            # Otherwise, merge
            logger.debug(f"Merging for {k}")
            a[k] = a_value + b_value

    return a


def write_hists_to_file(hists: Mapping[Any, Any], f: BinaryIO, prefix: str = "") -> bool:
    """Recursively write histograms to a given file.

    Args:
        hists: Hists to be written.
        f: File to be written to. Usually a file opened with uproot.
        prefix: Prefix to append to all keys. Default: "". This is rarely set by the user
            directly. Instead, it's used when recursing to keep track of the path.

    Returns:
        True if writing was successful.
    """
    for k, v in hists.items():
        if isinstance(v, dict):
            write_hists_to_file(hists=v, f=f, prefix=f"{prefix}_{k}")
        else:
            write_name = str(k) if not prefix else f"{prefix}_{k}"
            f[write_name] = v  # type: ignore[index]

    return True
