"""Output utilities

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

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
                        return (True, "already processed (confirmed)")
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


def write_hists_to_file(hists: Mapping[Any, Any], f: BinaryIO, prefix: str = "", separator: str = "_") -> bool:
    """Recursively write histograms to a given file.

    NOTE:
        The default separator will flatten the histogram names. If you want to preserve the
        dict structure, you should use "/", which will prompt uproot to maintain the structure.

    Args:
        hists: Hists to be written.
        f: File to be written to. Usually a file opened with uproot.
        prefix: Prefix to append to all keys. Default: "". This is rarely set by the user
            directly. Instead, it's used when recursing to keep track of the path.
        separator: Separator to use between keys. Default: "_", which implicitly flattens the
            dict structure. If you want to preserve the dict structure, use "/",

    Returns:
        True if writing was successful.
    """
    for k, v in hists.items():
        if isinstance(v, dict):
            write_hists_to_file(hists=v, f=f, prefix=f"{prefix}{separator}{k}", separator=separator)
        else:
            write_name = str(k) if not prefix else f"{prefix}{separator}{k}"
            f[write_name] = v  # type: ignore[index]

    return True


# Explicitly list what is supported. To support trees, we would need to handle
# extending, etc, which is less trivial, so just do this for now.
_shadd_supported_types = (
    uproot.behaviors.TH1.Histogram,
    # NOTE: The model only works here because there's apparently only one version of the TList streamer.
    #       There's no equivalent of the TList behavior, so we take this as good enough.
    # NOTE: We don't use Sequence (which TList also inherits from) because it is too loose - we need something with names.
    uproot.models.TList.Model_TList,
)

# Apparently adding profile hists is not supported by hist, so we need to exclude them.
_shadd_unsupported_types = (uproot.behaviors.TProfile.Profile,)


def _format_indent(level: int) -> str:
    """Helper to figure out nice indentation for logging."""
    if level == 0:
        return ""
    if level == 1:
        return "> "
    return "-" * (level - 1) + "> "


def _filter_for_supported_merging_types(contents: dict[str, Any], level: int = 0) -> dict[str, Any]:
    """Filter out types which we can't merge

    As of June 2023, this practically means that only hists are supported.

    Args:
        contents: Contents to be filtered.
        level: Recursion level. Used for logging. Default: 0.
            In practice, the user won't touch this.

    Returns:
        Filtered contents.
    """
    output = {}
    for k, v in contents.items():
        logger.debug(f"Processing {k}")
        # We can only support some types.
        # We also have an explicitly unsupported list.
        if (not isinstance(v, _shadd_supported_types)) or isinstance(v, _shadd_unsupported_types):
            # If there's one base and it's a TList, we can handle it. This is still lossy, but we may not care.
            # One prominent example: AliEmcalList, where the additional fields are irrelevant for our purposes.
            if (
                hasattr(v, "bases")
                and len(v.bases) == 1
                and isinstance(v.bases[0], _shadd_supported_types)
                # NOTE: If it's unsupported, don't try here even if everything else fits. We want to disable this
                #       because it might give unexpected answers (eg. since profiles base are histograms, this would
                #       retrieve it as a histogram even after we labeled it as explicitly unsupported. This isn't
                #       great since it will be seen as a regular hist and give wrong answers...
                and not isinstance(v, _shadd_unsupported_types)
            ):
                msg = (
                    _format_indent(level)
                    + f"Narrowing type of {k} from {v!r} to {v.bases[0]!r} since we don't have the relevant streamers available."
                )
                logger.warning(msg)
                v = v.bases[0]  # noqa: PLW2901
            else:
                msg = _format_indent(level) + f"{k}: Skipping unsupported type {v!r}"
                logger.error(msg)
                continue

        # Recurse
        # NOTE: Since we only support TNamed, we can treat the TList as a mapping
        if isinstance(v, uproot.models.TList.Model_TList):
            msg = _format_indent(level) + f"Recursing for {k}"
            logger.info(msg)
            output[k] = _filter_for_supported_merging_types(
                contents={entry.name: entry for entry in v}, level=level + 2
            )
        else:
            try:
                # We're ready to proceed, so we convert to a hist object which supports __add__
                output[k] = v.to_hist() if isinstance(v, uproot.behaviors.TH1.Histogram) else v
                logger.debug(f"Successfully wrote {k}")
            except ValueError as e:
                logger.warning(f"Skipping {k} since it can't be converted to a histogram: {e}")
                continue

    return output


def shit_hadd() -> None:
    """Shit (eg. "simple") version of histogram add (`hadd`) to avoid needing root.

    Args:
        output_filename: Output filename to be written.
        input_filenames: List of input filenames to be added together.

    Returns:
        Path to the output file.
    """
    # Delayed import since this is self contained
    import argparse

    import mammoth.helpers

    # Setup
    mammoth.helpers.setup_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="shadd: Shi^H^H^HSimple hadd replacement", formatter_class=mammoth.helpers.RichHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True, nargs="+", type=Path, help="Input filename(s)")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output filename")

    args = parser.parse_args()
    output_filename: Path = args.output

    if output_filename.exists():
        msg = f"Output already exists! {args.output}"
        raise ValueError(msg)

    with uproot.recreate(output_filename) as f_out:
        hists: dict[str, Any] = {}
        with mammoth.helpers.progress_bar() as progress:
            track_results = progress.add_task(total=len(args.input), description="Processing inputs...")
            for input_filename in args.input:
                logger.info(f"Processing {input_filename}")
                # with uproot.open(input_filename, custom_classes={"AliEmcalList": uproot.models.TList.Model_TList}) as f_in:
                with uproot.open(input_filename) as f_in:
                    hists = merge_results(
                        hists,
                        _filter_for_supported_merging_types(
                            # NOTE: We do these minor gymnastics so we can avoid having to remove the cycle (eg. ";1") by hand.
                            #       It's not that hard, but no point in reinventing the wheel.
                            contents={k: f_in[k] for k in f_in.keys(cycle=False)}
                        ),
                    )
                progress.update(track_results, advance=1)

        # NOTE: This is a little perverse, but even if we have a nested dict, uproot won't write in a TDirectory unless there
        #      is a "/" in the name. So we just add one here as the separator.
        write_hists_to_file(hists=hists, f=f_out, separator="/")

    logger.info(f'🎉 Finished merging "{output_filename}"')
