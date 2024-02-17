""" Extract scale factors from all repaired files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import uproot
from pachyderm import binned_data, yaml

from mammoth import helpers
from mammoth.framework import utils
from mammoth.framework.analysis import objects as analysis_objects

logger = logging.getLogger(__name__)


def scale_factor_ROOT_wrapper(base_path: Path, train_number: int) -> tuple[int, int, Any, Any]:
    # Setup
    filenames = utils.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_ROOT(filenames)


def scale_factor_ROOT(filenames: Sequence[Path], list_name: str = "") -> tuple[int, int, Any, Any]:
    """Calculate the scale factor for a given train.

    Args:
        filenames: Filenames for output from a given train.
        list_name: Name of the list from which we will retrieve the hists.
    Returns:
        n_accepted_events, n_entries, cross_section, n_trials
    """
    # Validation
    if not list_name:
        list_name = "*DynamicalGrooming*"
    # Delay import to avoid direct dependence
    from mammoth.framework import root_utils

    ROOT = root_utils.import_ROOT()

    cross_section_hists = []
    n_trials_hists = []
    n_entries = 0
    n_accepted_events = 0
    for filename in filenames:
        f = ROOT.TFile(str(filename), "READ")

        # Retrieve the output list. First try the embedding hists. If not available, then
        # try to find the task output.
        hists = f.Get("AliAnalysisTaskEmcalEmbeddingHelper_histos")
        # If the embedding helper isn't available, let's try to get the task output.
        if not hists:
            # If the embedding helper isn't available, try to move on to an analysis task.
            # We will search through the keys using a glob.
            task_hists_name = fnmatch.filter([k.GetName() for k in f.GetListOfKeys()], list_name)
            # And then require that "Tree" is not in the name (otherwise we're likely to put up the TTree)
            task_hists_name = [name for name in task_hists_name if "Tree" not in name and "tree" not in name]
            if len(task_hists_name) != 1:
                _msg = f"Cannot find unique task name. Names: {task_hists_name}. Skipping!"
                raise RuntimeError(_msg)
            else:  # noqa: RET506
                hists = f.Get(task_hists_name[0])
                if not hists:
                    _msg = (
                        f"Cannot find a task output list. Tried: {task_hists_name[0]}. Keys: {list(f.GetListOfKeys())}"
                    )
                    raise RuntimeError(_msg)

        # This list is usually an AliEmcalList. Although we don't care about any of the AliEmcalList functionality
        # here, this still requires an AliPhysics installation, which may not always be so convenient.
        # Since all we want is the TList base class, we can get ROOT to cast it into a TList.
        # NOTE: Apparently I can't do a standard isinstance check, so this will have to do...
        if "AliEmcalList" in str(type(hists)):
            # This is basically a c++ cast
            hists = ROOT.bind_object(ROOT.addressof(hists), "TList")

        cross_section_hists.append(hists.FindObject("fHistXsection"))
        cross_section_hists[-1].SetDirectory(0)
        n_entries += cross_section_hists[-1].GetEntries()
        n_trials_hists.append(hists.FindObject("fHistTrials"))
        n_trials_hists[-1].SetDirectory(0)

        # Keep track of accepted events for normalizing the scale factors later.
        n_events_hist = hists.FindObject("fHistEventCount")
        n_accepted_events += n_events_hist.GetBinContent(1)

        f.Close()

    cross_section = cross_section_hists[0]
    # Add the rest...
    [cross_section.Add(other) for other in cross_section_hists[1:]]
    n_trials = n_trials_hists[0]
    # Add the rest...
    [n_trials.Add(other) for other in n_trials_hists[1:]]

    return n_accepted_events, n_entries, cross_section, n_trials


def scale_factor_uproot_wrapper(base_path: Path, train_number: int) -> tuple[int, int, Any, Any]:
    # Setup
    filenames = utils.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_uproot(filenames=filenames)


def _find_list_with_hists_via_uproot(f: Any, list_name: str) -> list[Any]:
    """Retrieve the list which contains the hists that we're interested in."""
    # Need to determine the list which contains the histograms of interest.
    # First, try to retrieve the embedding helper to extract the cross section and ntrials.
    hists = f.get("AliAnalysisTaskEmcalEmbeddingHelper_histos", None)
    if not hists:
        # If not the embedding helper, look for the analysis task output.
        logger.debug(f"Searching for task hists with the name pattern '{list_name}'")
        # Search for keys which contain the provided tree name. Very nicely, uproot already has this built-in
        _possible_task_hists_names = f.keys(
            cycle=False, filter_name=list_name, filter_classname=["AliEmcalList", "TList"]
        )
        if len(_possible_task_hists_names) != 1:
            _msg = (
                f"Ambiguous list name '{list_name}'. Please revise it as needed. Options: {_possible_task_hists_names}"
            )
            raise ValueError(_msg)
        # We're good - let's keep going
        hists = f.get(_possible_task_hists_names[0], None)

    # This list is usually an AliEmcalList, but we don't care about any of the AliEmcalList functionality
    # (and uproot doesn't know about it anyway), so we extract the TList via the base class.
    # NOTE: We assume the TList is the first (and only) base class. As of Feb 2023, this seems to be
    #       a reasonable assumption.
    if not isinstance(hists, uproot.models.TList.Model_TList):
        # Grab the underlying TList rather than the AliEmcalList...
        hists = hists.bases[0]

    return hists  # type: ignore[no-any-return]


def scale_factor_uproot(filenames: Sequence[Path], list_name: str = "") -> tuple[int, int, Any, Any]:
    # Validation
    if not list_name:
        list_name = "*TrackSkim*"

    cross_section_hists = []
    n_trials_hists = []
    n_entries_list = []
    n_accepted_events = []
    for filename in filenames:
        with uproot.open(filename) as f:
            # Retrieve hists of interest
            hists = _find_list_with_hists_via_uproot(f=f, list_name=list_name)

            # Now, onto the information that we're actually interested in!
            # We'll use the `hist` package for simplicity.
            cross_section_hist = [  # noqa: RUF015
                h.to_hist() for h in hists if h.name == "fHistXsection"
            ][0]
            # Number of entries in the cross section
            n_entries_list.append(cross_section_hist.counts())
            # And store the cross section and n_trials hists
            cross_section_hists.append(cross_section_hist)
            n_trials_hists.append(
                [h.to_hist() for h in hists if h.name == "fHistTrials"][0]  # noqa: RUF015
            )

            # Don't forget to keep track of accepted events for potentially relatively
            # normalizing the scale factors later.
            n_events_hist = binned_data.BinnedData.from_existing_data(
                [h for h in hists if h.name == "fHistEventCount"][0]  # noqa: RUF015
            )
            n_accepted_events.append(n_events_hist.values[0])

    # We need to find the first non-empty bin to get the number of entries
    # from the histogram values. There should only be one non-empty bin if
    # we've correctly called this with files from only one pt hat bin.
    # To help with this, we convert to the list of arrays to a single numpy
    # array for help in finding this bin.
    n_entries = np.array(n_entries_list)
    # Next, sum up all of the counts from the different hists into one array
    # NOTE: We will get nan if we have no entries in empty bins or in an empty file.
    #       The empty file means that we can get nan even in our bin of interest for
    #       a particular file. This values seem to be introduced by the
    #       TProfile -> hist conversion for counts, and appear to be a convention choice
    #       by hist. However, this won't cause any issues as long we use `nansum`.
    n_entries = np.nansum(n_entries, axis=0)

    return (
        sum(n_accepted_events),
        # Take the first non-zero value of n_entries (there should only be 1 since we
        # process them as separate pt hat bins by convention!)
        n_entries[(n_entries > 0).argmax(axis=0)],
        sum(cross_section_hists),
        sum(n_trials_hists),
    )


def create_scale_factor_tree_for_cross_check_task_output(
    filename: Path,
    scale_factor: float,
) -> bool:
    """Create scale factor for a single embedded output for the cross check task.

    As of May 2021, this is deprecated, but we keep it around as an example
    """
    # Get number of entries in the tree to determine
    with uproot.open(filename) as f:
        # This should usually get us the tree name, regardless of what task actually generated it.
        # NOTE: Adding a suffix will yield "Raw{grooming_method}Tree", so instead we search for "tree"
        #       and one of the task names.
        tree_name = [k for k in f if "RawTree" in k and ("HardestKt" in k or "DynamicalGrooming" in k)][0]  # noqa: RUF015
        n_entries = f[tree_name].num_entries
        logger.debug(f"n entries: {n_entries}")

    # We want the scale_factor directory to be in the main train directory.
    base_dir = filename.parent
    if base_dir.name == "skim":
        # If we're in the skim dir, we need to move up one more level.
        base_dir = base_dir.parent
    output_filename = base_dir / "scale_factor" / filename.name
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing scale_factor to {output_filename}")
    with uproot.recreate(output_filename) as output_file:
        # Write all of the calculations
        output_file["tree"] = {"scale_factor": np.full(n_entries, scale_factor, dtype=np.float32)}

    return True


def pt_hat_spectra_from_hists(
    filenames: Mapping[int, Sequence[Path]],
    scale_factors: Mapping[int, float],
    output_filename: Path,
    list_name: str = "",
) -> bool:
    """Extract and save pt hard spectra from embedding or pythia.

    This functionality is exceptional because we only have the histograms, not the tree.

    Note:
        I write to yaml using binned_data because I'm not sure errors, etc would be handled properly
        when writing the hist with uproot3.

    Args:
        filenames: Filenames as a function of pt hard bin.
        scale_factors: Pt hard scale factors.
        output_filename: Where the spectra should be saved (in yaml).
    Returns:
        True if successful.
    """
    # Validation
    if not list_name:
        list_name = "*DynamicalGrooming*"

    pt_hard_spectra = []
    for pt_hard_bin, pt_hard_filenames in filenames.items():
        single_bin_pt_hard_spectra = []
        for filename in pt_hard_filenames:
            with uproot.open(filename) as f:
                # Retrieve hists of interest
                hists = _find_list_with_hists_via_uproot(f=f, list_name=list_name)
                single_bin_pt_hard_spectra.append(
                    binned_data.BinnedData.from_existing_data(
                        [h for h in hists if h.has_member("fName") and h.member("fName") == "fHistPtHard"][0]  # noqa: RUF015
                    )
                )
        h_temp = sum(single_bin_pt_hard_spectra)
        # The scale factor may not be defined if (for example) working with a test production without all pt hard bins
        # If it's not available, don't bother trying to append the spectra - it doesn't gain anything, and it's likely
        # to cause additional issues.
        _scale_factor = scale_factors.get(pt_hard_bin)
        if _scale_factor:
            pt_hard_spectra.append(h_temp * _scale_factor)

    final_spectra = sum(pt_hard_spectra)

    output_filename.parent.mkdir(exist_ok=True, parents=True)
    y = yaml.yaml(modules_to_register=[binned_data])
    with output_filename.open("w") as f_out:
        y.dump([final_spectra, dict(enumerate(pt_hard_spectra, start=1))], f_out)

    return True


def scale_factor_from_hists(
    n_accepted_events: int, n_entries: int, cross_section: Any, n_trials: Any
) -> analysis_objects.ScaleFactor:
    scale_factor = analysis_objects.ScaleFactor.from_hists(
        n_accepted_events=n_accepted_events,
        cross_section=cross_section,
        n_trials=n_trials,
        n_entries=n_entries,
    )

    return scale_factor  # noqa: RET504


def are_scale_factors_close(
    scale_factors_one: Mapping[int, analysis_objects.ScaleFactor],
    scale_factors_two: Mapping[int, analysis_objects.ScaleFactor],
) -> bool:
    """Convenience function for comparing scale factors.

    Useful for comparing eg. values calculated separately with ROOT and uproot. There's almost certainly
    a better way to do this test for nested dicts with objects, but this was very convenient and easy.
    """
    # Ensure that we've actually compared something (ie. avoid trivially saying they agree because they're empty)
    if len(list(scale_factors_one)) == 0:
        return False

    # Make sure we have the same pt hat bins
    if list(scale_factors_one) != list(scale_factors_two):
        return False

    # Now check each value, returning False if they fail
    for pt_hat_bin_1 in scale_factors_one:
        if not np.isclose(
            scale_factors_one[pt_hat_bin_1].cross_section,
            scale_factors_two[pt_hat_bin_1].cross_section,
        ):
            return False
        if not np.isclose(
            scale_factors_one[pt_hat_bin_1].n_trials_total,
            scale_factors_two[pt_hat_bin_1].n_trials_total,
        ):
            return False
        if not np.isclose(
            scale_factors_one[pt_hat_bin_1].n_entries,
            scale_factors_two[pt_hat_bin_1].n_entries,
        ):
            return False
        if not np.isclose(
            scale_factors_one[pt_hat_bin_1].n_accepted_events,
            scale_factors_two[pt_hat_bin_1].n_accepted_events,
        ):
            return False

    # Looks good!
    return True


def test() -> None:
    scale_factors_ROOT = {}
    scale_factors_uproot = {}

    base_path = Path("trains/pythia/2619")

    # for pt_hat_bin in [12, 13]:
    # NOTE: Going past bin 15 or so for ROOT will cause it to be force closed for using too much memory on macOS
    #       as of Feb 2023. It's unclear why this is possibility happening, but seems to be a ROOT bug with 6.24.06 .
    for pt_hat_bin in range(10, 21):
        logger.info(f"Processing {pt_hat_bin=}")
        input_files = list(base_path.glob(f"run_by_run/LHC18b8_*/*/{pt_hat_bin}/AnalysisResults.*.root"))
        # To save time and memory
        # input_files = input_files[:20]
        # print(f"{input_files=}")
        scale_factors_ROOT[pt_hat_bin] = scale_factor_from_hists(
            *scale_factor_ROOT(filenames=input_files, list_name="*TrackSkim*")
        )
        scale_factors_uproot[pt_hat_bin] = scale_factor_from_hists(
            *scale_factor_uproot(filenames=input_files, list_name="*TrackSkim*")
        )
        # res_ROOT = scale_factor_ROOT(base_path=base_path, train_number=train_number)
        # res_uproot = scale_factor_uproot(base_path=base_path, train_number=train_number)

    # y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    # with open("test.yaml", "w") as f:
    #    y.dump(scale_factors_ROOT, f)

    logger.info(f"scale_factors_ROOT: {scale_factors_ROOT}")
    logger.info(f"scale_factors_uproot: {scale_factors_uproot}")
    logger.info(f"Equal? {are_scale_factors_close(scale_factors_ROOT, scale_factors_uproot)}")
    import IPython

    IPython.start_ipython(user_ns=locals())  # type: ignore[no-untyped-call]


def run() -> None:
    scale_factors_ROOT = {}
    # scale_factors_uproot = {}

    base_path = Path("trains/pythia/2619")

    for pt_hat_bin in range(1, 21):
        input_files = list(base_path.glob(f"run_by_run/LHC18b8_*/*/{pt_hat_bin}/AnalysisResults.*.root"))
        # To save time and memory
        # input_files = input_files[:20]
        logger.info(f"{input_files=}")
        scale_factors_ROOT[pt_hat_bin] = scale_factor_from_hists(
            *scale_factor_ROOT(filenames=input_files, list_name="*TrackSkim*")
        )
        # scale_factors_uproot[pt_hat_bin] = scale_factor_from_hists(
        #    *scale_factor_uproot(filenames=input_files, run_despite_issues=True)
        # )
        # res_ROOT = scale_factor_ROOT(base_path=base_path, train_number=train_number)
        # res_uproot = scale_factor_uproot(base_path=base_path, train_number=train_number)

    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    with Path("trains/pythia/LHC18b8_AOD_2619/scale_factors_ROOT.yaml").open("w") as f:
        y.dump(scale_factors_ROOT, f)
        # y.dump(scale_factors_uproot, f)


if __name__ == "__main__":
    helpers.setup_logging()
    test()
    # run()
