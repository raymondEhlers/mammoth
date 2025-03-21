"""Skim full structured splitting tree to flat tree of groomed splittings for further analysis

Applies a variety of grooming algorithms, including SoftDrop + Dynamical Grooming

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Union

import attrs
import awkward as ak
import numba as nb
import numpy as np
import numpy.typing as npt
import uproot
from pachyderm import yaml

from mammoth.framework.analysis import jet_substructure as analysis_jet_substructure
from mammoth.framework.typing import AwkwardArray, Scalar

logger = logging.getLogger(__name__)

T_GroomingResults = dict[str, Union[npt.NDArray[np.float32], npt.NDArray[np.int16]]]  # noqa: UP007


@attrs.define
class Calculation:
    """Similar to `FillHistogramInput`, but adds the splittings indices.

    Note:
        The splitting indices are the overall indices of the input splittings within
        the entire splittings array. The indices are those of the splittings selected
        by the calculation.
    """

    input_jets: ak.Array = attrs.field()
    input_splittings: analysis_jet_substructure.JetSplittingArray = attrs.field()
    input_splittings_indices: AwkwardArray[AwkwardArray[int]] = attrs.field()
    values: npt.NDArray[np.float32] = attrs.field()
    indices: AwkwardArray[int] = attrs.field()
    # If there's no additional grooming selection, then this will be identical to input_splittings_indices.
    possible_indices: AwkwardArray[AwkwardArray[int]] = attrs.field()
    # NOTE: We don't initialize here because we want to cache the calculation of the selected set of splittings
    _restricted_splittings: analysis_jet_substructure.JetSplittingArray = attrs.field(init=False)

    @property
    def splittings(self) -> analysis_jet_substructure.JetSplittingArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: analysis_jet_substructure.JetSplittingArray = self.input_splittings[
                self.indices
            ]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """Number of jets."""
        # We flatten the splittings because there may be jets (and consequently splittings) which aren't selected
        # at all due to the grooming (such as a z cut). Thus, we use the selected splittings directly.
        return len(self.splittings.flatten())

    def __getitem__(self, mask: npt.NDArray[np.bool_]) -> Calculation:
        """Mask the stored values, returning a new object."""
        # Validation
        if len(self.input_jets) != len(mask):
            _msg = f"Mask length is different than array lengths. mask length: {len(mask)}, array lengths: {len(self.input_jets)}"
            raise ValueError(_msg)

        # Return the masked arrays in a new object.
        return type(self)(
            # NOTE: It's super important to use the input variables. Otherwise, we'll try to apply the indices twice
            #       (which won't work for the masked object).
            input_jets=self.input_jets[mask],
            input_splittings=self.input_splittings[mask],
            input_splittings_indices=self.input_splittings_indices[mask],
            values=self.values[mask],
            indices=self.indices[mask],
            possible_indices=self.possible_indices[mask],
        )


@attrs.define
class MaskedJets:
    """Container for masked jets.

    This just provides a consistent named interface to keep track of everything.
    """

    jets: ak.Array = attrs.field()
    selected_splittings: analysis_jet_substructure.JetSplittingArray = attrs.field()
    selected_splittings_index: AwkwardArray[AwkwardArray[int]] = attrs.field()


@attrs.define
class GroomingResultForTree:
    grooming_method: str = attrs.field()
    delta_R: npt.NDArray[np.float32] = attrs.field()
    z: npt.NDArray[np.float32] = attrs.field()
    kt: npt.NDArray[np.float32] = attrs.field()
    tau: npt.NDArray[np.float32] | None = attrs.field()
    n_to_split: npt.NDArray[np.int16] = attrs.field()
    n_groomed_to_split: npt.NDArray[np.int16] = attrs.field()
    # For SoftDrop, this is equivalent to n_sd.
    n_passed_grooming: npt.NDArray[np.int16] = attrs.field()

    def asdict(self, prefix: str) -> Iterable[tuple[str, npt.NDArray[np.generic]]]:
        for k, v in attrs.asdict(self, recurse=False).items():
            # Skip the grooming_method label or anything that is not available
            # (which is stored as None)
            if isinstance(v, str) or v is None:
                continue
            yield "_".join([self.grooming_method, prefix, k]), v


def _define_calculation_functions(
    jet_R: float,
    iterative_splittings: bool,
    selected_grooming_methods: list[str] | None = None,
) -> dict[str, functools.partial[tuple[npt.NDArray[Scalar], AwkwardArray[int], AwkwardArray[AwkwardArray[int]]]]]:
    """Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        jet_R: Jet resolution parameter.
        iterative_splittings: Whether calculating iterative splittings or not.
        selected_grooming_methods: Only return the calculations for the specified
            grooming methods. Used if we need to keep the output size down (eg. tau
            reclustering). Default: None. This indicates that all methods should be used.
            We also interpret passing an empty list as making no selection.
    Returns:
        dynamical_core, dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt z>0.2, leading_kt z>0.4, SD z>0.2, SD z>0.4
    """
    # Validation
    # This handles if we pass an empty selection of grooming methods, which we interpret as no selection.
    if not selected_grooming_methods:
        selected_grooming_methods = None

    functions: dict[
        str, functools.partial[tuple[npt.NDArray[Scalar], AwkwardArray[int], AwkwardArray[AwkwardArray[int]]]]
    ] = {
        "dynamical_core": functools.partial(analysis_jet_substructure.JetSplittingArray.dynamical_core, R=jet_R),  # type: ignore[dict-item]
        "dynamical_z": functools.partial(analysis_jet_substructure.JetSplittingArray.dynamical_z, R=jet_R),  # type: ignore[dict-item]
        "dynamical_kt": functools.partial(analysis_jet_substructure.JetSplittingArray.dynamical_kt, R=jet_R),  # type: ignore[dict-item]
        "dynamical_time": functools.partial(analysis_jet_substructure.JetSplittingArray.dynamical_time, R=jet_R),  # type: ignore[dict-item]
        "dynamical_core_z_cut_02": functools.partial(  # type: ignore[dict-item]
            analysis_jet_substructure.JetSplittingArray.dynamical_core,
            z_cutoff=0.2,
            R=jet_R,
        ),
        "dynamical_kt_z_cut_02": functools.partial(  # type: ignore[dict-item]
            analysis_jet_substructure.JetSplittingArray.dynamical_kt,
            z_cutoff=0.2,
            R=jet_R,
        ),
        "dynamical_time_z_cut_02": functools.partial(  # type: ignore[dict-item]
            analysis_jet_substructure.JetSplittingArray.dynamical_time,
            z_cutoff=0.2,
            R=jet_R,
        ),
        "leading_kt": functools.partial(  # type: ignore[dict-item]
            analysis_jet_substructure.JetSplittingArray.leading_kt,
        ),
        "leading_kt_z_cut_02": functools.partial(analysis_jet_substructure.JetSplittingArray.leading_kt, z_cutoff=0.2),  # type: ignore[dict-item]
        "leading_kt_z_cut_04": functools.partial(analysis_jet_substructure.JetSplittingArray.leading_kt, z_cutoff=0.4),  # type: ignore[dict-item]
    }
    # NOTE: This currently only works for iterative splittings...
    #       Calculating recursive is way harder in any array-like manner.
    if iterative_splittings:
        functions["soft_drop_z_cut_02"] = functools.partial(  # type: ignore[assignment]
            analysis_jet_substructure.JetSplittingArray.soft_drop, z_cutoff=0.2
        )
        functions["soft_drop_z_cut_04"] = functools.partial(  # type: ignore[assignment]
            analysis_jet_substructure.JetSplittingArray.soft_drop, z_cutoff=0.4
        )
    # Only apply the selected grooming methods if meaningful
    if selected_grooming_methods is not None:
        functions = {k: v for k, v in functions.items() if k in selected_grooming_methods}

    if not functions:
        msg = f"Provided selection of grooming methods ({selected_grooming_methods}), but none were selected! Check your input"
        raise ValueError(msg)

    return functions


def _select_and_retrieve_splittings(
    jets: ak.Array, mask: AwkwardArray[bool], iterative_splittings: bool
) -> tuple[ak.Array, analysis_jet_substructure.JetSplittingArray, AwkwardArray[AwkwardArray[int]]]:
    """Generalization of the function in analyze_tree to add the splitting index."""
    # Ensure that there are sufficient counts
    restricted_jets = jets[mask]

    # Add the splittings and indices.
    if iterative_splittings:
        # Only keep iterative splittings.
        restricted_splittings = restricted_jets.jet_splittings.iterative_splittings(restricted_jets.subjets)

        # Enable this test to determine if we've selected different sets of splittings with the
        # recursive vs iterative selections.
        # if (splittings.counts != restricted_jets.jet_splittings.counts).any():
        #    logger.warning("Disagreement between number of inclusive and recursive splittings (as expected!)")
        #    IPython.embed()
        restricted_splittings_indices = restricted_jets.subjets.iterative_splitting_index
    else:
        restricted_splittings = restricted_jets.jet_splittings
        # NOTE: Probably needs to be verified fully, but a quick check in Jan 2023 seems okay!
        restricted_splittings_indices = ak.local_index(restricted_jets.jet_splittings.kt)

    return restricted_jets, restricted_splittings, restricted_splittings_indices


@nb.njit  # type: ignore[misc]
def _calculate_splitting_number(
    all_splittings: analysis_jet_substructure.JetSplittingArray,
    selected_splittings: analysis_jet_substructure.JetSplittingArray,
    restricted_splittings_indices: AwkwardArray[AwkwardArray[int]],
    debug: bool = False,
) -> npt.NDArray[np.int16]:
    # NOTE: Could fit inside of a uint8, but we use int16 because uint8 wasn't written properly
    #       by uproot3, and ROOT doesn't handle int8 correctly.
    output = np.zeros(len(selected_splittings), dtype=np.int16)

    for i, (selected_splitting, restricted_splitting_indices, available_splittings_parents) in enumerate(
        zip(selected_splittings, restricted_splittings_indices, all_splittings.parent_index)  # noqa: B905
    ):
        # This would be to help out mypy, but it will probably interfere with numba, so we
        # just tell it to ignore the type. The point here is that parent_index of all_splittings is
        # equivalent to two levels of AwkwardArrays, but it's not so easy to type it that way.
        # available_splittings_parents = cast(AwkwardArray[int], available_splittings_parents_temp)
        # restricted_splitting_indices = restricted_splittings_indices[i]
        # available_splittings_parents = all_splittings[i].parent_index

        parent_indices = selected_splitting.parent_index
        if len(parent_indices):
            # We have at least one splitting, so we add an entry for it.
            output[i] += 1

            parent_index = parent_indices[0]
            if debug:
                print("parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)  # noqa: T201
            # print("i", i, "parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            # if i == 27:
            #    print("parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            while parent_index != -1:
                # Apparently contains isn't implemented either. So we just implement by hand.
                # if parent_index in restricted_splitting_indices:
                for index in restricted_splitting_indices:
                    # print("parent_index: {parent_index}, index: {index}".format(parent_index=parent_index, index=index))
                    if debug:
                        print("parent_index", parent_index, "index", index)  # noqa: T201
                    # print("parent_index, index: %d, %d" % (parent_index, index))
                    # print("i", i, "parent_index", parent_index, "index", index)
                    if parent_index == index:
                        if debug:
                            print("Found parent index:", index)  # noqa: T201
                        output[i] += 1
                        # import IPython; IPython.embed()
                        parent_index = available_splittings_parents[parent_index]  # type: ignore[index]
                        if debug:
                            print("New parent index:", parent_index)  # noqa: T201
                        # print("Breaking...")
                        break
                else:
                    # We didn't find it, but we need to advance forward.
                    parent_index = available_splittings_parents[parent_index]  # type: ignore[index]

            if debug:
                print("output[i]", output[i])  # noqa: T201

    return output


def calculate_splitting_number(
    all_splittings: analysis_jet_substructure.JetSplittingArray,
    selected_splittings: analysis_jet_substructure.JetSplittingArray,
    restricted_splittings_indices: AwkwardArray[AwkwardArray[int]],
    debug: bool = False,
) -> npt.NDArray[np.int16]:
    """Wrapper around calculating the splitting number

    The wrapper takes care of the case where there are no jets with selected splittings.
    In that case, the parent_index type in the awkward array is ambiguous (ie. n * "unknown"),
    which then breaks numba compilation. This wrapper checks for this condition, and if it finds
    it, immediately returns all 0s (ie. untagged).

    Note that this may be convolving truly untagged with taking the first split. However, we're not
    overly worried about this in Oct 2022, especially since we just look at these calculations for QA.
    So nothing further is done for now. (I think it's convolved, but I'm not certain)
    """
    if ak.any(ak.num(selected_splittings.parent_index, axis=1) > 0):
        return _calculate_splitting_number(  # type: ignore[no-any-return]
            all_splittings=all_splittings,
            selected_splittings=selected_splittings,
            restricted_splittings_indices=restricted_splittings_indices,
            debug=debug,
        )
    logger.warning(
        "There were no jets with selected splittings, so we short circuited the splittings calculation."
        " This avoids issues with slicing with numba. This should be most common when working with low pt hat bins."
    )
    return np.zeros(len(selected_splittings), dtype=np.int16)


@nb.njit  # type: ignore[misc]
def _find_contributing_subjets(input_jet: ak.Array, groomed_index: int) -> list[analysis_jet_substructure.Subjet]:
    """Find subjets which contribute to a given grooming index.

    Args:
        input_jet: Inputs jets.
        groomed_index: Selected grooming index (ie. splitting).
    Returns:
        Subjets contributing to the splitting.
    """
    # subjets = []
    # for sj in input_jet.subjets:
    #    if sj.parent_splitting_index == groomed_index:
    #        subjets.append(sj)
    # return subjets
    return [sj for sj in input_jet.subjets if sj.parent_splitting_index == groomed_index]


@nb.njit  # type: ignore[misc]
def _sort_subjets(
    input_jet: ak.Array, input_subjets: list[analysis_jet_substructure.Subjet]
) -> tuple[analysis_jet_substructure.Subjet, analysis_jet_substructure.Subjet]:
    pts = []
    for sj in input_subjets:
        px = 0
        py = 0
        for constituent_index in sj.constituent_indices:
            constituent = input_jet.jet_constituents[constituent_index]
            px += constituent.pt * np.cos(constituent.phi)
            py += constituent.pt * np.sin(constituent.phi)
        pts.append(np.sqrt(px**2 + py**2))

    leading = input_subjets[0]
    subleading = input_subjets[1]

    if pts[1] > pts[0]:
        leading, subleading = subleading, leading

    return leading, subleading


@nb.njit  # type: ignore[misc]
def _subjet_shared_momentum(
    generator_like_subjet: analysis_jet_substructure.Subjet,
    generator_like_jet: ak.Array,
    measured_like_subjet: analysis_jet_substructure.Subjet,
    measured_like_jet: ak.Array,
    match_using_distance: bool = False,
) -> float:
    sum_pt = 0
    delta = analysis_jet_substructure.DISTANCE_DELTA

    for generator_like_constituent_index in generator_like_subjet.constituent_indices:
        generator_like_constituent = generator_like_jet.jet_constituents[generator_like_constituent_index]
        for measured_like_constituent_index in measured_like_subjet.constituent_indices:
            measured_like_constituent = measured_like_jet.jet_constituents[measured_like_constituent_index]
            if match_using_distance:
                if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                    continue
                if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                    continue
            else:  # noqa: PLR5501
                if generator_like_constituent.id != measured_like_constituent.id:
                    continue

            sum_pt += generator_like_constituent.pt
            # We've matched once - no need to match again.
            # Otherwise, the run the risk of summing a generator-like constituent pt twice.
            break

    return sum_pt


@nb.njit  # type: ignore[misc]
def _subjet_pt(subjet: analysis_jet_substructure.Subjet, jet: ak.Array) -> float:
    """Calculate subjet pt by hand.

    Since we have the full vectors, we calculate the vectors and then take the magnitude.

    Note:
        This would have been natural to do with vector. However, when it was written, vector
        wasn't available, so we did it by hand.
    """
    px: float = 0.0
    py: float = 0.0
    for constituent_index in subjet.constituent_indices:
        constituent = jet.jet_constituents[constituent_index]
        px += constituent.pt * np.cos(constituent.phi)
        py += constituent.pt * np.sin(constituent.phi)
    return np.sqrt(px**2 + py**2)  # type: ignore[no-any-return]


@nb.njit  # type: ignore[misc]
def _subjet_contained_in_subjet(
    generator_like_subjet: analysis_jet_substructure.Subjet,
    generator_like_jet: ak.Array,
    measured_like_subjet: analysis_jet_substructure.Subjet,
    measured_like_jet: ak.Array,
    match_using_distance: bool = False,
) -> bool:
    return (  # type: ignore[no-any-return]
        _subjet_shared_momentum(
            generator_like_subjet=generator_like_subjet,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subjet,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        )
        / _subjet_pt(generator_like_subjet, generator_like_jet)
    ) > 0.5


@nb.njit  # type: ignore[misc]
def determine_matched_jets_numba(
    generator_like_jets: ak.Array,
    generator_like_splittings: analysis_jet_substructure.JetSplittingArray,
    generator_like_groomed_values: AwkwardArray[float],
    generator_like_groomed_indices: AwkwardArray[int],
    measured_like_jets: ak.Array,
    measured_like_splittings: analysis_jet_substructure.JetSplittingArray,
    measured_like_groomed_values: AwkwardArray[float],
    measured_like_groomed_indices: AwkwardArray[int],
    match_using_distance: bool,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    n_jets = len(measured_like_jets)
    leading_matching = np.full(n_jets, -1, dtype=np.int16)
    subleading_matching = np.full(n_jets, -1, dtype=np.int16)

    for (
        i,
        (
            generator_like_jet,
            _generator_like_splitting,
            _generator_like_groomed_value,
            generator_like_groomed_index_array,
            measured_like_jet,
            _measured_like_splitting,
            _measured_like_groomed_value,
            measured_like_groomed_index_array,
        ),
    ) in enumerate(
        zip(  # noqa: B905
            generator_like_jets,
            generator_like_splittings,
            generator_like_groomed_values,
            generator_like_groomed_indices,
            measured_like_jets,
            measured_like_splittings,
            measured_like_groomed_values,
            measured_like_groomed_indices,
        )
    ):
        # Find the selected index if it's available.
        if len(measured_like_groomed_index_array) > 0 and len(generator_like_groomed_index_array) > 0:
            # This is required. If not, we handle the other cases and continue.
            pass
        elif len(measured_like_groomed_index_array) > 0:
            # Assign 0 for this case and move on.
            leading_matching[i] = 0
            subleading_matching[i] = 0
            continue
        else:
            # Use the default values and continue
            continue

        # We maintain the singles structure per jet so that each index can be applied to each jet (ie. array entry)
        # (this also lets us keep empty cases accounted for). However, we've now already accounted for empty cases,
        # and it's much easier to work with the individual values, so we extract them. We know each one will have only
        # one entry because it's from an argmax call.
        generator_like_groomed_index = generator_like_groomed_index_array[0]
        measured_like_groomed_index = measured_like_groomed_index_array[0]

        # Find the contributing subjets
        generator_like_subjets = _find_contributing_subjets(generator_like_jet, generator_like_groomed_index)
        measured_like_subjets = _find_contributing_subjets(measured_like_jet, measured_like_groomed_index)
        # Sort
        generator_like_leading, generator_like_subleading = _sort_subjets(generator_like_jet, generator_like_subjets)
        measured_like_leading, measured_like_subleading = _sort_subjets(measured_like_jet, measured_like_subjets)

        # Compare
        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 2
        else:
            leading_matching[i] = 3

        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 2
        else:
            subleading_matching[i] = 3

    return leading_matching, subleading_matching


def prong_matching_numba_wrapper(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
    match_using_distance: bool = False,
) -> dict[str, npt.NDArray[np.int16]]:
    """Performs prong matching for the provided collections.

    Note:
        0 is there were insufficient constituents to form a splitting, 1 is properly matched, 2 is mistagged
        (leading -> subleading or subleading -> leading), 3 is untagged (failed).

    Args:
        measured_like_jets_calculation: Grooming calculation for measured-like jets (hybrid for hybrid-det level matching).
        measured_like_jets_label: Label for measured jets (hybrid for hybrid-det level matching).
        generator_like_jets_calculation: Grooming calculation for generator-like jets (det level for hybrid-det level matching).
        generator_like_jets_label: Label for generator jets (det_level for hybrid-det level matching).
        grooming_method: Name of the grooming method.
        match_using_distance: If True, match using distance. Otherwise, match using the stored label.
    Returns:
        Matching and subleading matching values.
    """
    # Matching
    grooming_results = {}
    logger.info(f"Performing {measured_like_jets_label}-{generator_like_jets_label} matching for {grooming_method}")
    # If there are only single particle jets, awkward cannot determine the proper type for the input_splittings,
    # which then causes numba compilation to fail. To workaround this issue, we look for "unknown" in the type,
    # and in that case, we skip the matching. The know that we can assign it to -1 because if there are no splittings
    # (which must be the case for single particle jets), then the subjets can't possibly be matched to other subjets.
    _contains_only_single_particle_jets = {
        generator_like_jets_label: "unknown" in str(ak.type(generator_like_jets_calculation.input_splittings)),
        measured_like_jets_label: "unknown" in str(ak.type(measured_like_jets_calculation.input_splittings)),
    }
    if any(_contains_only_single_particle_jets.values()):
        logger.warning(
            f"Only single particle jets for {' + '.join([repr(k) for k, v in _contains_only_single_particle_jets.items() if v])} so we skip the subjet matching (there will be no matches)!"
        )
        # Initialize the full set of matching values to -1 to indicate that there is no match.
        # NOTE: If the matched jets index conventions change, it must also be changed here!
        n_jets = len(measured_like_jets_calculation.input_jets)
        leading_matching = np.full(n_jets, -1, dtype=np.int16)
        subleading_matching = np.full(n_jets, -1, dtype=np.int16)
    else:
        leading_matching, subleading_matching = determine_matched_jets_numba(
            generator_like_jets=generator_like_jets_calculation.input_jets,
            generator_like_splittings=generator_like_jets_calculation.input_splittings,
            generator_like_groomed_values=generator_like_jets_calculation.values,
            generator_like_groomed_indices=generator_like_jets_calculation.indices,
            measured_like_jets=measured_like_jets_calculation.input_jets,
            measured_like_splittings=measured_like_jets_calculation.input_splittings,
            measured_like_groomed_values=measured_like_jets_calculation.values,
            measured_like_groomed_indices=measured_like_jets_calculation.indices,
            match_using_distance=match_using_distance,
        )

    for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
        grooming_results[
            f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
        ] = matching

    return grooming_results


@nb.njit  # type: ignore[misc]
def _subjet_momentum_fraction_in_jet(
    generator_like_subjet: analysis_jet_substructure.Subjet,
    generator_like_jet: ak.Array,
    measured_like_jet: ak.Array,
    match_using_distance: bool = False,
) -> float:
    """Calculate subjet momentum fraction contained within another jet.

    Unfortunately, we can't blindly use the `_subjet_shared_momentum` function because
    the interfaces vary between jet constituents and subjet constituents. We could refactor them,
    but the code is simple enough that it's easier just to implement the different versions.
    """
    sum_pt: float = 0.0
    delta = analysis_jet_substructure.DISTANCE_DELTA

    for generator_like_constituent_index in generator_like_subjet.constituent_indices:
        generator_like_constituent = generator_like_jet.jet_constituents[generator_like_constituent_index]
        for measured_like_constituent in measured_like_jet.jet_constituents:
            if match_using_distance:
                if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                    continue
                if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                    continue
            else:  # noqa: PLR5501
                if generator_like_constituent.id != measured_like_constituent.id:
                    continue

            sum_pt += generator_like_constituent.pt
            # We've matched once - no need to match again.
            # Otherwise, the run the risk of summing a generator-like constituent pt twice.
            break

    return sum_pt / _subjet_pt(generator_like_subjet, generator_like_jet)  # type: ignore[no-any-return]


@nb.njit  # type: ignore[misc]
def generator_subjet_momentum_fraction_in_measured_jet_numba(
    generator_like_jets: ak.Array,
    generator_like_splittings: analysis_jet_substructure.JetSplittingArray,
    generator_like_groomed_indices: AwkwardArray[AwkwardArray[int]],
    measured_like_jets: ak.Array,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Determine the generator-like subjet momentum fraction stored in a measured-like jet.

    Note:
        This isn't looking at the measured-like subjet. It's about finding where subjets go,
        especially for those which aren't matched at the subjet level (they of course must have
        matched at the overall jet level).
    """
    n_jets = len(measured_like_jets)
    leading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)
    subleading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)

    for (
        i,
        (
            generator_like_jet,
            _generator_like_splitting,
            generator_like_groomed_index_array,
            measured_like_jet,
        ),
    ) in enumerate(
        zip(  # noqa: B905
            generator_like_jets,
            generator_like_splittings,
            generator_like_groomed_indices,
            measured_like_jets,
        )
    ):
        # Find the selected index if it's available.
        if len(generator_like_groomed_index_array) > 0:
            # This is required. Otherwise, we just skip case.
            pass
        else:
            # Use the default values and continue
            continue

        # Retrieve the generator like subjet.
        # We know each one will have only one entry because it's from an argmax call, so we extract it.
        generator_like_groomed_index = generator_like_groomed_index_array[0]
        # Find the contributing subjets
        generator_like_subjets = _find_contributing_subjets(generator_like_jet, generator_like_groomed_index)
        # Sort
        generator_like_leading, generator_like_subleading = _sort_subjets(generator_like_jet, generator_like_subjets)

        leading_momentum_fraction[i] = _subjet_momentum_fraction_in_jet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_jet=measured_like_jet,
        )
        subleading_momentum_fraction[i] = _subjet_momentum_fraction_in_jet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_jet=measured_like_jet,
        )

    return leading_momentum_fraction, subleading_momentum_fraction


def generator_subjet_momentum_fraction_in_measured_jet_numba_wrapper(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
) -> dict[str, npt.NDArray[np.float32]]:
    grooming_results = {}

    _contains_only_single_particle_jets = {
        generator_like_jets_label: "unknown" in str(ak.type(generator_like_jets_calculation.input_splittings)),
        measured_like_jets_label: "unknown" in str(ak.type(measured_like_jets_calculation.input_splittings)),
    }
    if any(_contains_only_single_particle_jets.values()):
        logger.warning(
            f"Only single particle jets for {' + '.join([repr(k) for k, v in _contains_only_single_particle_jets.items() if v])} so we skip the subjet momentum fraction calculation (the fraction will always be zero if there are no subjets)!"
        )
        # Initialize the full set of matching values to -1 to indicate that there is no match.
        # NOTE: If the matched jets index conventions change, it must also be changed here!
        n_jets = len(measured_like_jets_calculation.input_jets)
        leading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)
        subleading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)
    else:
        (
            leading_momentum_fraction,
            subleading_momentum_fraction,
        ) = generator_subjet_momentum_fraction_in_measured_jet_numba(
            generator_like_jets=generator_like_jets_calculation.input_jets,
            generator_like_splittings=generator_like_jets_calculation.input_splittings,
            generator_like_groomed_indices=generator_like_jets_calculation.indices,
            measured_like_jets=measured_like_jets_calculation.input_jets,
        )

    for label, momentum_fraction in [
        ("leading", leading_momentum_fraction),
        ("subleading", subleading_momentum_fraction),
    ]:
        # groomingMethod + "_hybrid_det_level_matching_leading_pt_fraction_in_hybrid_jet"
        # NOTE: This is different than the name in the hardest kt cross check task. Since I had more time to think about it,
        #       this name makes more sense to me.
        grooming_results[
            f"{grooming_method}_{generator_like_jets_label}_{label}_subjet_momentum_fraction_in_{measured_like_jets_label}_jet"
        ] = momentum_fraction

    return grooming_results


def _calculate_jet_kinematics(
    constituents: analysis_jet_substructure.JetConstituentArray, float_type: npt.DTypeLike | None = None
) -> tuple[ak.Array, ak.Array]:
    """Calculate jet kinematics.

    Since `vector` isn't yet available, we perform the four vector calculations by hand.

    Args:
        constituents: Jet constituents.
        float_type: Float to be used for conversion. Default: None. This uses the ak default.
    Returns:
        eta, phi
    """
    # jet_four_vec = jets.jet_constituents.four_vectors().sum()
    # Since vector isn't ready yet, just do this by hand...
    px = ak.sum(constituents.pt * np.cos(constituents.phi), axis=1)  # type: ignore[call-overload]
    py = ak.sum(constituents.pt * np.sin(constituents.phi), axis=1)  # type: ignore[call-overload]
    pz = ak.sum(constituents.pt * np.sinh(constituents.eta), axis=1)  # type: ignore[call-overload]
    # Formulas just from inverting the above.
    eta = np.arcsinh(pz / np.sqrt(px**2 + py**2))
    phi = np.arctan2(py, px)
    if float_type is None:
        return eta, phi
    return ak.values_astype(eta, float_type), ak.values_astype(phi, float_type)


def calculate_three_input_level_skim_impl(
    all_jets: ak.Array,
    prefixes: Mapping[str, str],
    iterative_splittings: bool,
    jet_R: float,
    scale_factor: float,
    create_friend_tree: bool = False,
    draw_example_splittings_to_filename: Path | None = None,
    selected_grooming_methods: list[str] | None = None,
    match_det_level_true_using_distance: bool = False,
) -> T_GroomingResults:
    """Calculate the flat skim based on the all_jets input for three input levels.

    In practice, this means skimming to a flat tree for embedding.

    Args:
        all_jets: All jets structured dict
        prefixes: Mapping from our standard names to those which are used in the stored data.
        iterative_splittings: If True, we should only consider iterative splittings.
        jet_R: Jet radius.
        scale_factor: Scale factor.
        create_friend_tree: Create a friend tree instead of the standard tree. It contains
            supplemental information. See the code for precisely what it contains. Default: False.
        draw_example_splittings: If True, draw a few interesting splitting graphs. Default: False.
        selected_grooming_methods: Select a subset of grooming methods. Default: None, which
            includes all grooming methods.
        match_det_level_true_using_distance: If True, match using distance. Otherwise, match
            using the stored label. Default: False (i.e. use label)

    Returns:
        Grooming results.
    """
    # Setup
    # Output consistent types.
    float_type = np.float32
    to_float = functools.partial(ak.values_astype, to=np.float32)

    # Do the calculations
    # Do not mask on the number of constituents. This would prevent tagged <-> untagged migrations in the response.
    # _has_splittings_mask = [ak.num(_j.jet_constituents, axis=1) > 1 for _j in all_jets.values()]
    # mask = functools.reduce(operator.and_, _has_splittings_mask)
    # This is where the double counting cut was formerly implemented. Now, we implement it (or any variations on it)
    # at the analysis level. That way, we maintain full flexibility at the skim level
    # As of 7 Feb 2023, this is just an all true mask, but it's fine to leave it here,
    # since it makes it convenient to add a possible mask in the future
    mask = all_jets["hybrid"].jet_pt > 0

    # Mask the jets
    masked_jets: dict[str, MaskedJets] = {}
    for prefix, input_jets in all_jets.items():
        masked_jets[prefix] = MaskedJets(
            *_select_and_retrieve_splittings(
                input_jets,
                mask,
                iterative_splittings=iterative_splittings,
            )
        )

    # Results output
    grooming_results = {}
    if create_friend_tree:
        # Possible additional fields to calculate and store in a friend tree.
        # As of July 2023, we don't have anything that we want to add this way
        # NOTE: As the skim is re-run, values are generally transitioned to the standard tree
        # the next time it's generated.
        ...
    else:
        grooming_results["scale_factor"] = to_float((masked_jets["true"].jets.jet_pt[mask] * 0) + scale_factor)

        for prefix, input_jets in masked_jets.items():
            # Add jet pt and general jet properties.
            # Jet kinematics
            grooming_results[f"{prefix}_jet_pt"] = to_float(input_jets.jets.jet_pt)
            grooming_results[f"{prefix}_jet_eta"], grooming_results[f"{prefix}_jet_phi"] = _calculate_jet_kinematics(
                input_jets.jets.jet_constituents,
                float_type=float_type,
            )
            # Leading track
            leading_track_name = f"{prefix}_leading_track_pt"
            if prefix == "hybrid":
                # First, store the unsubtracted (which we use for the double counting cut) as the normal leading track pt.
                if "leading_track_pt" in ak.fields(input_jets.jets):
                    grooming_results[leading_track_name] = to_float(input_jets.jets["leading_track_pt"])
                # Then update the name for the subtracted constituents in data.
                leading_track_name = f"{prefix}_leading_track_pt_sub"
            grooming_results[leading_track_name] = to_float(
                ak.max(input_jets.jets.jet_constituents.pt, axis=1, mask_identity=False)
            )
            # Cross check that we haven't somehow found a case where we have jets, but no leading track.
            # Obviously this shouldn't be possible, but ak.max forces us to lose this safety (otherwise,
            # it returns an option type which is a pain).
            assert len(grooming_results[leading_track_name]) == len(grooming_results[f"{prefix}_jet_pt"])

        # Perform our calculations.
        functions: dict[
            str, functools.partial[tuple[npt.NDArray[np.float32], AwkwardArray[int], AwkwardArray[AwkwardArray[int]]]]
        ] = _define_calculation_functions(
            jet_R=jet_R, iterative_splittings=iterative_splittings, selected_grooming_methods=selected_grooming_methods
        )
        for func_name, func in functions.items():
            logger.debug(f"func_name: {func_name}")
            calculations = {
                prefix: Calculation(
                    masked_jets[prefix].jets,
                    masked_jets[prefix].selected_splittings,
                    masked_jets[prefix].selected_splittings_index,
                    *func(masked_jets[prefix].selected_splittings),
                )
                for prefix in prefixes
            }

            for prefix, calculation in calculations.items():
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                    debug=False,
                )
                logger.info(f"Done with first splitting calculation, {func_name}, {prefix}")
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                    debug=False,
                )
                logger.info(f"Done with second splitting calculation, {func_name}, {prefix}")

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.delta_R, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    ),
                    z=to_float(
                        ak.flatten(
                            ak.fill_none(ak.pad_none(groomed_splittings.z, 1), analysis_jet_substructure.UNFILLED_VALUE)
                        )
                    ),
                    kt=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.kt, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    ),
                    # Only include this if it's available.
                    tau=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.tau, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    )
                    if "tau" in ak.fields(groomed_splittings) and groomed_splittings.tau is not None
                    else None,
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    # NOTE: Could fit inside of a uint8, but we use int16 because uint8 wasn't written properly
                    #       by uproot3, and ROOT doesn't handle int8 correctly.
                    n_passed_grooming=ak.values_astype(ak.num(calculation.possible_indices, axis=1), np.int16),
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

            logger.debug("Before prong matching")
            # Hybrid-det level matching.
            # We match using labels here because the we were able to propagate them through the subtraction.
            hybrid_det_level_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=calculations["hybrid"],
                measured_like_jets_label="hybrid",
                generator_like_jets_calculation=calculations["det_level"],
                generator_like_jets_label="det_level",
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(hybrid_det_level_matching_results)
            logger.debug("Done with first prong matching")
            # Det level-true matching
            # We match using labels here because otherwise the reconstruction can cause the particles to move
            # enough that they may not match within a particular distance. (plus, the labels work!)
            det_level_true_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=calculations["det_level"],
                measured_like_jets_label="det_level",
                generator_like_jets_calculation=calculations["true"],
                generator_like_jets_label="true",
                grooming_method=func_name,
                match_using_distance=match_det_level_true_using_distance,
            )
            grooming_results.update(det_level_true_matching_results)
            logger.debug("Done with second prong matching")
            # Subjet momentum fraction in hybrid
            grooming_results.update(
                generator_subjet_momentum_fraction_in_measured_jet_numba_wrapper(
                    measured_like_jets_calculation=calculations["hybrid"],
                    measured_like_jets_label="hybrid",
                    generator_like_jets_calculation=calculations["det_level"],
                    generator_like_jets_label="det_level",
                    grooming_method=func_name,
                )
            )
            logger.debug("Done with det level subjet momentum fraction in hybrid jets")

            # Look for leading kt just because it's easier to understand conceptually.
            hybrid_det_level_leading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_leading"]
            hybrid_det_level_subleading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_subleading"]
            if (
                draw_example_splittings_to_filename is not None
                and func_name == "leading_kt"
                and ak.any((hybrid_det_level_leading_matching == 1) & (hybrid_det_level_subleading_matching == 3))
            ):
                from jet_substructure.analysis import draw_splitting  # pyright: ignore [reportMissingImports]
                from networkx.drawing.nx_pylab import draw  # pyright: ignore [reportMissingModuleSource] # noqa: F401

                # Find a sufficiently interesting jet (ie high enough pt)
                mask_jets_of_interest = (
                    (hybrid_det_level_leading_matching.properly & hybrid_det_level_subleading_matching.failed)
                    & (masked_jets["hybrid"].jets.jet_pt > 80)
                    & ak.flatten(calculations["det_level"].splittings.kt > 10)
                )

                # Look at most the first 5 jets.
                for i, hybrid_jet in enumerate(masked_jets["hybrid"].jets[mask_jets_of_interest][:5]):
                    # Find the hybrid jet and splitting of interest.
                    # hybrid_jet = masked_hybrid_jets[mask_jets_of_interest][0]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    hybrid_jet_selected_splitting_index = calculations["hybrid"].indices[mask_jets_of_interest][i][0]
                    # Same for det level.
                    det_level_jet = masked_jets["det_level"].jets[mask_jets_of_interest][i]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    det_level_jet_selected_splitting_index = calculations["det_level"].indices[mask_jets_of_interest][
                        i
                    ][0]

                    splitting_graph_output_dir = draw_example_splittings_to_filename.parent

                    # Draw the splittings
                    draw_splitting.splittings_graph(
                        jet=hybrid_jet,
                        path=splitting_graph_output_dir / "leading_correct_subleading_failed/",
                        filename=f"{i}_hybrid_splittings_jet_pt_{hybrid_jet.jet_pt:.1f}GeV_selected_splitting_index_{hybrid_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=hybrid_jet_selected_splitting_index,
                    )
                    draw_splitting.splittings_graph(
                        jet=det_level_jet,
                        path=splitting_graph_output_dir / "leading_correct_subleading_failed/",
                        filename=f"{i}_det_level_splittings_jet_pt_{det_level_jet.jet_pt:.1f}GeV_selected_splitting_index_{det_level_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=det_level_jet_selected_splitting_index,
                    )

            logger.debug(f"Completed {func_name}")

    # Since we're just returning a dict of np arrays, it's better to ensure that
    # they're consistently cast as np arrays (ie. a "regular" array could still
    # be wrapped in an ak.Array)
    return {k: np.asarray(v) for k, v in grooming_results.items()}


def calculate_one_or_two_input_level_skim_impl(
    all_jets: ak.Array,
    collision_system: str,
    prefixes: Mapping[str, str],
    iterative_splittings: bool,
    jet_R: float,
    create_friend_tree: bool = False,
    scale_factors: Mapping[int, float] | None = None,
    selected_grooming_methods: list[str] | None = None,
) -> T_GroomingResults:
    """Calculate the flat skim based on the all_jets input for one or two input levels.

    In practice, this means skimming to a flat tree for data or MC.

    Args:
        all_jets: All jets structured dict
        collision_system: Collision system.
        prefixes: Mapping from our standard names to those which are used in the stored data.
        iterative_splittings: If True, we should only consider iterative splittings.
        jet_R: Jet radius.
        create_friend_tree: Create a friend tree instead of the standard tree. It contains
            supplemental information. See the code for precisely what it contains. Default: False.
        scale_factors: Scale factors.
        selected_grooming_methods: Select a subset of grooming methods. Default: None, which
            includes all grooming methods.

    Returns:
        Grooming results.
    """
    # Setup
    # Output consistent types.
    float_type = np.float32
    to_float = functools.partial(ak.values_astype, to=np.float32)

    # Dataset wide masks
    # Select everything by default. We know that there must be at least one set of jets, so we're safe to select on 0.
    mask = all_jets[next(iter(prefixes.keys()))].jet_pt > 0
    # Special selections for pythia.
    # Apparently I can get pt hard < 5. Which is bizarre, at least according to the binning...
    # Filter these out when applicable.
    if collision_system in ["pythia", "pp_MC", "PbPb_MC"] and "pt_hard" in all_jets:
        # The jets object will contain the pt hard bin if it's available.
        mask = mask & (all_jets["pt_hard"] >= 5.0)

    masked_jets: dict[str, MaskedJets] = {}
    # for prefix, input_jets in all_jets.items():
    for prefix in prefixes:
        input_jets = all_jets[prefix]
        masked_jets[prefix] = MaskedJets(
            *_select_and_retrieve_splittings(
                input_jets,
                mask,
                iterative_splittings=iterative_splittings,
            )
        )

    # Results output
    grooming_results = {}
    # And start constructing the tree
    if create_friend_tree:
        # Possible additional fields to calculate and store in a friend tree.
        # As of July 2023, we don't have anything that we want to add this way
        # NOTE: As the skim is re-run, values are generally transitioned to the standard tree
        # the next time it's generated.
        ...
    else:
        for prefix, input_jets in masked_jets.items():
            # Add jet pt and general jet properties.
            # Jet kinematics
            grooming_results[f"{prefix}_jet_pt"] = to_float(input_jets.jets.jet_pt)
            grooming_results[f"{prefix}_jet_eta"], grooming_results[f"{prefix}_jet_phi"] = _calculate_jet_kinematics(
                input_jets.jets.jet_constituents,
                float_type=float_type,
            )
            # Leading track
            # NOTE: Since this is for data, it doesn't really matter, but better to always do the right thing.
            leading_track_name = f"{prefix}_leading_track_pt"
            # NOTE: We would include embedPythia here, but we don't run the embedding through this function, so we can ignore it.
            if prefix == "data" and collision_system == "PbPb":
                # First, store the unsubtracted (which we use for the double counting cut).
                if "leading_track_pt" in ak.fields(input_jets.jets):
                    grooming_results[leading_track_name] = to_float(input_jets.jets["leading_track_pt"])
                # Then update the name for the subtracted constituents in data.
                leading_track_name = f"{prefix}_leading_track_pt_sub"
            grooming_results[leading_track_name] = to_float(
                ak.max(input_jets.jets.jet_constituents.pt, axis=1, mask_identity=False)
            )
            # Cross check that we haven't somehow found a case where we have jets, but no leading track.
            # Obviously this shouldn't be possible, but ak.max forces us to lose this safety (otherwise,
            # it returns an option type which is a pain).
            assert len(grooming_results[leading_track_name]) == len(grooming_results[f"{prefix}_jet_pt"])

            # Perform our calculations.
            functions: dict[
                str,
                functools.partial[tuple[npt.NDArray[np.float32], AwkwardArray[int], AwkwardArray[AwkwardArray[int]]]],
            ] = _define_calculation_functions(
                jet_R=jet_R,
                iterative_splittings=iterative_splittings,
                selected_grooming_methods=selected_grooming_methods,
            )
            for func_name, func in functions.items():
                logger.debug(f"prefix: {prefix}, grooming function: {func_name}")
                calculation = Calculation(
                    input_jets.jets,
                    input_jets.selected_splittings,
                    input_jets.selected_splittings_index,
                    *func(input_jets.selected_splittings),
                )
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                )
                logger.debug("Done with first splitting calculation")
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                    debug=False,
                )
                logger.debug("Done with second splitting calculation")

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.delta_R, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    ),
                    z=to_float(
                        ak.flatten(
                            ak.fill_none(ak.pad_none(groomed_splittings.z, 1), analysis_jet_substructure.UNFILLED_VALUE)
                        )
                    ),
                    kt=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.kt, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    ),
                    # Only include this if it's available.
                    tau=to_float(
                        ak.flatten(
                            ak.fill_none(
                                ak.pad_none(groomed_splittings.tau, 1), analysis_jet_substructure.UNFILLED_VALUE
                            )
                        )
                    )
                    if "tau" in ak.fields(groomed_splittings) and groomed_splittings.tau is not None
                    else None,
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    # NOTE: Could fit inside of a uint8, but we use int16 because uint8 wasn't written properly
                    #       by uproot3, and ROOT doesn't handle int8 correctly.
                    n_passed_grooming=ak.values_astype(ak.num(calculation.possible_indices, axis=1), np.int16),
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

        # Add scale factors when appropriate (ie for pythia)
        if collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
            # Help out mypy...
            assert scale_factors is not None
            # Validation. We make a copy to ensure that we don't modify the input.
            output_scale_factors = dict(scale_factors)

            # There is apparently a pt hard > 1000 in this dataset! This ends up with an entry in bin 21, which is weird.
            # So we copy the scale factor for pt hard bin 20 to 21 to cover it. It should be more or less correct.
            output_scale_factors[21] = output_scale_factors[20]

            # Need to mask because we didn't when masking the original jets.
            pt_hard_bins = np.array(all_jets["pt_hard_bin"][mask], dtype=np.int16)
            logger.debug(f"Pt hard bins contained in the file: {np.unique(pt_hard_bins)}")
            pythia_specific_columns = {
                "scale_factor": np.array([output_scale_factors[b] for b in pt_hard_bins], dtype=np.float32),
                "pt_hard_bin": pt_hard_bins,
            }
            # The track skim doesn't have this info available, so we might have to skip it.
            if "pt_hard" in all_jets:
                pythia_specific_columns["pt_hard"] = to_float(all_jets["pt_hard"][mask])
            grooming_results.update(pythia_specific_columns)

    # Since we're just returning a dict of np arrays, it's better to ensure that
    # they're consistently cast as np arrays (ie. a "regular" array could still
    # be wrapped in an ak.Array)
    return {k: np.asarray(v) for k, v in grooming_results.items()}


def _write_skim_output(
    grooming_results: T_GroomingResults,
    output_filename: Path,
    write_root: bool = True,
    output_tree_name: str = "tree",
    write_parquet: bool = False,
    write_feather: bool = False,
    create_friend_tree: bool = False,
) -> bool:
    """Write skim output to file.

    OBSOLETE! (practically - as part of mammoth v1)

    Args:
        grooming_results: The grooming results to write.
        output_filename: The output filename.
        write_root: Whether to write a root file.
        output_tree_name: The name of the ROOT output tree.
        write_parquet: Whether to write a parquet file.
        write_feather: Whether to write a feather file.
        create_friend_tree: Whether to create a friend tree. Note that this just changes the filename.
    Returns:
        True if successful, False otherwise.
    """
    # Setup
    # In the case of creating a friend tree, we've added some fields to an existing tree,
    # which we'll load as a friend. As of July 2023, we don't use this functionality anymore
    # since we can analyze the track skim directly, but it could bea useful in the future,
    # so we leave it here. In this case, we want the same write functionality - we just want
    # to change the name.
    if create_friend_tree:
        output_filename = Path(str(output_filename.with_suffix("")) + "_friend.root")

    # Write output
    # For extra safety
    if write_root or write_parquet or write_feather:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
    # Standard is to write out a flat root tree
    if write_root:
        # First, convert to numpy since we want to write to an output tree.
        logger.info(f"Writing data skim to {output_filename}")
        # Write with uproot
        with uproot.recreate(output_filename) as output_file:
            # Write all of the calculations
            output_file[output_tree_name] = grooming_results
    # Some alternative formats for other analysis techniques.
    if write_parquet:
        logger.info("Writing parquet...")
        ak.to_parquet(grooming_results, output_filename.with_suffix(".parquet"), compression="zstd")
    if write_feather:
        logger.info("Writing feather...")
        import pyarrow.feather

        pa_table = ak.to_arrow_table(grooming_results)
        pyarrow.feather.write_feather(pa_table, output_filename.with_suffix(".feather"), compression="zstd")

    return True


def _calculate_embedding_skim_mammoth_framework_v1(
    all_jets: ak.Array,
    input_filename: Path,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    jet_R: float,
    output_filename: Path,
    scale_factor: float,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    write_root: bool = True,
    write_feather: bool = False,
    write_parquet: bool = False,
    draw_example_splittings_to_filename: Path | None = None,
    selected_grooming_methods: list[str] | None = None,
) -> tuple[bool, Path, str]:
    """Calculate the embedding skim using the mammoth v1 framework.

    OBSOLETE! (practically - as part of mammoth v1)
    """
    # First, directly run the skim
    grooming_results = calculate_three_input_level_skim_impl(
        all_jets=all_jets,
        prefixes=prefixes,
        iterative_splittings=iterative_splittings,
        jet_R=jet_R,
        create_friend_tree=create_friend_tree,
        scale_factor=scale_factor,
        selected_grooming_methods=selected_grooming_methods,
        draw_example_splittings_to_filename=draw_example_splittings_to_filename,
    )

    # Then handle the I/O once finished
    logger.info(f"Finished processing tree from file {input_filename}")
    _write_skim_output(
        grooming_results=grooming_results,
        output_filename=output_filename,
        write_root=write_root,
        output_tree_name=output_tree_name,
        write_parquet=write_parquet,
        write_feather=write_feather,
        create_friend_tree=create_friend_tree,
    )
    return True, output_filename, "processed"


def _calculate_embedding_skim(
    input_filename: Path,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    scale_factors: Mapping[int, float],
    train_directory: Path,
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    draw_example_splittings: bool = False,
    write_feather: bool = False,
    write_parquet: bool = False,
    selected_grooming_methods: list[str] | None = None,
) -> tuple[bool, Path, str]:
    """Entry point for calculating the embedding skim.

    OBSOLETE! (practically - as part of mammoth v1)
    """
    # Validation
    # Bail out early if the file already exists.
    if output_filename.exists():
        return True, output_filename, "already exists"

    # Setup
    # Use the train configuration to extract the train number and pt hard bin, which are used to get the scale factor.
    y = yaml.yaml()
    with (train_directory / "config.yaml").open() as f:
        train_config = y.load(f)
    train_number = train_config["number"]
    pt_hard_bin = train_config["pt_hard_bin"]
    logger.debug(f"Extracted train number: {train_number}, pt hard bin: {pt_hard_bin}")
    scale_factor = scale_factors[pt_hard_bin]

    # Jets setup.
    logger.info(f"Skimming tree from file {input_filename}")
    all_jets = analysis_jet_substructure.parquet_to_substructure_analysis(filename=input_filename, prefixes=prefixes)

    return _calculate_embedding_skim_mammoth_framework_v1(
        all_jets=all_jets,
        input_filename=input_filename,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        scale_factor=scale_factor,
        jet_R=jet_R,
        output_filename=output_filename,
        output_tree_name=output_tree_name,
        create_friend_tree=create_friend_tree,
        draw_example_splittings_to_filename=output_filename.parent if draw_example_splittings else None,
        write_feather=write_feather,
        write_parquet=write_parquet,
        selected_grooming_methods=selected_grooming_methods,
    )


def _calculate_data_skim_mammoth_framework_v1(
    all_jets: ak.Array,
    input_filename: Path,
    collision_system: str,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    scale_factors: Mapping[int, float] | None = None,
    write_root: bool = True,
    write_feather: bool = False,
    write_parquet: bool = False,
    selected_grooming_methods: list[str] | None = None,
) -> tuple[bool, Path, str]:
    """Calculate the data skim using the mammoth v1 framework.

    OBSOLETE! (practically - as part of mammoth v1)
    """
    # First, directly run the skim
    grooming_results = calculate_one_or_two_input_level_skim_impl(
        all_jets=all_jets,
        collision_system=collision_system,
        prefixes=prefixes,
        iterative_splittings=iterative_splittings,
        jet_R=jet_R,
        create_friend_tree=create_friend_tree,
        scale_factors=scale_factors,
        selected_grooming_methods=selected_grooming_methods,
    )

    # Then handle the I/O once finished
    logger.info(f"Finished processing tree from file {input_filename}")
    _write_skim_output(
        grooming_results=grooming_results,
        output_filename=output_filename,
        write_root=write_root,
        output_tree_name=output_tree_name,
        write_parquet=write_parquet,
        write_feather=write_feather,
        create_friend_tree=create_friend_tree,
    )
    return True, output_filename, "processed"


def _calculate_data_skim(
    input_filename: Path,
    collision_system: str,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    scale_factors: Mapping[int, float] | None = None,
    write_feather: bool = False,
    write_parquet: bool = False,
    selected_grooming_methods: list[str] | None = None,
) -> tuple[bool, Path, str]:
    """Entry point for calculating the embedding skim.

    OBSOLETE!
    """
    # Validation
    if scale_factors is None and collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
        _msg = f"Need scale factors for {collision_system} to be provided externally."
        raise ValueError(_msg)
    # Bail out early if the file already exists.
    if output_filename.exists():
        return True, output_filename, "already exists"

    # Jets setup
    logger.info(f"Skimming tree from file {input_filename}")
    # Careful, this can return general columns, not just jets in prefixes (for example, the pt_hard in pythia)
    all_jets = analysis_jet_substructure.parquet_to_substructure_analysis(filename=input_filename, prefixes=prefixes)

    return _calculate_data_skim_mammoth_framework_v1(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system=collision_system,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        jet_R=jet_R,
        output_filename=output_filename,
        output_tree_name=output_tree_name,
        create_friend_tree=create_friend_tree,
        scale_factors=scale_factors,
        write_feather=write_feather,
        write_parquet=write_parquet,
        selected_grooming_methods=selected_grooming_methods,
    )


if __name__ == "__main__":
    # An example for testing...
    from mammoth import helpers

    helpers.setup_logging()
    # res = _calculate_embedding_skim(
    #     input_filename=Path(
    #         "trains/embedPythia/6650/parquet/events_per_job_100000/AnalysisResults.18q.repaired.00.parquet"
    #     ),
    #     iterative_splittings=True,
    #     prefixes={"hybrid": "data", "true": "matched", "det_level": "detLevel"},
    #     scale_factors={1: 16.0695},
    #     train_directory=Path("trains/embedPythia/6650/"),
    #     jet_R=0.4,
    #     output_filename=Path(
    #         "trains/embedPythia/6650/skim/test/AnalysisResults.18q.repaired.00_iterative_splittings.root"
    #     ),
    #     write_parquet=True,
    #     write_feather=True,
    # )
    # Skim data.
    collision_system = "pythia"
    train_number = 2461
    res = _calculate_data_skim(
        input_filename=Path(
            f"trains/{collision_system}/{train_number}/parquet/events_per_job_200000/AnalysisResults.cent_woSDD.01.repaired.00.parquet"
        ),
        collision_system=collision_system,
        iterative_splittings=True,
        prefixes={
            "data": "data",
            "true": "matched",
        },
        # These are wrong, but we need to simulate all of them being available for testing.
        scale_factors={pt_hard_bin: 16.0695 for pt_hard_bin in range(1, 21)},
        jet_R=0.2,
        output_filename=Path(
            f"trains/{collision_system}/{train_number}/skim/test/AnalysisResults.cent_woSDD.01.repaired.00_iterative_splittings.root"
        ),
        write_parquet=True,
        write_feather=True,
    )
    logger.info(res)
    # import IPython; IPython.start_ipython(user_ns=locals())
