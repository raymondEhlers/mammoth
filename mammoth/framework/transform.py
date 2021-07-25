""" Collection of transforms for sources.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from typing import Mapping, Optional, Union

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

logger = logging.getLogger(__name__)


_default_particle_columns = {
    "px": np.float32,
    "py": np.float32,
    "pz": np.float32,
    "E": np.float32,
    "index": np.int64,
}


def data(
    arrays: ak.Array,
    rename_prefix: Mapping[str, str] = None,
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    """Transform into a form appropriate for data analysis.

    Args:
        arrays: Input arrays
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!  Default: "data" -> "data".
        mass_hypothesis: Mass hypothesis for the prefixes, or individually. Default: 0.139 GeV
            for all particle collections. (Same interface here even though we expect only one particle collection).
        particle_columns: Dtypes for particle columns (unused as of July 2021).

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    if rename_prefix is None:
        rename_prefix = {"data": "data"}
    _prefixes = list(rename_prefix.keys())
    if isinstance(mass_hypothesis, float):
        mass_hypotheses = {p: mass_hypothesis for p in _prefixes}
    else:
        mass_hypotheses = mass_hypothesis

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    data = arrays[rename_prefix["data"]]
    data["index"] = ak.local_index(data)
    if "m" not in ak.fields(data):
        data["m"] = data["pt"] * 0 + mass_hypotheses["data"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> data = ak.with_name(data, name="Momentum4D")
    data = vector.Array(data)

    # Combine inputs
    return ak.Array(
        {
            "data": data,
            # Include the rest of the non particle related fields (ie. event level info)
            **{k: v for k, v in zip(ak.fields(arrays), ak.unzip(arrays)) if k not in _prefixes},
        }
    )


def mc(
    arrays: ak.Array,
    rename_prefix: Mapping[str, str] = None,
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    """Transform into a form appropriate for MC analysis.

    Args:
        arrays: Input arrays
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!  Default: "part_level" -> "part_level", "det_level" -> "det_level".
        mass_hypothesis: Mass hypothesis for either all three prefixes, or individually. Default: 0.139 GeV
            for all particle collections.
        particle_columns: Dtypes for particle columns (unused as of July 2021).

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    if rename_prefix is None:
        rename_prefix = {
            "part_level": "part_level",
            "det_level": "det_level",
        }
    _prefixes = list(rename_prefix.keys())
    if isinstance(mass_hypothesis, float):
        mass_hypotheses = {p: mass_hypothesis for p in _prefixes}
    else:
        mass_hypotheses = mass_hypothesis

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays[rename_prefix["det_level"]]
    det_level["index"] = ak.local_index(det_level)
    if "m" not in ak.fields(det_level):
        det_level["m"] = det_level["pt"] * 0 + mass_hypotheses["det_level"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    part_level = arrays[rename_prefix["part_level"]]
    part_level["index"] = ak.local_index(part_level)
    if "m" not in ak.fields(part_level):
        part_level["m"] = part_level["pt"] * 0 + mass_hypotheses["part_level"]
    part_level = vector.Array(part_level)

    # Combine inputs
    return ak.Array(
        {
            "part_level": part_level,
            "det_level": det_level,
            # Include the rest of the non particle related fields (ie. event level info)
            **{k: v for k, v in zip(ak.fields(arrays), ak.unzip(arrays)) if k not in _prefixes},
        }
    )


def embedding(
    arrays: ak.Array,
    source_index_identifiers: Mapping[str, int],
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    """Transform into a form appropriate for embedding.

    Note:
        This performs embedding in the process of transforming.

    Args:
        arrays: Input arrays
        source_index_identifiers: Index offset map for each source.
        mass_hypothesis: Mass hypothesis for either all three prefixes, or individually. Default: 0.139 GeV
            for all particle collections.
        particle_columns: Dtypes for particle columns (unused as of July 2021).

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    _mass_hypothesis_prefixes = ["part_level", "det_level", "background"]
    if isinstance(mass_hypothesis, float):
        mass_hypotheses = {p: mass_hypothesis for p in _mass_hypothesis_prefixes}
    else:
        mass_hypotheses = mass_hypothesis

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays["signal"]["det_level"]
    det_level["index"] = ak.local_index(det_level) + source_index_identifiers["signal"]
    if "m" not in ak.fields(det_level):
        det_level["m"] = det_level["pt"] * 0 + mass_hypotheses["det_level"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    part_level = arrays["signal"]["part_level"]
    part_level["index"] = ak.local_index(part_level) + source_index_identifiers["signal"]
    if "m" not in ak.fields(part_level):
        part_level["m"] = part_level["pt"] * 0 + mass_hypotheses["part_level"]
    part_level = vector.Array(part_level)
    background = arrays["background"]
    background["index"] = ak.local_index(background) + source_index_identifiers["background"]
    if "m" not in ak.fields(background):
        background["m"] = background["pt"] * 0 + mass_hypotheses["background"]
    background = vector.Array(background)

    # Combine inputs
    logger.debug("Embedding...")
    return ak.Array(
        {
            "part_level": part_level,
            "det_level": det_level,
            # Practically, this is where we are performing the embedding
            # Need to rezip so it applies the vector at the same level as the other collections
            # (ie. we want `var * Momentum4D[...]`, but without the zip, we have `Momentum4D[var * ...]`)
            # NOTE: For some reason, ak.concatenate returns float64 here. I'm not sure why, but for now
            #       it's not diving into.
            "hybrid": vector.zip(
                dict(
                    zip(
                        particle_columns.keys(),
                        ak.unzip(
                            ak.concatenate(
                                [
                                    ak.Array({k: getattr(det_level, k) for k in particle_columns}),
                                    ak.Array({k: getattr(background, k) for k in particle_columns}),
                                ],
                                axis=1,
                            )
                        ),
                    )
                )
            ),
            # Include the rest of the non particle related fields (ie. event level info)
            **{
                k: v
                for k, v in zip(ak.fields(arrays["signal"]), ak.unzip(arrays["signal"]))
                if k not in ["det_level", "part_level"]
            },
        }
    )