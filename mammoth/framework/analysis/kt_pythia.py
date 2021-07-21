"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

from mammoth.framework import jet_finding, sources


logger = logging.getLogger(__name__)
vector.register_awkward()


def _transform_inputs(
    arrays: ak.Array,
    mass_hypothesis: float = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    # Setup
    if not particle_columns:
        particle_columns = {
            "px": np.float32,
            "py": np.float32,
            "pz": np.float32,
            "E": np.float32,
            "index": np.int64,
        }

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays["det_level"]
    det_level["index"] = ak.local_index(det_level)
    det_level["m"] = det_level["pt"] * mass_hypothesis
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    part_level = arrays["part_level"]
    part_level["index"] = ak.local_index(part_level)
    part_level["m"] = part_level["pt"] * mass_hypothesis
    part_level = vector.Array(part_level)

    # Combine inputs
    logger.debug("Embedding...")
    return ak.Array(
        {
            "part_level": part_level,
            "det_level": det_level,
            # Include the rest of the non particle related fields (ie. event level info)
            **{
                k: v
                for k, v in zip(ak.fields(arrays), ak.unzip(arrays))
                if k not in ["det_level", "part_level"]
            },
        }
    )


def load_data(
    pythia_filename: Path,
) -> ak.Array:
    # Signal
    pythia_source = sources.ParquetSource(
        filename=pythia_filename,
    )
    return pythia_source.data()


def setup_logging(level: int = logging.DEBUG) -> None:
    # Basic setup
    logging.basicConfig(level=level, format="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)
    # Quiet down numba
    logging.getLogger("numba").setLevel(logging.INFO)


def analysis(jet_R: float = 0.4, min_pythia_jet_pt: float = 20.0) -> None:
    setup_logging(level=logging.INFO)
    logger.info("Start")
    arrays = load_data(
        pythia_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet")
    )
    logger.info("Transform")
    arrays = _transform_inputs(arrays=arrays)

    # Track cuts
    logger.info("Track level cuts")
    # Detector level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    det_track_pt_mask = (arrays["det_level"].pt >= 0.150)
    arrays["det_level"] = arrays["det_level"][det_track_pt_mask]

    # Jet finding
    logger.info("Find jets")
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                min_jet_pt=min_pythia_jet_pt,
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                min_jet_pt=min_pythia_jet_pt,
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    # **************
    # Remove detector level jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We skip at part level because it doesn't share these detector effects.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    det_level_max_track_pt_cut = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = jet_finding.area_percentage(60, jet_R),
    part_level_min_area_mask = jets["part_level", "area"] > min_area
    det_level_min_area_mask = jets["det_level", "area"] > min_area

    # Apply the cuts
    jets["part_level"] = jets["part_level"][
        part_level_min_area_mask
    ]
    jets["det_level"] = jets["det_level"][
        det_level_max_track_pt_cut
        & det_level_min_area_mask
    ]

    logger.info("Matching jets")
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=0.4,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, and require that there
    # is a valid match.
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (ak.num(jets["part_level"], axis=1) > 0) & (ak.num(jets["det_level"], axis=1) > 0)
    jets = jets[jets_present_mask]
    # Now, onto the individual jet collections
    # Require valid matched jet indices.
    part_level_matched_jets_mask = jets["part_level"]["matching"] > -1
    jets["part_level"] = jets["part_level"][part_level_matched_jets_mask]
    det_level_matched_jets_mask = jets["det_level"]["matching"] > -1
    jets["det_level"] = jets["det_level"][det_level_matched_jets_mask]

    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level"]:
        logger.info(f"Reclustering {level}")
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level]
        )
    logger.info("Done with reclustering")

    import IPython; IPython.embed()


if __name__ == "__main__":
    analysis()
