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

from mammoth.framework import jet_finding, sources, transform


logger = logging.getLogger(__name__)
vector.register_awkward()


def embed_into_thermal_model_data(
    pythia_filename: Path,
) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = sources.ParquetSource(
        filename=pythia_filename,
    )
    # Background
    thermal_source = sources.ThermalModelExponential(
        # Chunk sizee will be set when combining the sources.
        chunk_size=-1,
        n_particles_per_event_mean=2500,
        n_particles_per_event_sigma=500,
        # TEMP: Make smaller to speed up
        # n_particles_per_event_mean=100,
        # n_particles_per_event_sigma=5,
        # ENDTEMP
        pt_exponential_scale=0.4,
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"signal": pythia_source},
        chunked_sources={"background": thermal_source},
        source_index_identifiers=source_index_identifiers,
    )

    return source_index_identifiers, combined_source.data()


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


def analysis(jet_R: float = 0.2, min_hybrid_jet_pt: float = 15.0, min_pythia_jet_pt: float = 5.0) -> None:
    setup_logging(level=logging.INFO)
    logger.info("Start")
    source_index_identifiers, arrays = embed_into_thermal_model_data(
        pythia_filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.parquet")
    )
    logger.info("Transform")
    arrays = transform.embedding(
        arrays=arrays,
        source_index_identifiers=source_index_identifiers,
        mass_hypothesis={
            "part_level": 0.139,
            "det_level": 0.139,
            "background": 0.0,
        },
    )
    logger.info("Find jets")

    # TEMP: Reduce number of events to speed calculations
    # arrays = arrays[:500]
    # ENDTEMP

    # logger.info("Part level start")
    # t = time.time()
    # part_level = jet_finding.find_jets(
    #    particles=arrays["part_level"], algorithm="anti-kt", jet_R=jet_R, min_jet_pt=min_pythia_jet_pt,
    #    area_settings=jet_finding.AREA_PP,
    # )
    # logger.info(f"Done with part level. Time: {time.time() - t}")
    # import IPython; IPython.embed()
    # return part_level
    # t = time.time()
    # det_level = jet_finding.find_jets(
    #    particles=arrays["det_level"], algorithm="anti-kt", jet_R=jet_R, min_jet_pt=min_pythia_jet_pt,
    # )
    # logger.info(f"Done with det level. Time: {time.time() - t}")
    # t = time.time()
    # hybrid = jet_finding.find_jets(
    #    particles=arrays["hybrid"], algorithm="anti-kt", jet_R=jet_R, min_jet_pt=min_hybrid_jet_pt,
    # )
    # logger.info(f"Done with hybrid level. Time: {time.time() - t}")
    # jets = ak.zip(
    #    {
    #        "part_level": part_level,
    #        "det_level": det_level,
    #        "hybrid": hybrid,
    #    },
    #    depth_limit=1,
    # )

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
            "hybrid": jet_finding.find_jets(
                particles=arrays["hybrid"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_AA,
                min_jet_pt=min_hybrid_jet_pt,
                constituent_subtraction=jet_finding.ConstituentSubtractionSettings(r_max=0.25),
            ),
        },
        depth_limit=1,
    )

    logger.info("Matching jets")
    jets["det_level", "matching"], jets["hybrid", "matching"] = jet_finding.jet_matching(
        jets_base=jets["det_level"],
        jets_tag=jets["hybrid"],
        max_matching_distance=0.4,
    )
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=0.4,
    )
    # Semi-validated result:
    # det <-> part for the thermal model looks like:
    # part: ([[0, 3, 1, 2, 4, 5], [0, 1, -1], [], [0], [1, 0, -1]],
    # det:   [[0, 2, 3, 1, 4, 5], [0, 1], [], [0], [1, 0]])

    # TODO: Use matching info!

    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level", "hybrid"]:
        logger.info(f"Reclustering {level}")
        jets[level, "reclustering"] = jet_finding.recluster_jets(jets=jets[level])
    logger.info("Done with reclustering")

    # import IPython; IPython.embed()


if __name__ == "__main__":
    analysis()
