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
    source_index_identifiers: Mapping[str, int],
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
    det_level = arrays["signal"]["det_level"]
    det_level["index"] = ak.local_index(det_level) + source_index_identifiers["signal"]
    det_level["m"] = det_level["pt"] * mass_hypothesis
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    part_level = arrays["signal"]["part_level"]
    part_level["index"] = ak.local_index(part_level) + source_index_identifiers["signal"]
    part_level["m"] = part_level["pt"] * mass_hypothesis
    part_level = vector.Array(part_level)
    background = arrays["background"]
    background["index"] = ak.local_index(background) + source_index_identifiers["background"]
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
    arrays = _transform_inputs(source_index_identifiers=source_index_identifiers, arrays=arrays)
    logger.info("Find jets")

    #logger.info("Part level start")
    #t = time.time()
    #part_level = jet_finding.find_jets(
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
                background_subtraction=True,
                # TODO: The min jet pt cut should be applied _after_ subtraction...
                #       Although I guess it also doesn't hurt to apply it before, since
                #       the subtraction always will lower the overall jet pt.
                # TODO: Apply the selector
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
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level]
        )
    logger.info("Done with reclustering")

    #import IPython; IPython.embed()


if __name__ == "__main__":
    analysis()
