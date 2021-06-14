"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

from mammoth.framework import sources

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


if __name__ == "__main__":
    source_index_identifiers, arrays = embed_into_thermal_model_data(
        pythia_filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.parquet")
    )
    arrays = _transform_inputs(source_index_identifiers=source_index_identifiers, arrays=arrays)

    import IPython; IPython.embed()
