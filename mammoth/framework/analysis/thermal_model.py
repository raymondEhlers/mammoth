"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

import awkward as ak

from mammoth.framework import sources


#def embed_into_thermal_model_data(pythia_filename: Path, chunk_size: int = 1000) -> None:
def embed_into_thermal_model_data(pythia_filename: Path) -> ak.Array:
    # Signal
    pythia_source = sources.ParquetSource(
        filename=pythia_filename,
    )
    # Background
    # We'll generate in chunks. They may not exactly match the pythia, but in that case,
    # it will just keep generating.
    thermal_source = sources.ThermalModelExponential(
        # Chunk sizee will be set when combining the sources.
        chunk_size=-1,
        n_particles_per_event_mean=2500,
        n_particles_per_event_sigma=500,
        pt_exponential_scale=0.4
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"signal": pythia_source},
        chunked_sources={"background": thermal_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    return combined_source.data()


if __name__ == "__main__":
    arrays = embed_into_thermal_model_data(pythia_filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.parquet"))

    import IPython; IPython.embed()
