"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

from mammoth.framework import sources


#def embed_into_thermal_model_data(pythia_filename: Path, chunk_size: int = 1000) -> None:
def embed_into_thermal_model_data(pythia_filename: Path, chunk_size: int = 18681) -> None:
    # Signal
    pythia_source = sources.ParquetSource(
        filename=pythia_filename,
    )
    # Background
    # We'll generate in chunks. They may not exactly match the pythia, but in that case,
    # it will just keep generating.
    thermal_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.ThermalBackgroundExponential(
            chunk_size=chunk_size,
            n_particles_per_event_mean=2500,
            n_particles_per_event_sigma=500,
            pt_exponential_scale=0.4
        ),
        repeat=True
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        sources={"signal": pythia_source, "background": thermal_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    return combined_source.data()

if __name__ == "__main__":
    data = embed_into_thermal_model_data(pythia_filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.parquet"))

    import IPython; IPython.embed()
