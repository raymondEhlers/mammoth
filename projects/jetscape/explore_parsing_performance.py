""" Tests related to parsing

They're not really up to unit tests yet - this is just for me to take a look at performance

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import timeit
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from mammoth.framework.io import _jetscape_parser


def test_read(filename: str | Path, events_per_chunk: int, max_chunks: int = 1) -> None:
    # Validation
    filename = Path(filename)

    # Compare against the known result to ensure that it's working correctly!
    ref = None
    for loader in ["np", "pandas", "python"]:
        for i, chunk_generator in enumerate(
            _jetscape_parser.read_events_in_chunks(filename=filename, events_per_chunk=events_per_chunk)
        ):
            # Bail out if we've done enough.
            if i == max_chunks:
                break

            if loader == "np":
                start_time = timeit.default_timer()
                hadrons = np.loadtxt(chunk_generator)
            elif loader == "pandas":
                start_time = timeit.default_timer()
                hadrons = pd.read_csv(
                    _jetscape_parser.FileLikeGenerator(iter(chunk_generator)),
                    names=["particle_index", "particle_ID", "status", "E", "px", "py", "pz", "eta", "phi"],
                    skiprows=[0],
                    header=None,
                    comment="#",
                    sep=r"\s+",
                    # Converting to numpy makes the dtype conversion moot.
                    # dtype={
                    #    "particle_index": np.int32, "particle_ID": np.int32, "status": np.int8,
                    #    "E": np.float32, "px": np.float32, "py": np.float32, "pz": np.float32,
                    #    "eta": np.float32, "phi": np.float32
                    # },
                    # We can reduce columns to save a little time reading.
                    # However, it makes little difference, and makes it less general. So we disable it for now.
                    # usecols=["particle_ID", "status", "E", "px", "py", "eta", "phi"],
                ).to_numpy()
                # NOTE: It's important that we convert to numpy before splitting. Otherwise, it will return columns names,
                #       which will break the indexing and therefore the conversion.
            elif loader == "python":
                # Python only solution because np.loadtxt isn't actually very fast...
                start_time = timeit.default_timer()
                particles = []
                for p in chunk_generator:
                    if not p.startswith("#"):
                        particles.append(np.array(p.rstrip("\n").split(), dtype=np.float64))
                hadrons = np.stack(particles)
            else:
                _msg = f"Unrecognized loader '{loader}'"
                raise ValueError(_msg)

            elapsed = timeit.default_timer() - start_time
            print(f"Loading {events_per_chunk} events with {loader}: {elapsed}")

            array_with_events = ak.Array(np.split(hadrons, chunk_generator.event_split_index()))
            if ref is None:
                ref = array_with_events

            # if loader == "pandas":
            #    import IPython; IPython.embed()

            assert array_with_events == ref


if __name__ == "__main__":
    for pt_hat_range in ["7_9", "20_25", "50_55", "100_110", "250_260", "500_550", "900_1000"]:
        filename = f"JetscapeHadronListBin{pt_hat_range}"
        directory_name = "5020_PbPb_0-10_0R25_1R0_1"
        full_filename = f"../phys_paper/AAPaperData/{directory_name}/{filename}.out"
        test_read(filename=full_filename, events_per_chunk=100, max_chunks=1)
