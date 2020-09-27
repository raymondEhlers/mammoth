#!/usr/bin/env python3

""" Tests for parsing JETSCAPE.

"""

from pathlib import Path
from typing import Tuple

import awkward1 as ak
import numpy as np

def parse_final_state_hadrons(filename: Path) -> np.ndarray:

    b = ak.ArrayBuilder()

    event = None
    columns = ["E", "px", "py", "pz", "eta", "phi"]
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                # Parse event info.
                #   0   Event1ID    240 pstat-EPx   Py  Pz  Eta Phi
                "  0   Event{event_ID}ID    {hybrid_id}"
                ...
                if event:
                    b.append(event)
                event = []
                continue
            else:
                #particle_index, particle_ID, pstat, E, px, py, pz, eta, phi = line.split(" ")
                event.append(np.array(line.split(" "), dtype=np.float64))


def jetscape_array() -> ak.Array:
    b = ak.ArrayBuilder()
    b.begin_record()
    with b.list():
        b.integer()

    # Desired output:

    ak.zip({
        "E": [
            [1,2,3],
            [4,5,6,7],
        ],
        "px": [
            [7,6,4],
            [3,4,5,6],
        ],
        #...
    })


def read(filename: Path):
    # We keep track of the location of where to split each event.
    # That way, we can come back later and split the 2D numpy array into an awkward array with a jagged structure.
    event_split_index = []

    def read_events_in_chunks(filename: Path, events_per_file: int = 1e5):
        """

        """

        def _handle_line(line: str, line_count: int) -> Tuple[str, bool]:
            print(f"{line_count}: line: {line}")
            time_to_stop = False
            if line.startswith("#"):
                # Need to check if event_split_index so that we get past 0...
                if event_split_index and len(event_split_index) % events_per_file == 0:
                    print("Time to stop!")
                    time_to_stop = True
                # Since this line will be skipped by loadtxt, we need to account for that
                # but subtracting the number of events so far.
                event_split_index.append(line_count - len(event_split_index))

            return line, time_to_stop

        # Keep track of the number of lines by hand because we can increment from multiple places.
        line_count = 0

        with open(filename, "r") as f:
            read_lines = iter(f.readlines())
            #read_lines = f.readline()
            return_count = 0
            for line in read_lines:
                def _inner():
                    nonlocal line
                    nonlocal line_count
                    yield _handle_line(line, line_count)
                    line_count += 1
                    # Handle additional lines
                    for local_line in read_lines:
                        l, time_to_stop = _handle_line(local_line, line_count)
                        line_count += 1
                        yield l
                        if time_to_stop:
                            print(f"event_split_index len: {len(event_split_index)} - {event_split_index}")
                            break
                        #if line_count > 5:
                        #    break
                        #if line.startswith("#"):
                        #    # Need to check if event_split_index so that we get past 0...
                        #    if event_split_index and len(event_split_index) % events_per_file == 0:
                        #        time_to_stop = True
                        #    # Since this line will be skipped by loadtxt, we need to account for that
                        #    # but subtracting the number of events so far.
                        #    event_split_index.append(line_count - len(event_split_index))

                        #yield line
                        #line_count += 1
                        #if time_to_stop:
                        #    return

                yield _inner()
                #line_count += 1
                return_count += 1
                print(f"line_count: {line_count}, return_count: {return_count}")
                #yield i, line
                #if return_count > 2:
                #    return
            print(f"return_count: {return_count}")

        #return wrap_reading_file

    import IPython; IPython.embed()

    event_split_index_offset = 0
    for chunk_generator in read_events_in_chunks(filename=filename, events_per_file=5):
        hadrons = np.loadtxt(chunk_generator)
        temp_arr = ak.Array(np.split(hadrons, event_split_index[event_split_index_offset + 1:]))
        print(len(event_split_index))
        print(hadrons)
        print(temp_arr.shape)
        event_split_index_offset = len(event_split_index) - 1

    # NOTE: Can parse / store more information here if it's helpful.
    def yield_final_state(filename: Path):
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()):
                if line.startswith("#"):
                    # Since this line will be skipped by loadtxt, we need to account for that
                    # but subtracting the number of events so far.
                    event_split_index.append(i - len(event_split_index))
                yield line

    #hadrons = np.loadtxt(yield_final_state(filename=filename))

    # We have to take [1:] because we need to skip the first line. Otherwise,
    # we used up with an empty first event.
    temp_arr = ak.Array(np.split(hadrons, event_split_index[1:]))

    array = ak.zip({
        # TODO: Does the conversion add any real computation time?
        "pid": ak.values_astype(temp_arr[:, :, 1], np.int32),
        # 2 == "status". Which I think is always 0 because we're looking at final state particles.
        "E": temp_arr[:, :, 3],
        "px": temp_arr[:, :, 4],
        "py": temp_arr[:, :, 5],
        "pz": temp_arr[:, :, 6],
    })
    # Test
    array = ak.zip({
        # TODO: Does the conversion add any real computation time?
        "pid": ak.values_astype(temp_arr[:, :, 1], np.int32),
        "E": ak.values_astype(temp_arr[:, :, 3], np.float32),
        "px": ak.values_astype(temp_arr[:, :, 4], np.float32),
        "py": ak.values_astype(temp_arr[:, :, 5], np.float32),
        "pz": ak.values_astype(temp_arr[:, :, 6], np.float32),
    })

    # Parquet doesn't appear to save space...
    # We do save space by converting types and dropping useless columns
    ak.to_parquet(array, "test.parquet")

    #import IPython; IPython.embed()


if __name__ == "__main__":
    read(filename="final_state_hadrons.dat")

