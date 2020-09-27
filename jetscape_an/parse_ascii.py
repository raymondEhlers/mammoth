#!/usr/bin/env python3

""" Tests for parsing JETSCAPE.

"""

import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

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
    def read_events_in_chunks(filename: Path, events_per_file: int = 1e5):
        """
        We define the closure here so that we have access to the event_split_index outside of everything else.
        """
        # We keep track of the location of where to split each event.
        # That way, we can come back later and split the 2D numpy array into an awkward array with a jagged structure.
        #event_split_index = []

        header_regex = re.compile(r"\d+")
        def _handle_line(line: str, n_events: int) -> Tuple[bool, Optional[Any]]:
            """ Parse line as appropriate.

            If it's just a standard line, we just pass it on. However, if it's a header, we parse the
            header, as well as perform both an event count check. This way, we can split files without
            having to worry about getting only part of an event (which would happen if we just trivially
            split on lines).

            Note:
                We don't even have to yield the line back because we don't ever modify it.

            Args:
                line: Line to be parsed.
                n_events: Number of events processed so far.
            Returns:
                Whether we've reached the desired number of events and should stop this block, any information parsed from the header.
                The header info is None if it's not a header line.
            """
            #print(f"{line_count}: line: {line}")
            time_to_stop = False
            header_info = None
            if line.startswith("#"):
                # Need to check if event_split_index so that we get past 0...
                if n_events > 0 and n_events % events_per_file == 0:
                    print("Time to stop!")
                    time_to_stop = True

                # TODO: We can parse the header for additional info if so inclined.
                #       As of the first prototype, it's not yet necessary.
                #       For now, we just indicated that we found a header by setting it to True.
                #       We just need something to differentiate it from not finding a header.
                header_info = [int(s) for s in re.findall(header_regex, line)[1:]]
                # Since this line will be skipped by loadtxt, we need to account for that
                # but subtracting the number of events so far.
                #event_split_index.append(line_count - len(event_split_index))

            return time_to_stop, header_info

        # Keep track of the number of lines by hand because we can increment the iterator from multiple places.
        line_count = 0

        with open(filename, "r") as f:
            read_lines = iter(f.readlines())
            #read_lines = f.readline()
            return_count = 0
            keep_header_for_next_iter = None
            for line in read_lines:
                line_count = 0
                event_split_index = []
                print(f"return_count: {return_count}, keep_header_for_next_iter: {keep_header_for_next_iter}")
                #event_header_info = [keep_header_for_next_iter] if keep_header_for_next_iter else []
                event_header_info = []
                n_events = 0
                if keep_header_for_next_iter:
                    event_header_info.append(keep_header_for_next_iter)
                    n_events += 1
                    keep_header_for_next_iter = None
                def _inner():
                    nonlocal line
                    nonlocal line_count
                    nonlocal keep_header_for_next_iter
                    nonlocal n_events
                    _, header_info = _handle_line(line, n_events)
                    yield line
                    line_count += 1
                    if header_info:
                        n_events += 1
                        event_header_info.append(header_info)
                    # Handle additional lines
                    for local_line in read_lines:
                        time_to_stop, header_info = _handle_line(local_line, n_events)
                        line_count += 1
                        if header_info:
                            n_events += 1
                            if not time_to_stop:
                                event_header_info.append(header_info)
                                event_split_index.append(line_count - len(event_split_index))
                            else:
                                keep_header_for_next_iter = header_info
                                print(f"header_info: {header_info}, keep_header_for_next_iter: {keep_header_for_next_iter}")
                        yield local_line
                        if time_to_stop:
                            print(f"event_split_index len: {len(event_split_index)} - {event_split_index}")
                            break

                yield _inner(), event_split_index, event_header_info
                #line_count += 1
                return_count += 1
                print(f"line_count: {line_count}, return_count: {return_count}")
                print(f"keep_header_for_next_iter: {keep_header_for_next_iter}")
                #yield i, line
                #if return_count > 2:
                #    return
            #print(f"return_count: {return_count}")

        #return wrap_reading_file

    #import IPython; IPython.embed()

    #event_split_index_offset = 0
    count = 0
    for chunk_generator, event_split_index, event_header_info in read_events_in_chunks(filename=filename, events_per_file=5):
        hadrons = np.loadtxt(chunk_generator)
        #temp_arr = ak.Array(np.split(hadrons, event_split_index[event_split_index_offset + 1:]))
        temp_arr = ak.Array(np.split(hadrons, event_split_index))
        print(len(event_split_index))
        print(f"hadrons: {hadrons}")
        print(f"temp_arr: {temp_arr}")
        print(ak.type(temp_arr))
        print(f"Event header info: {event_header_info}")
        #event_split_index_offset = len(event_split_index) - 1
        #import IPython; IPython.embed()

        # TEMP
        #if count > 2:
        #    break
        #count += 1
        # ENDTEMP

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

