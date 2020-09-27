#!/usr/bin/env python3

""" Tests for parsing JETSCAPE.

"""

import logging
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union, Tuple

import awkward1 as ak
import numpy as np


logger = logging.getLogger(__name__)


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


_header_regex = re.compile(r"\d+")


def _handle_line(line: str, n_events: int, events_per_chunk: int) -> Tuple[bool, Optional[Any]]:
    """ Parse line as appropriate.

    If it's just a standard particle line, we just pass it on. However, if it's a header, we parse
    the header, as well as perform both an event count check. This way, we can split files without
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
    time_to_stop = False
    header_info = None
    if line.startswith("#"):
        # We've found a header line.

        # We need to be able to chunk the files into something smaller and more manageable.
        # Therefore, when we hit the target, we provide a signal that it's time to end the block.
        # This is a bit awkward because we don't know that an event has ended until we get to the
        # next event. However, we check for exact agreement because we don't increment the number
        # of events until after we find a header. Basically, it all works out here.
        #if n_events > 0 and n_events % events_per_chunk == 0:
        if n_events == events_per_chunk:
            logger.debug("Time to stop!")
            time_to_stop = True

        # Parse the header string.
        # As of September 2020, the formatting isn't really right. This should be fixed in JS.
        # Due to this formatting issue:
        # - We ignore all of the column names.
        # - We only parse the numbers:
        #   1. I'm not sure what this number means
        #   2. Event number. int
        #   3. Hydro ID. int.
        #
        # For now, we don't construct any objects to contain the information because
        # it's not worth the computing time - we're not really using this information...
        header_info = [int(s) for s in re.findall(_header_regex, line)[1:]]

    return time_to_stop, header_info


def read(filename: Union[Path, str]):
    # Validation
    filename = Path(filename)

    def read_events_in_chunks(filename: Union[Path, str], events_per_chunk: int = int(1e5)):
        """ Read events in chunks from stored JETSCAPE ASCII files.

        Args:
            filename: Path to the file.
            events_per_chunk: Number of events to store in each chunk. Default: 1e5.
        Returns:
            Chunks generator. When this generator is consumed, it will generate lines from the file
                until it hits the number of events mark.
        """
        # Validation
        filename = Path(filename)

        # We keep track of the location of where to split each event.
        # That way, we can come back later and split the 2D numpy array into an awkward array with a jagged structure.
        #event_split_index = []

        with open(filename, "r") as f:
            # Setup
            # This is just for convenience.
            return_count = 0
            # This is used to pass a header to the next chunk. This is necessary because we don't know an event
            # is over until we already get the header for the next event. We could keep that line and reparse,
            # but there's no need to parse a file twice.
            keep_header_for_next_chunk = None

            # Define an iterator so we can increment it in different locations in the code.
            read_lines = iter(f.readlines())
            #read_lines = f.readline()

            for line in read_lines:
                # Setup
                # Keep track of the number of lines by hand because we can increment the iterator from multiple places.
                line_count = 0
                event_split_index: List[int] = []
                print(f"return_count: {return_count}, keep_header_for_next_chunk: {keep_header_for_next_chunk}")
                #event_header_info = [keep_header_for_next_chunk] if keep_header_for_next_chunk else []
                event_header_info = []
                if keep_header_for_next_chunk:
                    event_header_info.append(keep_header_for_next_chunk)
                    keep_header_for_next_chunk = None

                def _inner(kept_header: bool) -> Iterable[str]:
                    """

                    """
                    # 
                    nonlocal line
                    nonlocal line_count
                    nonlocal keep_header_for_next_chunk
                    # If we already have a header, then we already have an event, so we need to increment immediately.
                    # NOTE: Together with storing with handling the header in the first line a few lines below, we're
                    #       effectively 1-indexing n_events.
                    n_events = 0
                    if kept_header:
                        n_events += 1

                    # Handle the first line from the generator.
                    _, header_info = _handle_line(line, n_events, events_per_chunk=events_per_chunk)
                    yield line
                    # We always increment after yielding.
                    line_count += 1
                    # If we come across a header immediately (should only happen for the first line of the file),
                    # we note the new event, and store the header info.
                    # NOTE: Together with incrementing n_events above, we're effectively 1-indexing n_events.
                    if header_info:
                        n_events += 1
                        event_header_info.append(header_info)

                    # Handle additional lines
                    for local_line in read_lines:
                        time_to_stop, header_info = _handle_line(local_line, n_events, events_per_chunk=events_per_chunk)
                        line_count += 1
                        if header_info:
                            n_events += 1
                            if not time_to_stop:
                                event_header_info.append(header_info)
                                # Since the header line will be skipped by loadtxt, we need to account for that
                                # by subtracting the number of events so far.
                                event_split_index.append(line_count - len(event_split_index))
                            else:
                                keep_header_for_next_chunk = header_info
                                print(f"header_info: {header_info}, keep_header_for_next_chunk: {keep_header_for_next_chunk}")
                        yield local_line
                        if time_to_stop:
                            print(f"event_split_index len: {len(event_split_index)} - {event_split_index}")
                            break

                yield _inner(kept_header=len(event_header_info) > 0), event_split_index, event_header_info
                #line_count += 1
                return_count += 1
                print(f"line_count: {line_count}, return_count: {return_count}")
                print(f"keep_header_for_next_chunk: {keep_header_for_next_chunk}")
                #yield i, line
                #if return_count > 2:
                #    return
            #print(f"return_count: {return_count}")

        #return wrap_reading_file

    #import IPython; IPython.embed()

    #event_split_index_offset = 0
    count = 0
    for chunk_generator, event_split_index, event_header_info in read_events_in_chunks(filename=filename, events_per_chunk=5):
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

