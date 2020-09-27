#!/usr/bin/env python3

""" Tests for parsing JETSCAPE.

"""

import logging
import re
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Sequence, Union, Tuple

import awkward1 as ak
import numpy as np


logger = logging.getLogger(__name__)


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


def read_events_in_chunks(filename: Union[Path, str], events_per_chunk: int = int(1e5)) -> Iterable[Tuple[Iterable[str], List[int], List[Any]]]:
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

    with open(filename, "r") as f:
        # Setup
        # This is just for convenience.
        return_count = 0
        # This is used to pass a header to the next chunk. This is necessary because we don't know an event
        # is over until we already get the header for the next event. We could keep that line and reparse,
        # but there's no need to parse a file twice.
        keep_header_for_next_chunk = None

        # Define an iterator so we can increment it in different locations in the code.
        # Fine to use if it the entire file fits in memory.
        #read_lines = iter(f.readlines())
        # Use this if the file doesn't fit in memory (fairly likely for these type of files)
        read_lines = iter(f)

        for line in read_lines:
            # Setup
            # We keep track of the location of where to split each event. That way, we can come back later
            # and split the 2D numpy array into an awkward array with a jagged structure.
            event_split_index: List[int] = []
            # Store the event header info, to be returned alongside the particles and event split index.
            event_header_info = []
            # If we've kept around a header from a previous chunk, then store that for this iteration.
            if keep_header_for_next_chunk:
                event_header_info.append(keep_header_for_next_chunk)
                # Now that we've stored it, reset it to ensure that it doesn't cause problems for future iterations.
                keep_header_for_next_chunk = None

            def _inner(line: str, kept_header: bool) -> Iterable[str]:
                """ Closure to generate a chunk of events.

                The closure ensures access to the same generator used to access the file.

                Args:
                    kept_header: If True, it means that we kept the header from a previous chunk.
                Returns:
                    Generator yielding the lines of the file.
                """
                # Anything that is returned from this function will be consumed by np.loadtxt, so we can't
                # directly return any value. Instead, we have to make this nonlocal so we can set it here,
                # and the result will be accessible outside during the next chunk.
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
                # Instead of defining the variable here, we account for it in the enumeration below by
                # starting at 1.

                # If we come across a header immediately (should only happen for the first line of the file),
                # we note the new event, and store the header info.
                # NOTE: Together with incrementing n_events above, we're effectively 1-indexing n_events.
                if header_info:
                    n_events += 1
                    event_header_info.append(header_info)
                    # NOTE: We don't record any info here for event_split_index because this is line 0, and
                    #       it would try to split on both sides of it, leading to an empty first event.

                # Handle additional lines
                # Start at one to account for the first land already being handled.
                for line_count, local_line in enumerate(read_lines, start=1):
                    time_to_stop, header_info = _handle_line(local_line, n_events, events_per_chunk=events_per_chunk)
                    line_count += 1

                    # A new header signals a new event. It needs some careful handling.
                    if header_info:
                        n_events += 1
                        # If it's just some event in the middle of the chunk, then we just store the header and event split information.
                        if not time_to_stop:
                            event_header_info.append(header_info)
                            # Since the header line will be skipped by loadtxt, we need to account for that
                            # by subtracting the number of events so far.
                            event_split_index.append(line_count - len(event_split_index))
                        else:
                            # If we're about to end this chunk, we need to hold on to the most recent header
                            # (which signaled that we're ready to end the chunk). We'll hold onto it until we
                            # look at the next chunk.
                            keep_header_for_next_chunk = header_info
                            #print(f"header_info: {header_info}, keep_header_for_next_chunk: {keep_header_for_next_chunk}")

                    # Regardless of the header status, we should always yield the line so it can be handled downstream.
                    yield local_line

                    # Finally, if it's time to end the chunk, we need to fully break the loop. We'll pick
                    # up in the next chunk from the first line of the new event (with the header info that's
                    # stored above).
                    if time_to_stop:
                        #print(f"event_split_index len: {len(event_split_index)} - {event_split_index}")
                        break

            # Yield the generator for the chunk, along with useful information.
            # NOTE: When we pass these lists, they're empty. This only works because lists are mutable, and thus
            #       our changes are passed on.
            yield _inner(line=line, kept_header=len(event_header_info) > 0), event_split_index, event_header_info

            # Keep track of what's going on. This is basically a debugging tool.
            return_count += 1
            #print(f"return_count: {return_count}")
            #print(f"keep_header_for_next_chunk: {keep_header_for_next_chunk}")

        #print(f"return_count: {return_count}")

    # If we've gotten here, that means we've finally exhausted the file. There's nothing else to do!


def read(filename: Union[Path, str], events_per_chunk: int) -> None:
    # Validation
    filename = Path(filename)

    # Read the file, creating chunks of events.
    for chunk_generator, event_split_index, event_header_info in read_events_in_chunks(filename=filename, events_per_chunk=events_per_chunk):
        hadrons = np.loadtxt(chunk_generator)
        array_with_events = ak.Array(np.split(hadrons, event_split_index))
        # Cross check
        if events_per_chunk > 0:
            assert len(event_split_index) == events_per_chunk - 1
            assert len(event_header_info) == events_per_chunk
        #print(len(event_split_index))
        #print(f"hadrons: {hadrons}")
        #print(f"array_with_events: {array_with_events}")
        #print(ak.type(array_with_events))
        #print(f"Event header info: {event_header_info}")
        #import IPython; IPython.embed()

        # Test
        array = ak.zip({
            # TODO: Does the conversion add any real computation time?
            "particle_ID": ak.values_astype(array_with_events[:, :, 1], np.int32),
            # I think the status is always 0 because we're looking at final state particles. So we skip storing it.
            #"status": ak.values_astype(array_with_events[:, :, 2], np.int32),
            "E": ak.values_astype(array_with_events[:, :, 3], np.float32),
            "px": ak.values_astype(array_with_events[:, :, 4], np.float32),
            "py": ak.values_astype(array_with_events[:, :, 5], np.float32),
            "pz": ak.values_astype(array_with_events[:, :, 6], np.float32),
            # Skip these because we're going to be working with four vectors anyway, so it shouldn't be a
            # big deal to recalculate them, especially compare to the added storage space.
            #"eta": ak.values_astype(array_with_events[:, :, 7], np.float32),
            #"phi": ak.values_astype(array_with_events[:, :, 8], np.float32),
        })

        # TODO: Write this out with a proper filename...
        # Parquet doesn't appear to save space vs tar.gz for all columns...
        # However, we do save space by converting types and dropping unneeded columns.
        # And it should load much faster!
        ak.to_parquet(array, "test.parquet")

    import IPython; IPython.embed()


if __name__ == "__main__":
    read(filename="final_state_hadrons.dat", events_per_chunk=5)

