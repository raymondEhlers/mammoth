#!/usr/bin/env python3

""" Tests for parsing JETSCAPE.

"""

from pathlib import Path

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
    event_line_number = []

    def gen_events(filename: Path, events_per_file: int = 1e5):
        #def yield_lines(i, line):
        #    if line.startswith("#"):
        #        if len(event_line_number) == events_per_file:
        #            yield StopIteration
        #        # Since this line will be skipped by loadtxt, we need to account for that
        #        # but subtracting the number of events so far.
        #        event_line_number.append(i - len(event_line_number))
        #    yield line

        #def read_from_file():
        #    stopped = False
        #    with open(filename, "r") as f:
        #        #yield from yield_lines(f.readlines())
        #        for i, line in enumerate(f.readlines()):
        #            res = yield i, line, stopped
        #            if res is not None:
        #                stopped = res

        #def wrap_reading_file():
        #    generate_from_file = read_from_file()
        #    for i, line, stopped in generate_from_file:
        #        if line.startswith("#"):
        #            if len(event_line_number) % events_per_file == 0 and stopped == False:
        #                print("Stopping")
        #                generate_from_file.send(True)
        #                stopped = True
        #                raise StopIteration
        #            # Since this line will be skipped by loadtxt, we need to account for that
        #            # but subtracting the number of events so far.
        #            event_line_number.append(i - len(event_line_number))
        #        yield line

        #def read_from_file():
        #    with open(filename, "r") as f:
        #        #yield from yield_lines(f.readlines())
        #        for i, line in enumerate(f.readlines()):
        #            yield i, line

        #def wrap_reading_file():
        #    generate_from_file = read_from_file()
        #    for i, line in generate_from_file:
        #        def _inner():
        #            time_to_stop = False
        #            if line.startswith("#"):
        #                # Need to check if event_line_number so that we get past 0...
        #                if event_line_number and len(event_line_number) % events_per_file == 0:
        #                    time_to_stop = True
        #                # Since this line will be skipped by loadtxt, we need to account for that
        #                # but subtracting the number of events so far.
        #                event_line_number.append(i - len(event_line_number))

        #                if time_to_stop:
        #                    return

        #            yield line
        #        return _inner()

        line_count = 0

        def _handle_line(line, line_count):
            print(f"{line_count}: line: {line}")
            time_to_stop = False
            if line.startswith("#"):
                # Need to check if event_line_number so that we get past 0...
                if event_line_number and len(event_line_number) % events_per_file == 0:
                    print("Time to stop!")
                    time_to_stop = True
                # Since this line will be skipped by loadtxt, we need to account for that
                # but subtracting the number of events so far.
                event_line_number.append(line_count - len(event_line_number))

            return line, time_to_stop

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
                            print(f"event_line_number len: {len(event_line_number)} - {event_line_number}")
                            break
                        #if line_count > 5:
                        #    break
                        #if line.startswith("#"):
                        #    # Need to check if event_line_number so that we get past 0...
                        #    if event_line_number and len(event_line_number) % events_per_file == 0:
                        #        time_to_stop = True
                        #    # Since this line will be skipped by loadtxt, we need to account for that
                        #    # but subtracting the number of events so far.
                        #    event_line_number.append(line_count - len(event_line_number))

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

    # NOTE: Can parse / store more information here if it's helpful.
    def yield_final_state(filename: Path):
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()):
                if line.startswith("#"):
                    # Since this line will be skipped by loadtxt, we need to account for that
                    # but subtracting the number of events so far.
                    event_line_number.append(i - len(event_line_number))
                yield line


    for generate_block in gen_events(filename=filename, events_per_file=5):
        hadrons = np.loadtxt(generate_block())
        print(len(event_line_number))

    #hadrons = np.loadtxt(yield_final_state(filename=filename))

    # We have to take [1:] because we need to skip the first line. Otherwise,
    # we used up with an empty first event.
    temp_arr = ak.Array(np.split(hadrons, event_line_number[1:]))

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

