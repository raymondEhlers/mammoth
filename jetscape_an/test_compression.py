
""" Compression tests.

"""

import base64
import tarfile
import timeit
from pathlib import Path

import awkward1 as ak
import numpy as np

from jetscape_an import parse_ascii


def setup(filename: str, events_per_chunk: int):
    # Inputs

    directory_name = "5020_PbPb_0-10_0R25_1R0_1"
    full_filename = f"../phys_paper/AAPaperData/{directory_name}/{filename}.out"
    max_chunks = 1

    print("First ieration - saving with awkawrd")
    for i, (chunk_generator, event_split_index, event_header_info) in enumerate(parse_ascii.read_events_in_chunks(filename=full_filename, events_per_chunk=events_per_chunk)):
        print("Loading chunk")
        start_time = timeit.default_timer()

        hadrons = np.loadtxt(chunk_generator)
        array_with_events = ak.Array(np.split(hadrons, event_split_index))
        elapsed = timeit.default_timer() - start_time
        print(f"Loading {events_per_chunk} events with numpy.loadtxt: {elapsed}")

        # Bail out after one chunk
        break

    # Save the output
    output_dir = Path("compression") / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(array_with_events, output_dir / f"{filename}_{events_per_chunk}_00.parquet")

    # NOTE: We have to do a second separate iteration because we can't clone generators easily. (Maybe itertools.tee, but this is also easier...)
    #       Plus, I'd like to get a reasonable estimate for the np.loadtxt performance alone.
    print("Second iteration - saving text file")
    lines = []
    for i, (chunk_generator, event_split_index, event_header_info) in enumerate(parse_ascii.read_events_in_chunks(filename=full_filename, events_per_chunk=events_per_chunk)):
        print("Loading chunk")
        start_time = timeit.default_timer()
        lines.extend([l for l in chunk_generator])
        elapsed = timeit.default_timer() - start_time
        print(f"Loading {events_per_chunk} events with text: {elapsed}")

        # Bail out after one chunk
        break

    # Write the text for size comparison
    output_filename = output_dir / f"{filename}_{events_per_chunk}_00.out"
    with open(output_filename, "w") as f:
        f.write("".join(lines))
    # Write the tars here for convenience.
    with tarfile.open(output_filename.with_suffix(".tar.gz"), "w:gz") as tar:
        tar.add(output_filename, arcname=output_filename.name)

    # Try also writing in binary encoding with utf-8
    output_filename = output_dir / f"{filename}_{events_per_chunk}_00_binary_utf-8.out"
    with open(output_filename, "wb") as f:
        f.write("".join(lines).encode("utf-8"))
    # Write the tars here for convenience.
    with tarfile.open(output_filename.with_suffix(".tar.gz"), "w:gz") as tar:
        tar.add(output_filename, arcname=output_filename.name)

    # Try also writing in binary encoding with ascii
    output_filename = output_dir / f"{filename}_{events_per_chunk}_00_binary_ascii.out"
    with open(output_filename, "wb") as f:
        f.write("".join(lines).encode("ascii"))
    # Write the tars here for convenience.
    with tarfile.open(output_filename.with_suffix(".tar.gz"), "w:gz") as tar:
        tar.add(output_filename, arcname=output_filename.name)


def write_trees_with_root(arrays: ak.Array, tag: str = ""):
    # NOTE: This won't work with uproot because it only supports simple branches - we need to use ROOT :-(
    if tag:
        tag = f"_{tag}"
    output_dir = Path("compression/ROOT")
    output_dir.mkdir(parents=True, exist_ok=True)

    import ROOT

    tree = ROOT.TTree("tree", "tree")

    # This is dumb. We should be able to use np arrays...
    particle_ID = ROOT.vector("int32_t" if tag[1:] == "optimized_types" else "int")()
    tree.Branch("particle_ID", particle_ID)
    status = ROOT.vector("int8_t" if tag[1:] == "optimized_types" else "int")()
    tree.Branch("status", status)
    pt = ROOT.vector("float" if tag[1:] == "optimized_types" else "double")()
    tree.Branch("pt", pt)
    eta = ROOT.vector("float" if tag[1:] == "optimized_types" else "double")()
    tree.Branch("eta", eta)
    phi = ROOT.vector("float" if tag[1:] == "optimized_types" else "double")()
    tree.Branch("phi", phi)

    print("Filling tree")
    for event in arrays[:100]:
        for particle in event:
            # Apparently pyroot can't handle type conversions. Cool.
            particle_ID.push_back(int(particle["particle_ID"]))
            status.push_back(int(particle["status"]))
            pt.push_back(particle["pt"])
            eta.push_back(particle["eta"])
            phi.push_back(particle["phi"])
        tree.Fill()
    print("Done filling. Writing files...")

    # ROOT intuition from https://root-forum.cern.ch/t/new-compression-algorithm/27769/3:
    #
    # LZMA: Very slow, but highest compression ratio.
    # LZ4: Very fast, but relatively poor compression ratio.
    # ZLIB: Middle-ground. Modest compression ratio, modest speed.
    #
    # If you’re interested in LZ4, try kLZ4 as the algorithm and 4 as the level.

    #for level in [2, 3, 4, 5, 7]:
        #for name, compression in [(f"zlib_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB),
        #                          (f"lzma_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kLZMA),
        #                          (f"lz4_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kLZ4),
        #                          (f"zstd_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kZSTD)]:
    # Better, use the ROOT default levels...
    # If the case of uncompressed, the algorithm shouldn't matter.
    for name, compression, level in [("none_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB, ROOT.ROOT.RCompressionSetting.ELevel.kUncompressed),
                                     ("zlib_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB, ROOT.ROOT.RCompressionSetting.ELevel.kDefaultZLIB),
                                     ("lzma_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kLZMA, ROOT.ROOT.RCompressionSetting.ELevel.kDefaultLZMA),
                                     ("lz4_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kLZ4, ROOT.ROOT.RCompressionSetting.ELevel.kDefaultLZ4),
                                     ("zstd_{level}", ROOT.ROOT.RCompressionSetting.EAlgorithm.kZSTD, ROOT.ROOT.RCompressionSetting.ELevel.kDefaultZSTD)]:
        # Setup
        name = name.format(level=level)
        compress =  ROOT.ROOT.CompressionSettings(compression, level)
        start_time = timeit.default_timer()
        filename = output_dir / f"{name}{tag}.root"
        f = ROOT.TFile(str(filename), "RECREATE", "", compress)
        tree_to_write =  tree.Clone()
        tree_to_write.Write()
        f.Close()
        elapsed = timeit.default_timer() - start_time
        print(f"ROOT: {name}: {elapsed}")


def write_trees_with_awkward(arrays: ak.Array, tag: str = ""):
    if tag:
        tag = f"_{tag}"

    output_dir = Path("compression/parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Valid values: {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘LZO’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}.
    for compression in ["snappy",
                        "gzip",
                        # Skip lz4 due to some bug, apparently. The package reports the issue.
                        #"lz4",
                        "zstd"]:
        start_time = timeit.default_timer()
        ak.to_parquet(
            arrays, output_dir / f"{compression}{tag}.parquet", compression=compression,
            # In principle, we could select a particular level. But for now, we leave it to arrow to decide.
            compression_level=None
        )
        elapsed = timeit.default_timer() - start_time
        print(f"Parquet: {compression}, tag: \"{tag[1:]}\": {elapsed}")


def formatted(f):
    return format(f, '.6f').rstrip('0').rstrip('.')


def write_ascii_ish(arrays: ak.Array, tag: str = ""):
    if tag:
        tag = f"_{tag}"

    output_dir = Path("compression/ascii")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"ascii{tag}.out"

    print(f"Writing to ascii for \"{tag[1:]}\"")
    start_time = timeit.default_timer()
    with open(filename, "w") as f:
        for event in arrays:
            f.write("# Some random header...\n")
            #np.savetxt(ak.to_numpy(event[["particle_ID", "status", "pt", "eta", "phi"]]), fmt=)
            for particle in event:
                #print(f"{int(particle['particle_ID']):d} {int(particle['status']):d} {formatted(particle['pt'])} {formatted(particle['eta'])} {formatted(particle['phi'])}")
                f.write(f"{int(particle['particle_ID']):d} {int(particle['status']):d} {formatted(particle['pt'])} {formatted(particle['eta'])} {formatted(particle['phi'])}\n")
    elapsed = timeit.default_timer() - start_time
    print(f"Finshed writing in {elapsed}")

    # Write the tar here for convenience.
    with tarfile.open(filename.with_suffix(".tar.gz"), "w:gz") as tar:
        tar.add(filename, arcname=filename.name)


if __name__ == "__main__":
    # Setup
    events_per_chunk = 1000
    filename = "JetscapeHadronListBin100_110"
    input_filename = Path("compression") / "input" / f"{filename}_{events_per_chunk}_00.parquet"
    if not input_filename.exists():
        setup(filename=filename, events_per_chunk=events_per_chunk)

    input_arrays = ak.from_parquet(input_filename)
    # We use some very different value to make it clear if something ever goes wrong.
    # NOTE: It's important to do this before constructing anything else. Otherwise it can
    #       mess up the awkward1 behaviors.
    fill_none_value = -9999
    input_arrays = ak.fill_none(input_arrays, fill_none_value)

    # TODO: Compare just to the test size...
    # TODO: Convert the types here. That way, we know what we're saving from the uncompressed version.

    full_arrays = ak.zip(
        {
            "particle_ID": input_arrays[:, :, 1],
            "status": input_arrays[:, :, 2],
            "E": input_arrays[:, :, 3],
            "px": input_arrays[:, :, 4],
            "py": input_arrays[:, :, 5],
            "pz": input_arrays[:, :, 6],
            "eta": input_arrays[:, :, 7],
            "phi": input_arrays[:, :, 8],
        },
        depth_limit = None,
    )
    arrays = ak.zip({
        "pt": np.sqrt(full_arrays["px"] ** 2 + full_arrays["py"] ** 2),
        "eta": full_arrays["eta"],
        "phi": full_arrays["phi"],
        "particle_ID": full_arrays["particle_ID"],
        "status": full_arrays["status"],
    })
    arrays_type_conversion = ak.zip({
        "pt": ak.values_astype(np.sqrt(full_arrays["px"] ** 2 + full_arrays["py"] ** 2), np.float32),
        "eta": ak.values_astype(full_arrays["eta"], np.float32),
        "phi": ak.values_astype(full_arrays["phi"], np.float32),
        "particle_ID": ak.values_astype(full_arrays["particle_ID"], np.int32),
        "status": ak.values_astype(full_arrays["status"], np.int8),
    })

    # ROOT
    write_trees_with_root(arrays)
    # Intentionally let root do the type conversion...
    write_trees_with_root(arrays, "optimized_types")
    # Awkard
    write_trees_with_awkward(arrays)
    write_trees_with_awkward(arrays_type_conversion, "optimized_types")
    # Ascii
    write_ascii_ish(arrays)
    write_ascii_ish(arrays_type_conversion, "optimized_types")

