"""Convert track skim to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

import awkward as ak
import numpy as np

from mammoth.framework import sources, utils


def track_skim_to_awkward(
    filename: Path,
    collision_system: str,
) -> ak.Array:
    # First, event level properties
    event_level_columns = [
        "run_number",
        "trigger_bit_INT7",
    ]
    if collision_system == "PbPb":
        event_level_columns += [
            "centrality",
            "event_plane_V0M",
            "trigger_bit_central",
            "trigger_bit_semi_central",
        ]
    # Next, particle level columns
    _base_particle_columns = [
        "pt", "eta", "phi"
    ]
    _MC_particle_columns = [
        "particle_ID", "label",
    ]
    particle_columns = [f"particle_data_{c}" for c in _base_particle_columns]
    # Pick up the extra columns in the case of pythia
    if collision_system == "pythia":
        particle_columns += [f"particle_data_{c}" for c in _MC_particle_columns]
        # We skip particle_ID for the detector level
        particle_columns.pop(particle_columns.index("particle_data_particle_ID"))
        # And then do the same for particle_gen
        particle_columns += [f"particle_gen_{c}" for c in _base_particle_columns]
        particle_columns += [f"particle_gen_{c}" for c in _MC_particle_columns]

    data = sources.UprootSource(
        filename=filename,
        tree_name=f"AliAnalysisTaskTrackSkim_{collision_system}_tree",
        columns=event_level_columns + particle_columns,
    ).data()

    # NOTE: If there are no accepted tracks, we don't both storing the event.
    #       However, we attempt to preclude this at the AnalysisTask level by not filling events where therew
    #       are no accepted tracks in the first collection.

    # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
    particle_data_columns = [c for c in particle_columns if "particle_data" in c]
    if collision_system == "pythia":
        particle_gen_columns = [c for c in particle_columns if "particle_gen" in c]
        return ak.Array(
            {
                "det_level": ak.zip(
                    dict(
                        zip(
                            [c.replace("particle_data_", "") for c in list(particle_data_columns)],
                            ak.unzip(data[particle_data_columns]),
                        )
                    )
                ),
                "part_level": ak.zip(
                    dict(
                        zip(
                            [c.replace("particle_gen_", "") for c in list(particle_gen_columns)],
                            ak.unzip(data[particle_gen_columns]),
                        )
                    )
                ),
                **dict(
                    zip(
                        event_level_columns,
                        ak.unzip(data[event_level_columns]),
                    )
                ),
            },
        )

    return ak.Array(
        {
            "data": ak.zip(
                dict(
                    zip(
                        [c.replace("particle_data_", "") for c in list(particle_data_columns)],
                        ak.unzip(data[particle_data_columns]),
                    )
                )
            ),
            **dict(
                zip(
                    event_level_columns,
                    ak.unzip(data[event_level_columns]),
                )
            ),
        },
    )


def write_to_parquet(arrays: ak.Array, filename: Path, collision_system: str) -> bool:
    """Write the jagged track skim arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Determine the types for improved compression when writing
    # Ideally, we would determine these dynamically, but it's unclear how to do this at
    # the moment with awkward, so for now we specify them by hand...
    # float_types = [np.float32, np.float64]
    # float_columns = list(self.output_dataframe.select_dtypes(include=float_types).keys())
    # other_columns = list(self.output_dataframe.select_dtypes(exclude=float_types).keys())
    # Typing info
    # For pp
    # In [1]: arrays.type
    # Out[1]: 703 * {"data": var * {"pt": float32, "eta": float32, "phi": float32}, "run_number": int32, "trigger_bit_INT7": bool}

    # Apparently, specifying use_byte_stream_split=True causes bool to try to encode with the
    # byte stream, even if we specify dictionary encoding (from parquet metadata, I guess it may
    # be because bool can't be dictionary encoded either?). There are a few possible workarounds:
    #
    # 1. Use `values_astype` to convert bool to the next smallest type -> unsigned byte. This works,
    #    but costs storage.
    # 2. Specify `use_dictionary=True` to default encode as dictionary, and then specify the byte stream
    #    split columns by hand. This also works, but since dictionary is preferred over the byte stream
    #    (according to the parquet docs), that list of byte stream split columns is basically meaningless.
    #    So this is equivalent to not using byte stream split at all, which isn't very helpful.
    # 3. Specify both the dictionary columns and byte split stream columns explicitly. This seems to work,
    #    provides good compression, and doesn't error on bool. So we use this option.

    # Columns to store as integers
    dictionary_encoded_columns = [
        "run_number",
        "trigger_bit_INT7",
    ]
    if collision_system == "pythia":
        # NOTE: Uses notation from arrow/parquet
        #       `list.item` basically gets us to an column in the list.
        #       This may be a little brittle, but let's see.
        # NOTE: Recall that we don't include `particle_ID` for det_level because it's all 0s.
        dictionary_encoded_columns += [
            "det_level.list.item.label",
            "part_level.list.item.label",
            "part_level.list.item.particle_ID",
        ]
    if collision_system == "PbPb":
        dictionary_encoded_columns += [
            "centrality",
            "event_plane_V0M",
            "trigger_bit_central",
            "trigger_bit_semi_central",
        ]

    # Columns to store as float
    first_collection_name = "data" if collision_system != "pythia" else "det_level"
    byte_stream_split_columns = [
        f"{first_collection_name}.list.item.pt",
        f"{first_collection_name}.list.item.eta",
        f"{first_collection_name}.list.item.phi",
    ]
    if collision_system == "pythia":
        byte_stream_split_columns += [
            "part_level.list.item.pt",
            "part_level.list.item.eta",
            "part_level.list.item.phi",
        ]

    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Use for anything other than floats
        use_dictionary=dictionary_encoded_columns,
        # Optimize for floats for the rest
        use_byte_stream_split=byte_stream_split_columns,
    )

    return True


if __name__ == "__main__":
    # for collision_system in ["pythia"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        print(f"Converting collision system {collision_system}")
        arrays = track_skim_to_awkward(
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
            ),
            collision_system=collision_system,
        )

        write_to_parquet(
            arrays=arrays,
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
            ),
            collision_system=collision_system,
        )
