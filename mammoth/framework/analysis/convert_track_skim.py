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

    # Columns to store as integers
    use_dictionary = [
        "run_number",
        "trigger_bit_INT7",
    ]
    if collision_system == "pythia":
        # NOTE: Uses notation from arrow/parquet
        #       `list.item` basically gets us to an column in the list.
        #       This may be a little brittle, but let's see.
        use_dictionary += [
            "det_level.list.item.label",
            "part_level.list.item.label",
            "part_level.list.item.particle_ID",
        ]
    if collision_system == "PbPb":
        use_dictionary += [
            "centrality",
            "event_plane_V0M",
            "trigger_bit_central",
            "trigger_bit_semi_central",
        ]

    # Unfortunately, it appears that dictionary encoding doesn't pick up bool
    # correctly, so we convert it to the next smallest object that works - an unsigned byte
    for column_name in use_dictionary:
        if "trigger_bit" in column_name:
            arrays[column_name] = ak.values_astype(arrays[column_name], "B")

    #import IPython; IPython.embed()

    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Use for anything other than floats
        use_dictionary=use_dictionary,
        # Optimize for floats for the rest
        # Generally enabling seems to work better than specifying exactly the fields
        # because it's unclear how to specify nested fields here.
        use_byte_stream_split=True,
        version="2.0",
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
