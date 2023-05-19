"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, UCB/LBL
"""

from __future__ import annotations

import functools
import operator
from pathlib import Path
from typing import Any, Generator, Mapping, MutableMapping

import attrs
import awkward as ak
import numpy as np

from mammoth.framework import sources, utils


@attrs.frozen
class Columns:
    identifiers: dict[str, str]
    event_level: dict[str, str]
    _particle_level: dict[str, str]

    @classmethod
    def create(cls, collision_system: str, missing_particle_PID: bool = False) -> Columns:
        # Identifiers for reconstructing the event structure
        # According to James:
        # Both data and MC need run_number and ev_id.
        # Data additionally needs ev_id_ext
        identifiers = {
            "run_number": "run_number",
            "ev_id": "ev_id",
        }
        if collision_system in ["pp", "PbPb"]:
            identifiers.update({"ev_id_ext": "ev_id_ext"})

        # Event level properties
        event_level_columns = {"z_vtx_reco": "z_vtx_reco", "is_ev_rej": "is_ev_rej"}
        # Collision system customization
        if collision_system == "PbPb":
            event_level_columns.update({"centrality": "centrality"})
            # For the future, perhaps can add:
            # - event plane angle (but doesn't seem to be in HF tree output :-( )
        # Also need the identifiers
        event_level_columns = {
            **identifiers,
            **event_level_columns,
        }

        # Particle columns and names.
        particle_level_columns = {
            **identifiers,
            "ParticlePt": "pt",
            "ParticleEta": "eta",
            "ParticlePhi": "phi",
        }
        # This field is only available at particle level, and usually is also
        # dropped for the FastSim.
        if not missing_particle_PID:
            particle_level_columns["ParticlePID"] = "particle_ID"

        return cls(
            identifiers=identifiers,
            event_level=event_level_columns,
            particle_level=particle_level_columns,
        )

    def particle_level(self, data_fields_only: bool = True) -> dict[str, str]:
        fields_to_exclude: list[str] = []
        if data_fields_only:
            fields_to_exclude.append("ParticlePID")
        return {k: v for k, v in self._particle_level.items() if k not in fields_to_exclude}

    def standardized_particle_names(self, data_fields_only: bool = True) -> dict[str, str]:
        return {
            k: v for k, v in self.particle_level(data_fields_only=data_fields_only).items() if k not in self.identifiers
        }


@attrs.define
class FileSource:
    _filename: Path = attrs.field(converter=Path)
    _collision_system: str = attrs.field()
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(
        self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT
    ) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        # Setup
        # In the case of the FastSim, it looks like they usually drop the ParticlePID column
        fastsim_output = "FastSim" in self._filename.name

        # NOTE: We can only load a whole file and chunk it afterwards since the event boundaries
        #       are not known otherwise. Unfortunately, it's hacky, but it seems like the best
        #       bet for now as of Feb 2023.
        columns = Columns.create(collision_system=self._collision_system, missing_particle_PID=fastsim_output)
        _is_pp_MC = self._collision_system in ["pythia", "pp_MC"]

        # There is a prefix for the original HF tree creator, but not one for the FastSim
        tree_prefix = "" if fastsim_output else "PWGHF_TreeCreator/"

        # Grab the various trees. It's less efficient, but we'll have to grab the trees
        # with separate file opens since adding more trees to the UprootSource would be tricky.
        event_properties_source: sources.Source = sources.UprootSource(
            filename=self._filename,
            tree_name=f"{tree_prefix}tree_event_char",
            columns=list(columns.event_level),
        )
        data_source = sources.UprootSource(
            filename=self._filename,
            tree_name=f"{tree_prefix}tree_Particle",
            columns=list(columns.particle_level(data_fields_only=True)),
        )
        # NOTE: This is where we're defining the "det_level", "part_level", or "data" fields
        data_sources = {
            "event_level": event_properties_source,
            "det_level" if _is_pp_MC else "data": data_source,
        }
        if _is_pp_MC:
            data_sources["part_level"] = sources.UprootSource(
                filename=self._filename,
                tree_name=f"{tree_prefix}tree_Particle_gen",
                columns=list(columns.particle_level(data_fields_only=False)),
            )

        _transformed_data = _transform_output(
            # NOTE: We need to grab everything because the file structure is such that we
            #       need to reconstruct the event structure from the fully flat tree.
            #       Consequently, selecting on chunk size would be unlikely to select
            #       the boundaries of events, giving unexpected behavior. So we load it
            #       all now and add on the chunking afterwards. It's less efficient in terms
            #       of memory, but much more straightforward.
            gen_data={k: s.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE) for k, s in data_sources.items()},
            collision_system=self._collision_system,
            columns=columns,
        )
        return sources.generator_from_existing_data(
            data=next(_transformed_data),
            chunk_size=chunk_size,
            source_default_chunk_size=self._default_chunk_size,
        )

    @classmethod
    def create_deferred_source(
        cls,
        collision_system: str,
        default_chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> sources.SourceFromFilename:
        """Create a FileSource with a closure such that all arguments are set except for the filename.

        Args:
            collision_system: The collision system of the data.
            default_chunk_size: The default chunk size to use when generating data.

        Returns:
            A Callable which takes the filename and creates the FileSource.
        """
        if metadata is None:
            metadata = {}

        def wrap(filename: Path) -> FileSource:
            # Help out mypy
            assert metadata is not None
            return cls(
                filename=filename,
                collision_system=collision_system,
                default_chunk_size=default_chunk_size,
                metadata=metadata,
            )

        return wrap


def _transform_output(
    gen_data: Mapping[str, Generator[ak.Array, sources.T_ChunkSize | None, None]],
    collision_system: str,
    columns: Columns,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    # Setup
    _standardized_particle_names = columns.standardized_particle_names()
    _is_pp_MC = collision_system in ["pythia", "pp_MC"]

    try:
        data = {k: next(v) for k, v in gen_data.items()}
        while True:
            # First, convert the flat arrays into jagged arrays by grouping by the identifiers.
            # This allows us to work with the data as expected.
            # NOTE: In less condensed form, this is what we're doing:
            """
            det_level_tracks = utils.group_by(
                array=next(det_level_tracks_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)),
                by=identifiers,
            )
            # There is one entry per event, so we don't need to do any group by steps.
            event_properties = next(event_properties_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
            """
            event_data_in_jagged_format = {
                k: utils.group_by(
                    array=v,
                    by=list(columns.identifiers.values()),
                )
                for k, v in data.items()
                if k != "event_level"
            }
            # There is one entry per event, so we don't need to do any group by steps.
            event_data_in_jagged_format["event_level"] = data["event_level"]

            # Event selection
            # We apply the event selection implicitly to the particles by requiring the identifiers
            # of the part level, det level, and event properties match.
            # NOTE: Since we're currently using this for conversion, it's better to have looser
            #       conditions so we can delete the original files. So we disable this for now.
            # event_properties = event_properties[
            #     (event_properties["is_ev_rej"] == 0)
            #     & (np.abs(event_properties["z_vtx_reco"]) < 10)
            # ]

            # Now, we're on to merging the particles and event level information. Remarkably, the part level,
            # det level, and event properties all have a unique set of identifiers. None of them are entirely
            # subsets of the others. This isn't particularly intuitive to me, but in any case, this seems to
            # match up with the way that it's handled in pyjetty.
            # If there future issues with merging, check out some additional thoughts and suggestions on
            # merging here: https://github.com/scikit-hep/awkward-1.0/discussions/633

            # First, grab the identifiers from each collection so we can match them up.
            # NOTE: For the tracks, everything is broadcasted to the shape of the particles, which is jagged,
            #       so we take the first instance for each event (since it will be the same for every particle
            #       in an event).
            # NOTE: In less condensed form, this is what we're doing:
            """
            det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
            event_properties_identifiers = event_properties[identifiers]
            """
            identifiers = {
                k: v[list(columns.identifiers.values())][:, 0]
                for k, v in event_data_in_jagged_format.items()
                if k != "event_level"
            }
            identifiers["event_level"] = event_data_in_jagged_format["event_level"][list(columns.identifiers.values())]

            # Next, find the overlap for each collection with each other collection, storing the result in
            # a mask. As noted above, no collection appears to be a subset of the other.
            # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
            #       be flattened by numpy.  However, it works as expected if it's a structured array (which
            #       is the default approach for Array conversion, so we get a bit lucky here).
            # NOTE: In less condensed form, this is what we're doing:
            """
            det_level_tracks_mask = np.isin(
                np.asarray(det_level_tracks_identifiers),
                np.asarray(part_level_tracks_identifiers),
            ) & np.isin(
                np.asarray(det_level_tracks_identifiers),
                np.asarray(event_properties_identifiers),
            )
            det_level_tracks = det_level_tracks[det_level_tracks_mask]
            ...
            """
            masks_for_combining_levels = {
                k: functools.reduce(
                    operator.and_,
                    [
                        np.isin(np.asarray(v), np.asarray(identifiers[other_level]))
                        for other_level in identifiers
                        if k != other_level
                    ],
                )
                for k, v in identifiers.items()
            }
            # Once we have the mask, we immediately apply it.
            event_data_in_jagged_format = {
                k: v[masks_for_combining_levels[k]] for k, v in event_data_in_jagged_format.items()
            }

            if _is_pp_MC:
                # Now, some rearranging the field names for uniformity.
                # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
                # would be required, but apparently not.
                _result = yield ak.Array(
                    {
                        **{
                            particle_collection_name: ak.zip(
                                {
                                    v: event_data_in_jagged_format[particle_collection_name][k]
                                    for k, v in columns.standardized_particle_names(
                                        data_fields_only=particle_collection_name != "part_level"
                                    ).items()
                                }
                            )
                            for particle_collection_name in ["det_level", "part_level"]
                        },
                        **dict(
                            zip(
                                ak.fields(event_data_in_jagged_format["event_level"]),
                                ak.unzip(event_data_in_jagged_format["event_level"]),
                            )
                        ),
                    },
                )
            else:
                # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
                _result = yield ak.Array(
                    {
                        "data": ak.zip(
                            {
                                v: event_data_in_jagged_format["data"][k]
                                for k, v in columns.standardized_particle_names(data_fields_only=True).items()
                            }
                        ),
                        **dict(
                            zip(
                                ak.fields(event_data_in_jagged_format["event_level"]),
                                ak.unzip(event_data_in_jagged_format["event_level"]),
                            )
                        ),
                    },
                )

            # Update for next step
            data = {k: v.send(_result) for k, v in gen_data.items()}
    except StopIteration:
        pass


def write_to_parquet(arrays: ak.Array, filename: Path) -> bool:
    """Write the jagged HF tree arrays to parquet.

    In this form, they should be ready to analyze.
    """
    ak.to_parquet(
        array=arrays,
        destination=str(filename),
        compression="zstd",
        # Optimize the compression via improved encodings for floats and strings.
        # Conveniently, awkward 2.x will now select the right columns for each if simply set to `True`
        # Optimize for columns with anything other than floats
        parquet_dictionary_encoding=True,
        # Optimize for columns with floats
        parquet_byte_stream_split=True,
    )

    return True


def run_standalone_tests() -> None:
    # arrays = hf_tree_to_awkward(filename=Path("/software/rehlers/dev/substructure/trains/pythia/568/AnalysisResults.20g4.001.root"))
    # for collision_system in ["pythia"]:
    # for collision_system in ["pp", "pythia", "PbPb"]:
    #     print(f"Converting collision system {collision_system}")
    #     if collision_system == "pythia":
    #         arrays = hf_tree_to_awkward_MC(
    #             filename=Path(
    #                 f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
    #             ),
    #             collision_system=collision_system,
    #         )
    #     else:
    #         arrays = hf_tree_to_awkward_data(
    #             filename=Path(
    #                 f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
    #             ),
    #             collision_system=collision_system,
    #         )

    #     write_to_parquet(
    #         arrays=arrays,
    #         filename=Path(
    #             f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.parquet"
    #         ),
    #         collision_system=collision_system,
    #     )

    collision_system = "pythia"
    for generator, intermediate_path in [("herwig", "herwig_alice/tree_fastsim/266374/265216/260023")]:
        filename = Path(f"projects/lbl_fastsim/{intermediate_path}/12/93/AnalysisResultsFastSim.root")
        source = FileSource(
            filename=filename,
            collision_system=collision_system,
        )
        arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))

        write_to_parquet(
            arrays=arrays,
            filename=Path(f"projects/lbl_fastsim/{generator}_alice/AnalysisResults_HFTree.parquet"),
        )

        import IPython

        IPython.embed()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    run_standalone_tests()
