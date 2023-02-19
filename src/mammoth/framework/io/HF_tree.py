"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, UCB/LBL
"""

from __future__ import annotations

import functools
import operator
from pathlib import Path
from typing import Any, Dict, Generator, Mapping, MutableMapping

import attrs
import awkward as ak
import numpy as np

from mammoth.framework import sources, utils


@attrs.frozen
class Columns:
    """
    NOTE:
        This isn't implemented yet. I haven't gone through the steps because they're not yet needed,
        but this is the start.
    """

    identifiers: Dict[str, str]
    event_level: Dict[str, str]
    particle_level: Dict[str, str]

    @classmethod
    def create(cls, collision_system: str) -> "Columns":
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
            **{
                "ParticlePt": "pt",
                "ParticleEta": "eta",
                "ParticlePhi": "phi",
            },
        }

        return cls(
            identifiers=identifiers,
            event_level=event_level_columns,
            particle_level=particle_level_columns,
        )

    def standardized_particle_names(self) -> Dict[str, str]:
        return {
            k: v for k, v in self.particle_level.items() if k not in self.identifiers
        }


@attrs.define
class FileSource:
    _filename: Path = attrs.field(converter=Path)
    _collision_system: str
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(
        self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT
    ) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        # NOTE: We can only load a whole file and chunk it afterwards since the event boundaries
        #       are not known otherwise. Unfortunately, it's hacky, but it seems like the best
        #       bet for now as of Feb 2023.
        columns = Columns.create(collision_system=self._collision_system)
        _is_pp_MC = collision_system in ["pythia", "pp_MC"]

        # There is a prefix for the original HF tree creator, but not one for the FastSim
        tree_prefix = "PWGHF_TreeCreator/"
        if "FastSim" in self._filename.name:
            tree_prefix = ""

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
            columns=list(columns.particle_level),
        )
        # NOTE: This is where we're defining the "det_level", "part_level", or "data" fields
        data_sources = {
            "event_level": event_properties_source, "det_level" if _is_pp_MC else "data": data_source,
        }
        if _is_pp_MC:
            data_sources["part_level"] = sources.UprootSource(
                filename=self._filename,
                tree_name=f"{tree_prefix}tree_Particle_gen",
                columns=list(columns.particle_level),
            )

        _transformed_data = _transform_output(
            # NOTE: We need to grab everything because the file structure is such that we
            #       need to reconstruct the event structure from the fully flat tree.
            #       Consequently, selecting on chunk size would be unlikely to select
            #       the boundaries of events, giving unexpected behavior. So we load it
            #       all now and add on the chunking afterwards. It's less efficient in terms
            #       of memory, but much more straightforward.
            gen_data={
                k: s.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)
                for k, s in data_sources.items()
            },
            collision_system=self._collision_system,
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
    ) -> sources.SourceFromFilename:
        """Create a FileSource with a closure such that all arguments are set except for the filename.

        Args:
            collision_system: The collision system of the data.
            default_chunk_size: The default chunk size to use when generating data.

        Returns:
            A Callable which takes the filename and creates the FileSource.
        """

        def wrap(filename: Path) -> "FileSource":
            return cls(
                filename=filename,
                collision_system=collision_system,
                default_chunk_size=default_chunk_size,
            )

        return wrap


def _transform_output(
    gen_data: Mapping[str, Generator[ak.Array, sources.T_ChunkSize | None, None]],
    collision_system: str,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    # Setup
    _columns = Columns.create(collision_system=collision_system)
    _standardized_particle_names = _columns.standardized_particle_names()
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
                    by=list(_columns.identifiers.values()),
                )
                for k, v in data.items() if k != "event_level"
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
                k: v[list(_columns.identifiers.values())][:, 0]
                for k, v in event_data_in_jagged_format.items() if k != "event_level"
            }
            identifiers["event_level"] = event_data_in_jagged_format["event_level"][list(_columns.identifiers.values())]

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
                k: functools.reduce(operator.and_, [
                    np.isin(np.asarray(v), np.asarray(identifiers[other_level]))
                    for other_level in identifiers if k != other_level
                ])
                for k, v in identifiers.items()
            }
            # Once we have the mask, we immediately apply it.
            event_data_in_jagged_format = {
                k: v[masks_for_combining_levels[k]]
                for k, v in event_data_in_jagged_format.items()
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
                                    for k, v in _standardized_particle_names.items()
                                }
                            ) for particle_collection_name in ["det_level", "part_level"]
                        },
                        **dict(zip(ak.fields(event_data_in_jagged_format["event_level"]), ak.unzip(event_data_in_jagged_format["event_level"]))),
                    },
                )
            else:
                # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
                _result = yield ak.Array(
                    {
                        "data": ak.zip(
                            {
                                v: event_data_in_jagged_format["data"][k]
                                for k, v in _standardized_particle_names.items()
                            }
                        ),
                        **dict(zip(ak.fields(event_data_in_jagged_format["event_level"]), ak.unzip(event_data_in_jagged_format["event_level"]))),
                    },
                )

            # Update for next step
            data = {k: v.send(_result) for k, v in gen_data.items()}
    except StopIteration:
        pass


def hf_tree_to_awkward_MC(
    filename: Path,
    collision_system: str,
) -> ak.Array:
    # TODO: Consolidate with the _data function
    # Setup
    # First, we need the identifiers to group_by
    # According to James:
    # Both data and MC need run_number and ev_id.
    # Data additionally needs ev_id_ext
    identifiers = [
        "run_number",
        "ev_id",
    ]
    if collision_system in ["pp", "PbPb"]:
        identifiers += ["ev_id_ext"]
    # Particle columns and names.
    particle_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]
    # We want them to be stored in a standardized manner.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    # Detector level
    det_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=particle_level_columns,
    )
    # Particle level
    part_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle_gen",
        columns=particle_level_columns,
    )
    # Event level properties
    event_properties_columns = ["z_vtx_reco", "is_ev_rej"]
    # Collision system customization
    if collision_system == "PbPb":
        event_properties_columns += ["centrality"]
        # For the future, perhaps can add:
        # - event plane angle (but doesn't seem to be in HF tree output :-( )
    # It seems that the pythia relevant properties like pt hard bin, etc, are
    # all empty so nothing special to be done there.
    event_properties_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_event_char",
        columns=identifiers + event_properties_columns,
    )

    # Convert the flat arrays into jagged arrays by grouping by the identifiers.
    # This allows us to work with the data as expected.
    det_level_tracks = utils.group_by(
        array=next(det_level_tracks_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)),
        by=identifiers,
    )
    part_level_tracks = utils.group_by(
        array=next(part_level_tracks_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)),
        by=identifiers,
    )
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = next(event_properties_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))

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
    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    part_level_tracks_identifiers = part_level_tracks[identifiers][:, 0]
    event_properties_identifiers = event_properties[identifiers]

    # Next, find the overlap for each collection with each other collection, storing the result in
    # a mask.  As noted above, no collection appears to be a subset of the other.
    # Once we have the mask, we immediately apply it.
    # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
    #       be flattened by numpy.  However, it works as expected if it's a structured array (which
    #       is the default approach for Array conversion, so we get a bit lucky here).
    det_level_tracks_mask = np.isin(
        np.asarray(det_level_tracks_identifiers),
        np.asarray(part_level_tracks_identifiers),
    ) & np.isin(
        np.asarray(det_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    part_level_tracks_mask = np.isin(
        np.asarray(part_level_tracks_identifiers),
        np.asarray(det_level_tracks_identifiers),
    ) & np.isin(
        np.asarray(part_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    part_level_tracks = part_level_tracks[part_level_tracks_mask]
    event_properties_mask = np.isin(
        np.asarray(event_properties_identifiers),
        np.asarray(det_level_tracks_identifiers),
    ) & np.isin(
        np.asarray(event_properties_identifiers),
        np.asarray(part_level_tracks_identifiers),
    )
    event_properties = event_properties[event_properties_mask]

    # Now, some rearranging the field names for uniformity.
    # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
    # would be required, but apparently not.
    return ak.Array(
        {
            "det_level": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            "part_level": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )


def hf_tree_to_awkward_data(
    filename: Path,
    collision_system: str,
) -> ak.Array:
    # TODO: Consolidate with the _MC function
    # Setup
    # First, we need the identifiers to group_by
    # According to James:
    # Both data and MC need run_number and ev_id.
    # Data additionally needs ev_id_ext
    identifiers = [
        "run_number",
        "ev_id",
    ]
    if collision_system in ["pp", "PbPb"]:
        identifiers += ["ev_id_ext"]
    # Particle columns and names.
    particle_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]
    # We want them to be stored in a standardized manner.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    # Detector level
    det_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=particle_level_columns,
    )
    # Event level properties
    event_properties_columns = ["z_vtx_reco", "is_ev_rej"]
    # Collision system customization
    if collision_system == "PbPb":
        event_properties_columns += ["centrality"]
        # For the future, perhaps can add:
        # - event plane angle (but doesn't seem to be in HF tree output :-( )
    # It seems that the pythia relevant properties like pt hard bin, etc, are
    # all empty so nothing special to be done there.
    event_properties_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_event_char",
        columns=identifiers + event_properties_columns,
    )

    # Convert the flat arrays into jagged arrays by grouping by the identifiers.
    # This allows us to work with the data as expected.
    det_level_tracks = utils.group_by(
        array=next(det_level_tracks_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)),
        by=identifiers,
    )
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = next(event_properties_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))

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
    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    event_properties_identifiers = event_properties[identifiers]

    # Next, find the overlap for each collection with each other collection, storing the result in
    # a mask.  As noted above, no collection appears to be a subset of the other.
    # Once we have the mask, we immediately apply it.
    # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
    #       be flattened by numpy.  However, it works as expected if it's a structured array (which
    #       is the default approach for Array conversion, so we get a bit lucky here).
    det_level_tracks_mask = np.isin(
        np.asarray(det_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    event_properties_mask = np.isin(
        np.asarray(event_properties_identifiers),
        np.asarray(det_level_tracks_identifiers),
    )
    event_properties = event_properties[event_properties_mask]

    # Now, some rearranging the field names for uniformity.
    # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
    # would be required, but apparently not.
    return ak.Array(
        {
            "data": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )


def write_to_parquet(arrays: ak.Array, filename: Path, collision_system: str) -> bool:
    """Write the jagged HF tree arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Determine the types for improved compression when writing
    # Ideally, we would determine these dynamically, but it's unclear how to do this at
    # the moment with awkward, so for now we specify them by hand...
    # float_types = [np.float32, np.float64]
    # float_columns = list(self.output_dataframe.select_dtypes(include=float_types).keys())
    # other_columns = list(self.output_dataframe.select_dtypes(exclude=float_types).keys())
    # Typing info
    # In [8]: arrays.type
    # Out[8]: 18681 * {"det_level": var * {"pt": float32, "eta": float32, "phi": float32}, "part_level": var * {"pt": float32, "eta": float32, "phi": float32}, "run_number": int32, "ev_id": int32, "z_vtx_reco": float32, "is_ev_rej": int32}

    # Columns to store as integers
    use_dictionary = [
        "run_number",
        "ev_id",
        "is_ev_rej",
    ]
    if collision_system in ["pp", "PbPb"]:
        use_dictionary += ["ev_id_ext"]

    ak.to_parquet(
        array=arrays,
        destination=str(filename),
        compression="zstd",
        # Use for anything other than floats
        #use_dictionary=use_dictionary,
        # Optimize for floats for the rest
        # Generally enabling seems to work better than specifying exactly the fields
        # because it's unclear how to specify nested fields here.
        #use_byte_stream_split=True,
        # use_byte_stream_split=[
        #     "pt", "eta", "phi",
        #     #"det_level", "part_level",
        #     "z_vtx_reco",
        # ],
    )

    return True


if __name__ == "__main__":
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
            filename=Path(
                f"projects/lbl_fastsim/{generator}_alice/AnalysisResults_HFTree.parquet"
            ),
            collision_system=collision_system,
        )

        import IPython; IPython.embed()
