"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, UCB/LBL
"""

from __future__ import annotations

import copy
import functools
import logging
import operator
from collections.abc import Generator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import attrs
import awkward as ak
import numpy as np

from mammoth.framework import sources, utils

logger = logging.getLogger(__name__)


@attrs.frozen
class Columns:
    """Define the columns of interest for the HF tree track skim.

    For each set of columns, we map from input_column_name -> output_column_name.
    The one exception is the _particle_modifications, since it includes an additional
    "level" layer, which allows us to customize the particle columns for each level.
    """

    identifiers: dict[str, str]
    event: dict[str, str]
    _particle_base: dict[str, str]
    _particle_modifications: dict[str, dict[str, str]] = attrs.field(factory=dict)

    @classmethod
    def create(
        cls,
        collision_system: str,
        particle_column_modifications: dict[str, dict[str, str]],
    ) -> Columns:
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
        particle_level_base_columns = {
            **identifiers,
            "ParticlePt": "pt",
            "ParticleEta": "eta",
            "ParticlePhi": "phi",
            "ParticlePID": "particle_ID",
            # This is a JEWEL only field, so we handle adding it in those cases.
            # "Status": "identifier",
        }

        return cls(
            identifiers=identifiers,
            event=event_level_columns,
            particle_base=particle_level_base_columns,
            particle_modifications=particle_column_modifications,
        )

    def particle_level(self, level: str) -> dict[str, str]:
        """Return the particle columns for the given level, including more basic identifiers.

        These more basic identifiers are necessary for reconstructing the event structure,
        but the user generally isn't interested in them.

        Args:
            level: The level for which to return the particle columns.

        Returns:
            The particle columns for the given level.
        """
        particle_columns = copy.deepcopy(self._particle_base)
        # Modify the base column list to add or remove columns as needed
        particle_columns.update(self._particle_modifications.get(level, {}))
        # Remove columns without a map
        particle_columns_to_remove = {k for k, v in particle_columns.items() if v == ""}
        for k in particle_columns_to_remove:
            particle_columns.pop(k)
        return particle_columns

    def standardized_particle_names(self, level: str) -> dict[str, str]:
        """The particle columns for the given level.

        Here, we remove the identifiers, since we don't really want to propagate those
        to the users. They don't need to care about them, and once we've reconstructed
        the event structure, they're not needed.

        Args:
            level: The level for which to return the particle columns.

        Returns:
            The particle columns for the given level.
        """
        return {k: v for k, v in self.particle_level(level=level).items() if k not in self.identifiers}


@attrs.define
class FileSource:
    """HF Tree file source.

    Note:
        This is heavily dependent on the input metadata. It must be defined and included.

    An example of the relevant metadata of a JEWEL fastsim with recoils:

    ```yaml
    some_dataset:
      name: "..."
      collision_system: "..."
      ...
      generator:
        name: "jewel"
        parameters:
          recoils: true
          fastsim: true
      skim_type: "HF_tree_at_LBL"
      skim_parameters:
        levels: ["part_level", "det_level"]
        column_modifications:
          # To add: define the input column name -> output column name.
          #         NOTE: This also covers modifying the default column mapping.
          # To remove: Define the input column name -> empty string
          # In this example, this adds the "Status" -> "identifier" map
          part_level:
            "Status": "identifier"
          det_level:
            "Status": "identifier"
    ```

    """

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
        # Use the metadata to determine what we need to include.
        # NOTE: We're using the metadata that is provided by the user to avoid having to peek
        #       at the file itself. This may not be the ideal balance (since it shifts the burden
        #       to the user), but it's a reasonable starting point. We can always adjust it later.
        parameters = self.metadata["skim_parameters"]
        # Determine what we're working with
        levels = parameters["levels"]
        is_data = "data" in levels
        is_one_level_MC = "part_level" in levels and "det_level" not in levels
        is_two_level_MC = "part_level" in levels and "det_level" in levels
        # FastSim is a special case, so we need to handle it separately
        is_fastsim = parameters.get("fastsim", False)
        # Particle columns modifications
        # Dict of dict: {level: {column_input_name: column_normalized_name}}
        # To add: define the input column name -> output column name.
        #         NOTE: This also covers modifying the default column mapping.
        # To remove: Define the input column name -> empty string
        particle_column_modifications: dict[str, dict[str, str]] = parameters.get("column_modifications", {})
        tree_prefix_override = parameters.get("tree_prefix")

        # Validation
        if is_data and (is_one_level_MC or is_two_level_MC):
            msg = "Data and MC levels are mutually exclusive. Please select only one."
            raise ValueError(msg)
        # This is more of a crosscheck, since this isn't a hard and fast rule
        # We may not need to remove it from every column, but it probably needs to be removed
        # from at least one of the particle columns
        if is_fastsim and all(
            "ParticlePID" not in list(modifications.keys()) for modifications in particle_column_modifications.values()
        ):
            msg = "FastSim files rarely have the ParticlePID column - you probably want to remove it"
            logger.warning(msg)

        # NOTE: We can only load a whole file and chunk it afterwards since the event boundaries
        #       are not known otherwise. Unfortunately, it's hacky, but it seems like the best
        #       bet for now as of Feb 2023.
        columns = Columns.create(
            collision_system=self._collision_system,
            particle_column_modifications=particle_column_modifications,
        )

        # There is a prefix for the original HF tree creator, but not one for the FastSim
        tree_prefix = "" if is_fastsim else "PWGHF_TreeCreator/"
        if tree_prefix_override is not None:
            tree_prefix = tree_prefix_override

        # Grab the various trees. It's less efficient, but we'll have to grab the trees
        # with separate file opens since adding more trees to the UprootSource would be tricky.
        all_sources: dict[str, sources.Source] = {
            "event_level": sources.UprootSource(
                filename=self._filename,
                tree_name=f"{tree_prefix}tree_event_char",
                columns=list(columns.event),
            ),
        }
        # NOTE: This is also where we're defining the "det_level", "part_level", or "data" fields
        if is_data or is_two_level_MC:
            k = "det_level" if is_two_level_MC else "data"
            all_sources[k] = sources.UprootSource(
                filename=self._filename,
                tree_name=f"{tree_prefix}tree_Particle",
                columns=list(columns.particle_level(level="data" if is_data else "det_level")),
            )
        if is_one_level_MC or is_two_level_MC:
            # If we want to rename - e.g. if we're only analyzing this level alone,
            # we can use `loading_data_rename_levels` to change it to `data``.
            k = "part_level"
            all_sources[k] = sources.UprootSource(
                filename=self._filename,
                tree_name=f"{tree_prefix}tree_Particle_gen",
                columns=list(columns.particle_level(level="part_level")),
            )

        _transformed_data = _transform_output(
            # NOTE: We need to grab everything because the file structure is such that we
            #       need to reconstruct the event structure from the fully flat tree.
            #       Consequently, selecting on chunk size would be unlikely to select
            #       the boundaries of events, giving unexpected behavior. So we load it
            #       all now and add on the chunking afterwards. It's less efficient in terms
            #       of memory, but much more straightforward.
            gen_data={k: s.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE) for k, s in all_sources.items()},
            collision_system=self._collision_system,
            columns=columns,
            levels=levels,
        )
        return sources.generator_from_existing_data(
            data=next(_transformed_data),
            chunk_size=chunk_size,
            source_default_chunk_size=self._default_chunk_size,
        )


def _transform_output(
    gen_data: Mapping[str, Generator[ak.Array, sources.T_ChunkSize | None, None]],
    collision_system: str,  # noqa: ARG001
    columns: Columns,
    levels: list[str],
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
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

            # Now, some rearranging the field names for uniformity.
            # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
            # would be required, but apparently not.
            # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
            _result = yield ak.Array(
                {
                    **{
                        level: ak.zip(
                            {
                                v: event_data_in_jagged_format[level][k]
                                for k, v in columns.standardized_particle_names(level=level).items()
                            }
                        )
                        for level in levels
                    },
                    **dict(
                        zip(
                            ak.fields(event_data_in_jagged_format["event_level"]),
                            ak.unzip(event_data_in_jagged_format["event_level"]),
                            strict=True,
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
