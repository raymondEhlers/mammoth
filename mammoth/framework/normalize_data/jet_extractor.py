"""Convert jet extractor into expected awkward array format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)

def jet_extractor_to_awkward(
    filename: Path,
    jet_R: float,
    entry_range: Optional[Tuple[int, int]] = None,
) -> ak.Array:
    # For JEWEL, these were the only meaningful columns
    event_level_columns = {
        "Event_Weight": "event_weight",
        "Event_ImpactParameter": "event_impact_parameter",
    }
    particle_columns = {
        "Jet_Track_Pt": "pt",
        "Jet_Track_Eta": "eta",
        "Jet_Track_Phi": "phi",
        "Jet_Track_Charge": "charged",
        "Jet_Track_Label": "label",
    }

    additional_uproot_source_kwargs = {}
    if entry_range is not None:
        additional_uproot_source_kwargs = {
            "entry_range": entry_range
        }

    data = sources.UprootSource(
        filename=filename,
        tree_name=f"JetTree_AliAnalysisTaskJetExtractor_JetPartLevel_AKTChargedR{round(jet_R * 100):03}_mctracks_pT0150_E_scheme_allJets",
        columns=list(event_level_columns) + list(particle_columns),
        **additional_uproot_source_kwargs,  # type: ignore
    ).data()

    return ak.Array({
        "part_level": ak.zip(
            dict(
                zip(
                    #[c.replace("Jet_T", "t").lower() for c in list(particle_columns)],
                    list(particle_columns.values()),
                    ak.unzip(data[particle_columns]),
                )
            )
        ),
        **dict(
            zip(
                list(event_level_columns.values()),
                ak.unzip(data[event_level_columns]),
            )
        ),
    })


def write_to_parquet(arrays: ak.Array, filename: Path) -> bool:
    """Write the jagged track skim arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Determine the types for improved compression when writing
    # See the notes in track_skim for why some choices are made.
    # Columns to store as integers
    dictionary_encoded_columns = [
        # NOTE: Uses notation from arrow/parquet
        #       `list.item` basically gets us to an column in the list.
        #       This may be a little brittle, but let's see.
        "part_level.list.item.label",
    ]

    # Columns to store as float
    byte_stream_split_columns = [
        "event_weight",
        "event_impact_parameter",
        f"part_level.list.item.pt",
        f"part_level.list.item.eta",
        f"part_level.list.item.phi",
        f"part_level.list.item.charge",
    ]

    # NOTE: If there are issues about reading the files due to arrays being too short, check that
    #       there are no empty events. Empty events apparently cause issues for byte stream split
    #       encoding: https://issues.apache.org/jira/browse/ARROW-13024
    #       Unfortunately, this won't become clear until reading is attempted.
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
    from mammoth import helpers
    helpers.setup_logging()

    # TODO: Skim to parquet to get the file sizes more uniform.
    #       Then, make the JEWEL files a chunk source, and pass that into the embedding as a chunk source
    #       So that way, the background dictates the number of events

    chunk_size = int(1e5)
    filename = Path("/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/JEWEL_NoToy_PbPb_PtHard80_140.root")

    # Keep track of iteration
    start = 0
    continue_iterating = True
    index = 0
    while continue_iterating:
        end = start + chunk_size
        logger.info(f"Processing chunk {index} from {start}-{end}")

        arrays = jet_extractor_to_awkward(
            filename=filename,
            jet_R=0.6,
            entry_range=(start, end),
        )
        logger.info(f"Array length: {len(arrays)}")

        output_dir = filename.parent / "skim"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_to_parquet(
            arrays=arrays,
            filename=(output_dir / f"{filename.stem}_{index:03}").with_suffix('.parquet'),
        )

        if len(arrays) < (end - start):
            # We're out of entries - we're done.
            break

        # Move up to the next iteration.
        start = end
        index += 1

    logger.info(f"Finished at index {index}")

    import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})