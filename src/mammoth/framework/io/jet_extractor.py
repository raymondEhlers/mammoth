"""Convert jet extractor into expected awkward array format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Generator, MutableMapping, Optional

import attrs
import awkward as ak

from mammoth.framework import sources, utils

logger = logging.getLogger(__name__)


@attrs.frozen
class Columns:
    event_level: Dict[str, str]
    particle_level: Dict[str, str]

    @classmethod
    def create(cls) -> "Columns":
        # For JEWEL, these were the only meaningful columns
        event_level_columns = {
            "Event_Weight": "event_weight",
            "Event_ImpactParameter": "event_impact_parameter",
            # Hannah needs this for the extractor bin scaling
            "Jet_Pt": "jet_pt_original",
        }
        particle_columns = {
            "Jet_Track_Pt": "pt",
            "Jet_Track_Eta": "eta",
            "Jet_Track_Phi": "phi",
            "Jet_Track_Charge": "charged",
            "Jet_Track_Label": "label",
        }

        return cls(
            event_level=event_level_columns,
            particle_level=particle_columns,
        )


@attrs.define
class JEWELFileSource:
    _filename: Path = attrs.field(converter=Path)
    # We always want to pull in as many tracks as possible, so take the largest possible R
    _extractor_jet_R: float = attrs.field(default=0.6)
    _entry_range: utils.Range = attrs.field(
        converter=sources.convert_sequence_to_range, default=utils.Range(None, None)
    )
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(
        self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT
    ) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        columns = Columns.create()
        source = sources.UprootSource(
            filename=self._filename,
            tree_name=f"JetTree_AliAnalysisTaskJetExtractor_JetPartLevel_AKTChargedR{round(self._extractor_jet_R * 100):03}_mctracks_pT0150_E_scheme_allJets",
            entry_range=self._entry_range,
            columns=list(columns.event_level) + list(columns.particle_level),
        )
        return _transform_output(
            gen_data=source.gen_data(chunk_size=chunk_size),
        )


def _transform_output(
    gen_data: Generator[ak.Array, Optional[sources.T_ChunkSize], None],
) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
    _columns = Columns.create()

    try:
        data = next(gen_data)
        while True:
            # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
            _result = yield ak.Array(
                {
                    "part_level": ak.zip(
                        dict(
                            zip(
                                # [c.replace("Jet_T", "t").lower() for c in list(particle_columns)],
                                list(_columns.particle_level.values()),
                                ak.unzip(data[_columns.particle_level]),
                            )
                        )
                    ),
                    **dict(
                        zip(
                            list(_columns.event_level.values()),
                            ak.unzip(data[_columns.event_level]),
                        )
                    ),
                }
            )

            # Update for next step
            data = gen_data.send(_result)
    except StopIteration:
        pass


def write_to_parquet(arrays: ak.Array, filename: Path) -> bool:
    """Write the jagged track skim arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # NOTE: If there are issues about reading the files due to arrays being too short, check that
    #       there are no empty events. Empty events apparently cause issues for byte stream split
    #       encoding: https://issues.apache.org/jira/browse/ARROW-13024
    #       Unfortunately, this won't become clear until reading is attempted.
    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Optimize the compression via improved encodings for floats and strings.
        # Conveniently, awkward 2.x will now select the right columns for each if simply set to `True`
        # Optimize for columns with anything other than floats
        parquet_dictionary_encoding=True,
        # Optimize for columns with floats
        parquet_byte_stream_split=True,
    )

    return True


if __name__ == "__main__":
    import mammoth.helpers

    mammoth.helpers.setup_logging(level=logging.INFO)

    # Setup
    JEWEL_identifier = "NoToy_PbPb"
    # Since the PbPb files tend to have a few thousand or fewer events, we want the JEWEL chunk size
    # to be similar to that value. Otherwise, the start of JEWEL files will be embedded far too often,
    # while the ends will never be reached.
    # We also want to keep the num
    # ber of files in check. 5000 seems like a reasonable balance.
    chunk_size = int(5e3)

    # Map from JEWEL identifier to a somewhat clearer name for directories, etc
    JEWEL_label = {
        "NoToy_PbPb": "central_00_10",
        "NoToy_PbPb_3050": "semi_central_30_50",
    }

    for pt_hat_bin in [
        "05_15",
        "15_30",
        "30_45",
        "45_60",
        "60_80",
        "80_140",
    ]:
        filename = Path(
            f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/JEWEL_{JEWEL_identifier}_PtHard{pt_hat_bin}.root"
        )

        # Keep track of iteration
        start = 0
        continue_iterating = True
        index = 0
        while continue_iterating:
            end = start + chunk_size
            logger.info(f"Processing file {filename}, chunk {index} from {start}-{end}")

            source = JEWELFileSource(
                filename=filename,
                # Use jet R = 0.6 because this will contain more of the JEWEL particles.
                # We should be safe to use this for embedding for smaller R jets too, since they
                # should be encompassed within the R = 0.6 jet.
                extractor_jet_R=0.6,
                entry_range=(start, end),
            )
            arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
            # Just for confirmation that it matches the chunk size (or is smaller)
            logger.debug(f"Array length: {len(arrays)}")

            output_dir = filename.parent / "skim" / JEWEL_label[JEWEL_identifier]
            output_dir.mkdir(parents=True, exist_ok=True)
            write_to_parquet(
                arrays=arrays,
                filename=(output_dir / f"{filename.stem}_{index:03}").with_suffix(".parquet"),
            )

            if len(arrays) < (end - start):
                # We're out of entries - we're done.
                break

            # Move up to the next iteration.
            start = end
            index += 1

        logger.info(f"Finished at index {index} for pt hat bin {pt_hat_bin}")

    # import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})
