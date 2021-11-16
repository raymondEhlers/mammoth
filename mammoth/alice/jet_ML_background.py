"""Analysis code related to the jet ML background analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, Tuple

import attr
import awkward as ak
import numpy as np

from mammoth.framework import models, sources, transform
from mammoth.framework.normalize_data import jet_extractor, track_skim

logger = logging.getLogger(__name__)


def load_embedding(signal_filename: Path, background_filename: Path, background_collision_system_tag: str) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    signal_source = jet_extractor.JEWELSource(
        filename=signal_filename,
    )
    fast_sim_source = sources.ChunkSource(
        # The chunk size will be determined by the number of background events
        chunk_size=-1,
        sources=sources.ALICEFastSimTrackingEfficiency(
            particle_level_data=signal_source.data(),
            fast_sim_parameters=models.ALICEFastSimParameters(
                event_activity=models.ALICETrackingEfficiencyEventActivity.central_00_10,
                period=models.ALICETrackingEfficiencyPeriod.LHC15o,
            )
        ),
        repeat=True,
    )
    # Background
    # For embedding, we will always be embedding into PbPb
    background_source = track_skim.FileSource(filename=background_filename, collision_system=background_collision_system_tag)

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"background": background_source},
        chunked_sources={"signal": fast_sim_source},
        source_index_identifiers=source_index_identifiers,
    )

    logger.info("Transforming embedded")
    arrays = combined_source.data()
    #signal_fields = ak.fields(arrays["signal"])
    # Empty mask
    # TODO: Confirm that this masks as expected...
    #mask = arrays["signal"][signal_fields[0]] * 0 >= 0
    mask = np.ones(len(arrays)) > 0
    # NOTE: We can apply the signal selections in the analysis task later

    # Only apply background event selection if applicaable
    background_fields = ak.fields(arrays["background"])
    if "is_ev_rej" in background_fields:
        mask = mask & (arrays["background", "is_ev_rej"] == 0)
    if "z_vtx_reco" in background_fields:
        mask = mask & (np.abs(arrays["background", "z_vtx_reco"]) < 10)

    # Finally, apply selection
    arrays = arrays[mask]

    return source_index_identifiers, transform.embedding(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )


if __name__ == "__main__":
    import mammoth.helpers
    mammoth.helpers.setup_logging()

    JEWEL_identifier = "NoToy_PbPb"
    pt_hat_bin = "80_140"
    index = "000"
    logger.info("Loading embedding...")
    source_index_identifiers, arrays = load_embedding(
        signal_filename=Path(f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/JEWEL_{JEWEL_identifier}_PtHard{pt_hat_bin}_{index}.parquet"),
        background_filename=Path("/alf/data/rehlers/substructure/trains/PbPb/7666/run_by_run/LHC15o/246087/AnalysisResults.15o.825.root"),
        background_collision_system_tag="PbPb_central",
    )

    import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})