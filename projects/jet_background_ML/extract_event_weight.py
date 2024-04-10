"""Extract the average JEWEL event weight from the jet extractor.

This provides a single weight (the mean of the event weight) that we can use to reweight
each JEWEL pt hat bin. This is basically a hack so we can reweight after having done the
embedding (because there was some initial miscommunication about including the event weight
in the tree).  Future embedding won't need this, so we store it as a one-off script.

JEWEL 0-10% no recoils provides:

```
05_15: 2.521352877530242e-05
15_30: 2.4110665427370303e-07
30_45: 1.3567856525392011e-08
45_60: 2.482360195520933e-09
60_80: 5.986645955908365e-10
80_140: 7.583359752139478e-11
```

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import uproot

from mammoth import helpers

logger = logging.getLogger(__name__)


def extract_event_weights(base_dir: Path) -> None:
    jet_R = 0.6

    for pt_hat_bin in [
        "05_15",
        "15_30",
        "30_45",
        "45_60",
        "60_80",
        "80_140",
    ]:
        name = f"JEWEL_NoToy_PbPb_PtHard{pt_hat_bin}.root"
        filename = base_dir / name

        with uproot.open(filename) as f:
            t = f[
                f"JetTree_AliAnalysisTaskJetExtractor_JetPartLevel_AKTChargedR{round(jet_R * 100):03}_mctracks_pT0150_E_scheme_allJets"
            ]

            event_weight = t.arrays(["Event_Weight"])

            logger.info(f"{pt_hat_bin}: {ak.mean(event_weight)}")


if __name__ == "__main__":
    helpers.setup_logging()
    extract_event_weights(base_dir=Path("/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil"))
