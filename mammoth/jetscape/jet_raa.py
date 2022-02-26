#!/usr/bin/env python3

""" Jet RAA for comparison with the ALICE jet background ML analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import attr
import awkward as ak
import hist
import numpy as np
import numpy.typing as npt
import uproot

from mammoth import helpers
from mammoth.framework import jet_finding, particle_ID, sources, transform
from mammoth.jetscape import utils


logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class JetLabel:
    jet_R: float = attr.ib()
    label: str = attr.ib()

    def __str__(self) -> str:
        return f"{self.label}_jetR{round(self.jet_R * 100):03}"


def load_data(filename: Path) -> ak.Array:
    logger.info("Loading data")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transforming data")
    return transform.data(arrays=arrays, rename_prefix={"data": "particles"})


def find_jets_for_analysis(arrays: ak.Array, jet_R_values: Sequence[float], particle_column_name: str = "data", min_jet_pt: float = 30) -> Dict[JetLabel, ak.Array]:
    logger.info("Start analyzing")
    # Event selection
    # None for jetscape
    # Track cuts
    logger.info("Track level cuts")
    # Data track cuts:
    # - min: 150 MeV
    data_track_pt_mask = arrays[particle_column_name].pt >= 0.150
    arrays[particle_column_name] = arrays[particle_column_name][data_track_pt_mask]

    # Track selections:
    # - Signal particles vs holes
    signal_particles_mask = arrays[particle_column_name, "status"] == 0
    holes_mask = ~signal_particles_mask
    # - Charged particles only for charged-particle jets
    _charged_hadron_PIDs = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    charged_particles_mask = particle_ID.build_PID_selection_mask(
        arrays[particle_column_name], absolute_pids=_charged_hadron_PIDs
    )

    # We want to analyze both charged and full jets
    particles_signal = arrays[particle_column_name][signal_particles_mask]
    particles_signal_charged = arrays[particle_column_name][signal_particles_mask & charged_particles_mask]
    particles_holes = arrays[particle_column_name][holes_mask]

    # Finally, require that we have particles for each event
    # NOTE: We have to do it in a separate mask because the above is masked as the particle level,
    #       but here we need to mask at the event level. (If you try to mask at the particle, you'll
    #       end up with empty events)
    # NOTE: We store the mask because we need to apply it to the holes when we perform the subtraction below
    event_has_particles_signal = ak.num(particles_signal, axis=1) > 0
    particles_signal = particles_signal[event_has_particles_signal]
    event_has_particles_signal_charged = ak.num(particles_signal_charged, axis=1) > 0
    particles_signal_charged = particles_signal_charged[event_has_particles_signal_charged]

    # Jet finding
    logger.info("Find jets")
    # Always use the pp jet area because we aren't going to do subtraction via fastjet
    area_settings = jet_finding.AREA_PP
    jets = {}
    # NOTE: The dict comprehension that was here previously was cute, but it made it harder to
    #       debug issues, so we use a standard set of for loops here instead
    for jet_R in jet_R_values:
        #for particles, label in zip([particles_signal, particles_signal_charged], ["full", "charged"]):
        for particles, label in zip([particles_signal_charged, particles_signal], ["charged", "full"]):
            tag = JetLabel(jet_R=jet_R, label=label)
            logger.info(f"label: {tag}")
            jets[tag] = jet_finding.find_jets(
                particles=particles,
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=area_settings,
                min_jet_pt=min_jet_pt,
            )

    # Calculated the subtracted pt due to the holes.
    for jet_label, jet_collection in jets.items():
        jets[jet_label]["pt_subtracted"] = utils.subtract_holes_from_jet_pt(
            jets=jet_collection,
            # NOTE: There can be different number of events for full vs charged jets, so we need
            #       to apply the appropriate event mask to the holes
            particles_holes=particles_holes[
                event_has_particles_signal_charged if jet_label.label == "charged" else event_has_particles_signal
            ],
            jet_R=jet_label.jet_R,
            builder=ak.ArrayBuilder(),
        ).snapshot()

    # Store the cross section with each jet. This way, we can flatten from events -> jets
    for jet_label, jet_collection in jets.items():
        # Before any jets cuts, add in cross section
        # NOTE: There can be different number of events for full vs charged jets, so we need
        #       to apply the appropriate event mask to the holes
        jets[jet_label]["cross_section"] = arrays["cross_section"][
                event_has_particles_signal_charged if jet_label.label == "charged" else event_has_particles_signal
            ]

    # Apply jet level cuts.
    # Fiducial cuts for ALICE full jets
    for jet_label, jet_collection in jets.items():
        if jet_label.label == "full":
            fiducial_mask = np.abs(jet_collection.eta) <= (0.7 - jet_label.jet_R)
            jets[jet_label] = jet_collection[fiducial_mask]

    return jets


def write_tree(jets: ak.Array, filename: Path) -> bool:
    # Fields of interest. These shouldn't vary between each jet label
    fields = {
        "px": np.float32,
        "py": np.float32,
        "pz": np.float32,
        "E": np.float32,
        "area": np.float32,
        "pt_subtracted": np.float32,
        "cross_section": np.float32,
    }

    for jet_label in jets:
        # First, reduce the precision to save some space (and only save the required branches)
        any_jets = (len(ak.flatten(jets[jet_label])) != 0)
        arrays = ak.zip({
            # If there are jets, then take the usual field. If not, take the cross_section
            # because the type seems to be in tact. This is for sure a hack, but it should get
            # the job done.
            k: ak.values_astype(v if any_jets else jets[jet_label]["cross_section"], dtype)
            for (k, dtype), v in zip(fields.items(), ak.unzip(jets[jet_label][list(fields)]))
        })
        # NOTE: We don't want to flatten here because otherwise we lose the overall number of events.
        #       Plus, this ensures compatibility with the standard analysis.

        # Trying to write an empty file causes problems because the types are undetermined
        # NOTE: This kind of sucks because we lose the event count by doing this. So instead we use the hack above.
        # if len(ak.flatten(arrays)) == 0:
        #     logger.info(f"No jets found for {jet_label} - skipping writing tree to file.")
        #     import IPython; IPython.embed()
        #     #continue

        # Name of collection: path / jet_collection / filename
        # eg: ../jetSkim/charged_jetR040/JetsBin7_9_00.parquet
        jet_collection_filename = filename.parent / str(jet_label) / filename.name
        # Ensure the output directory is always available
        jet_collection_filename.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing columns: {ak.fields(arrays)}")
        if filename.suffix == ".parquet":
            logger.info(f"Writing to parquet file: {jet_collection_filename}")
            # Everything is floats at the moment, so there are no dictionary encoded fields
            dict_fields: List[str] = []
            ak.to_parquet(
                arrays,
                jet_collection_filename,
                compression="zstd",
                compression_level=None,
                explode_records=False,
                # Additional parquet options are based on https://stackoverflow.com/a/66854439/12907985
                # Default to byte stream split, skipping those which are specified to use dictionary encoding
                use_dictionary=dict_fields,
                use_byte_stream_split=True,
            )
        else:
            logger.info(f"Writing to root file: {jet_collection_filename}")
            # Write with uproot
            try:
                with uproot.recreate(jet_collection_filename) as f:
                    #f["tree"] = arrays if len(ak.flatten(arrays)) > 0 else ak.Array({k: [] * len(arrays) for k in ak.fields(arrays)})
                    f["tree"] = arrays
            except Exception as e:
                logger.exception(e)
                raise e from ValueError(
                    f"Jet label: {jet_label}, {arrays.type}, {arrays}"
                    f"\n{jets[jet_label][list(fields)]}"
                    f"\n{ak.flatten(jets[jet_label][list(fields)])}"
                    f"\n{len(ak.flatten(arrays['px']))}"
                )

    return True


def read_jet_skims(filename: Path, jet_R_values: Sequence[float]) -> Dict[JetLabel, ak.Array]:
    jet_inputs = {}
    for jet_R in jet_R_values:
        for label in ["charged", "full"]:
            tag = JetLabel(jet_R=jet_R, label=label)
            jet_collection_filename = filename.parent / str(tag) / filename.name
            try:
                with uproot.open(jet_collection_filename) as f:
                    jet_inputs[tag] = f["tree"].arrays()
            except IOError as e:
                logger.info(f"Skipping {tag} due to IO error {e}")
                continue

    return jet_inputs


def analyze_jets(arrays: ak.Array, jets: Mapping[JetLabel, ak.Array]) -> Dict[str, hist.Hist]:
    # Define hists
    hists = {}
    hists["n_events"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
    hists["n_events_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
    for jet_label in jets:
        hists[f"{jet_label}_jet_pt"] = hist.Hist(hist.axis.Regular(1000, 0, 1000, label="jet_pt"), storage=hist.storage.Weight())
        # Try a coarser binning to reduce outliers
        hists[f"{jet_label}_jet_pt_coarse_binned"] = hist.Hist(hist.axis.Regular(200, 0, 1000, label="jet_pt"), storage=hist.storage.Weight())
        # This is assuredly overkill, but it hopefully means that I won't need to mess with it anymore
        hists[f"{jet_label}_n_events"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_events_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_jets"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_jets_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))

    # Now, fill the hists
    hists["n_events"].fill(0, weight=len(arrays))
    # Just need it to get the first cross section value. It should be the same for all cases
    first_jet_label = next(iter(jets))
    # NOTE: Apparently this can be empty, so we have to retrieve the value carefully
    first_cross_section = ak.flatten(jets[first_jet_label].cross_section)
    if len(first_cross_section) == 0:
        _cross_section_weight_factor = 1
    else:
        _cross_section_weight_factor = first_cross_section[0]
    hists["n_events_weighted"].fill(0, weight=len(arrays) * _cross_section_weight_factor)
    for jet_label, jet_collection in jets.items():
        hists[f"{jet_label}_jet_pt"].fill(
            ak.flatten(jet_collection.pt_subtracted), weight=ak.flatten(jet_collection.cross_section)
        )
        hists[f"{jet_label}_jet_pt_coarse_binned"].fill(
            ak.flatten(jet_collection.pt_subtracted), weight=ak.flatten(jet_collection.cross_section)
        )
        hists[f"{jet_label}_n_events"].fill(0, weight=len(jet_collection))
        hists[f"{jet_label}_n_events_weighted"].fill(0, weight=len(jet_collection) * _cross_section_weight_factor)
        hists[f"{jet_label}_n_jets"].fill(0, weight=len(ak.flatten(jet_collection.pt_subtracted)))
        hists[f"{jet_label}_n_jets_weighted"].fill(0, weight=len(ak.flatten(jet_collection.pt_subtracted)) * _cross_section_weight_factor)

    return hists


def write_hists(hists: Dict[str, hist.Hist], filename: Path) -> bool:
    filename.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing hists to {filename}")
    with uproot.recreate(filename) as f:
        for k, v in hists.items():
            f[k] = v

    return True


def run(arrays: ak.Array,
        read_jet_skim_from_file: bool,
        min_jet_pt: float = 5,
        jet_R_values: Optional[Sequence[float]] = None,
        jets_skim_filename: Optional[Path] = None,
        write_hists_filename: Optional[Path] = None) -> Dict[str, hist.Hist]:
    # Validation
    if jet_R_values is None:
        jet_R_values = [0.2, 0.4, 0.6]
    if read_jet_skim_from_file and jets_skim_filename is None:
        raise ValueError("If reading jet skim from file, must pass jets skims filename")

    # Find jets
    if read_jet_skim_from_file:
        # Help out mypy...
        assert jets_skim_filename is not None
        jets = read_jet_skims(
            filename=jets_skim_filename,
            jet_R_values=jet_R_values,
        )
    else:
        jets = find_jets_for_analysis(
            arrays=arrays,
            jet_R_values=jet_R_values,
            min_jet_pt=min_jet_pt,
        )

    if jets_skim_filename and not read_jet_skim_from_file:
        write_tree(jets=jets, filename=jets_skim_filename)

    # Analyze the jets
    hists = analyze_jets(arrays=arrays, jets=jets)

    if write_hists_filename:
        write_hists(hists=hists, filename=write_hists_filename)

    return hists


if __name__ == "__main__":
    # Basic setup
    helpers.setup_logging(level=logging.INFO)

    hists = run(
        arrays=load_data(
            #Path(f"/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim/test/JetscapeHadronListBin7_9_00.parquet")
            #Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim/JetscapeHadronListBin270_280_01.parquet")
            #Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/skim/JetscapeHadronListBin7_9_01.parquet"),
            #Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/skim/JetscapeHadronListBin270_280_01.parquet"),
            Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/skim/JetscapeHadronListBin1_2_07.parquet"),
        ),
        read_jet_skim_from_file=False,
        # Low for testing
        min_jet_pt=5,
        # Jet one R for faster testing
        jet_R_values=[0.2],
        #write_jets_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/jetRAA/test/jetsSkim/JetsBin270_280_01.parquet"),
        #write_hists_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/jetRAA/test/hists/hists_Bin270_280_01.root"),
        #write_jets_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/jetsSkim/JetsBin7_9_01.parquet"),
        #write_jets_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/jetRAA/test/jetsSkim/JetsBin270_280_01.root"),
        #write_hists_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/jetRAA/test/hists/hists_Bin270_280_01.root"),
        jets_skim_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/jetRAA/test/jetsSkim/JetsBin1_2_07.root"),
        write_hists_filename=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/jetRAA/test/hists/hists_Bin1_2_07.root"),
    )

    import IPython

    IPython.embed()