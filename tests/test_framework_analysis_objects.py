"""Tests for analysis objects.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
from pachyderm import yaml

from mammoth.framework.analysis import objects as analysis_objects

logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"


def test_round_trip_serialization() -> None:
    """Check serialization, leveraging the existing track skim validation scale factors for convenience."""

    # Start with a consistent input
    # Basically reimplements `analysis_objects.read_extracted_scale_factors`, but we do it
    # separately to ensure that we're in full control
    path = Path(_track_skim_base_path / "input" / "LHC20g4_AOD_2640_scale_factors.yaml")
    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    with path.open() as f:
        scale_factors: dict[int, analysis_objects.ScaleFactor] = y.load(f)

    with tempfile.NamedTemporaryFile() as output_file:
        y.dump(scale_factors, output_file)
        output_file.seek(0)
        round_trip_scale_factors = y.load(output_file)

    assert scale_factors == round_trip_scale_factors


def test_extract_scale_factors_uproot_vs_root() -> None:
    from mammoth.alice import scale_factors

    ROOT = pytest.importorskip("ROOT")  # noqa: F841

    # This is artificial, but it gives us multiple files to work with...
    files = [
        Path(_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root"),
        Path(_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R040.root"),
        Path(_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R040.root"),
        Path(_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R040.root"),
        Path(_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R040.root"),
    ]
    # Local test - can't really commit these files...
    # Fails
    # files = list(Path("trains/pythia/2907/run_by_run/LHC23a3_cent_woSDD/282306/20").glob("*.root"))[:2]
    # Works
    # files = list(Path("trains/pythia/2907/run_by_run/LHC23a3_cent_woSDD/282306/20").glob("*.root"))[1:2]
    logger.warning(f"{files=}")

    sf_uproot = scale_factors.scale_factor_uproot(files)
    scale_factor_uproot = analysis_objects.ScaleFactor.from_hists(*sf_uproot)

    # Write out for testing...
    # import uproot
    # with uproot.recreate("tests/test_scale_factors_uproot.root") as f:
    #     f["x_sec"] = sf_uproot[2]
    #     f["n_trials"] = sf_uproot[3]

    sf_root = scale_factors.scale_factor_ROOT(files, "AliAnalysisTaskTrackSkim_pythia")
    scale_factor_ROOT = analysis_objects.ScaleFactor.from_hists(*sf_root)

    # Write out for testing...
    # with ROOT.TFile.Open("tests/test_scale_factors_ROOT.root", "RECREATE") as f_ROOT:
    #     sf_root[2].Write("x_sec")
    #     sf_root[3].Write("n_trials")

    assert scale_factor_uproot.value() == scale_factor_ROOT.value()
    assert scale_factor_uproot == scale_factor_ROOT
