"""Tests for the framework.io module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
import pytest
from pathlib import Path
from typing import Any

from mammoth.framework import io
from mammoth.framework import production

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "module, production_name", [
        ("jetscape", "JETSCAPE_PP19_1"),
        ("HF_tree", "LBL20_Pythia_FastSim_1_258314"),
        ("track_skim", "LHC18qr_pass3_AOD252_central_642"),
    ], ids=["jetscape", "HF_tree", "track_skim"]
)
def test_file_source(caplog: Any, module: str, production_name: str) -> None:
    # Setup
    # This is quite hacky, but it's convenient, so good enough
    mammoth_src = Path(production.__file__).parent.parent
    track_skim_config = production._read_full_config(config_path=mammoth_src / "alice" / "config" / "track_skim_config.yaml")

    file_source = io.file_source(file_source_config=track_skim_config["skimmed_datasets"][production_name])

    # Check that it's the correct type
    assert file_source.func.__name__ == "FileSource"  # type: ignore[attr-defined]
    assert file_source.func.__module__ == f"mammoth.framework.io.{module}"  # type: ignore[attr-defined]

    # Since it returns a partial, we can check what arguments are bound
    assert isinstance(file_source(filename=Path("test.root")), io.sources.Source)

