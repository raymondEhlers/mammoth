from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _setup_afterburner(code_dir: Path) -> None:
    """ Setup afterburner code to be run via ROOT """
    # Delayed import to avoid direct dependence.
    from mammoth.framework import root_utils
    ROOT = root_utils.import_ROOT()

    # Validation
    code_dir = Path(code_dir)

    # Nominally setup for MT. It's not really going to do us any good here, but it doesn't hurt anything.
    # NOTE: We do need to specify 1 to ensure that we don't use extra cores.
    ROOT.ROOT.EnableImplicitMT(1)
    # Load external libraries
    #ROOT.gSystem.Load("libRooUnfold")
    # Load the tree processing code utilities. This requires passing the path
    tree_processing_cxx = code_dir / "treeProcessing.C"
    # We only want to load it if it hasn't been already, so we use the `treeProcessing` function
    # as a proxy for this. Loading it twice appears to cause segfaults in some cases.
    if not hasattr(ROOT, "treeProcessing"):
        ROOT.gInterpreter.ProcessLine(f""".L {tree_processing_cxx!s} """)


def _array_to_ROOT(arr: list[str] | npt.NDArray[Any], type_name: str = "double") -> Any:
    """Convert numpy array to std::vector via ROOT.

    Because it apparently can't handle conversions directly. Which is really dumb...

    In principle, we could convert the numpy dtype into the c++ type, but that's a lot of mapping
    to be done for a function that (hopefully) isn't used so often. So we let the user decide.

    Args:
        arr: Numpy array to be converted.
        type_name: c++ type name to be used for the vector. Default: "double".
    Returns:
        std::vector containing the numpy array values.
    """
    from mammoth.framework import root_utils
    ROOT = root_utils.import_ROOT()

    vector = ROOT.std.vector(type_name)()
    for a in arr:
        vector.push_back(a)
    return vector


def run_afterburner(
    tree_processing_code_directory: Path,
    input_file: Path,
    geometry_file: Path,
    output_identifier: str,
    output_dir: Path,
    do_reclustering: bool = True,
    do_jet_finding: bool = True,
    do_calo_res: bool = False,
    is_single_particle_production: bool = False,
    max_n_events: int = -1,
    verbosity: int = 0,
    primary_track_source: int = 0,
    remove_tracklets: bool = False,
    track_projections_are_broken: bool = True,
    jet_algorithm: str = "anti-kt",
    jet_R_parameters: Sequence[float] = [0.3, 0.5, 0.8, 1.0],
    max_track_pt_in_jet: float = 30.0,
) -> tuple[bool, str]:
    # Setup
    _setup_afterburner(code_dir=tree_processing_code_directory)

    from mammoth.framework import root_utils
    ROOT = root_utils.import_ROOT()

    ROOT.treeProcessing(
        str(input_file),
        str(geometry_file),
        str(output_identifier),
        str(output_dir),
        max_n_events,
        do_reclustering,
        do_jet_finding,
        do_calo_res,
        is_single_particle_production,
        verbosity,
        primary_track_source,
        remove_tracklets,
        track_projections_are_broken,
        jet_algorithm,
        _array_to_ROOT(np.array(jet_R_parameters), "double"),
        max_track_pt_in_jet,
    )

    return True, f"Success for {output_identifier} with input file {input_file}"
