"""Generate run_info.yaml files to integrate with the existing infrastructure.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

try:
    from mammoth.helpers import setup_logging
except ImportError:

    def setup_logging(level: int) -> None:
        # First, setup logging
        FORMAT = "%(message)s"
        # NOTE: For shutting up parsl, it's import to set the level both in the logging setup
        #       as well as in the handler.
        logging.basicConfig(
            level=level,
            format=FORMAT,
            datefmt="[%X]",
        )

        # Capture warnings into logging
        logging.captureWarnings(True)


logger = logging.getLogger(__name__)

# Just extracted these values from how Dani ran the simulations
_centrality_dependence_of_N_events_per_task = {
    (0, 5): 5630,
    (5, 10): 4370,
    (10, 20): 5000,
    (20, 30): 5000,
    (30, 40): 5000,
    None: 75000,
}


def load_design_point_values(p: Path) -> npt.NDArray[np.float64]:
    return np.loadtxt(p)


def generate_run_info(
    base_path: Path,
    facility: str,
    run_number: int,
    design_point_index: int,
    centrality: tuple[int, int] | None,
    design_point_values: npt.NDArray[np.float64] | None,
) -> Path:
    """Generate the run info for a particular run."""
    # Calculate useful parameters
    run_number_formatted = f"Run{run_number:04d}"
    collision_system = "pp" if centrality is None else "PbPb"

    # Determine the parameters and fields that are needed
    run_info = {
        # Calculation type:
        #   str: `jet_energy_loss` or `pp_baseline`
        "calculation_type": "jet_energy_loss" if collision_system != "pp" else "pp_baseline",
        # Soft sector type:
        #   str: `real_time_hydro` or `precomputed_hydro` (or `N/A` for pp)
        "soft_sector_execution_type": "precomputed_hydro" if collision_system != "pp" else "N/A",
        # Pythia process:
        #   str: `hard_qcd`, `prompt_photon`, or `ew_boson`
        "pythia_process": "hard_qcd",
        # Run number:
        #   int: Run number
        "run_number": run_number,
        # sqrt_s:
        #   int: sqrt_s in GeV
        "sqrt_s": 5020,
        # Number of events per job:
        #   int: n_events
        "n_events_per_task": _centrality_dependence_of_N_events_per_task[centrality],
        # Number of events per parquet file:
        #   int: n_events. We have this hard coded
        "n_events_per_parquet_file": 6000,
        # Power law exponent
        #   float: Value
        "power_law": 4.0,
        # Minimum pt hat
        #   float: Value
        "min_pt_hat": 4.0,
        # Centrality
        #   tuple(float, float): Minimum and maximum values of the centrality. Only specified for AA
        "centrality": [],
        # Parametrization
        #   dict of parametrization info. Only specified for AA. See below
        "parametrization": {},
        # Soft sector parameters that we don't need for hybrid-bayesian productions as of 2026 March.
        # However, they're expected to have values, so we just generate them.
        # We have event-by-event hydro, so we don't need this map.
        "index_to_hydro_event": {},
        # NOTE(RJE): 2026 March, we could specify these the two reuse hydro parameters, but it depends
        #            on how Dani runs the simulations, which RJE doesn't know, and isn't important enough
        #            to find out right now.
        "set_reuse_hydro": None,
        "n_reuse_hydro": None,
        "skip_afterburner": None,
        "number_of_repeated_sampling": None,
        "write_qnvector": False,
    }

    if collision_system != "pp":
        run_info["centrality"] = list(centrality)
        run_info["parametrization"] = {
            # Parametrization type
            #   str: value
            # NOTE(RJE): This isn't necessarily the right label, but it's fine as of 2026 March since we only
            #            have one parametrization
            "type": "2026_03_standard",
            # Design point index
            #   int: Index of the design point run in this simulation
            "design_point_index": design_point_index,
            # Parametrization values
            #   dict: str -> float, mapping from parameter names to values
            "parametrization_values": {
                "L_res": float(design_point_values[design_point_index, 0]),
                "kappa_sc": float(design_point_values[design_point_index, 1]),
            },
        }

    # _default_run_info = {
    #    #"calculation_type": execution_config.type.label(),
    #    #"soft_sector_execution_type": execution_config.soft_sector_execution_type.label(),
    #    #"pythia_process": execution_config.process.label(),
    #    #"run_number": parameter_config.run_number,
    #    #"sqrt_s": parameter_config.sqrt_s,
    #    #"n_events_per_task": parameter_config.n_events_per_task,
    #    #"n_events_per_parquet_file": execution_config.n_events_per_parquet_file,
    #    #"power_law": power_law_power,
    #    #"min_pt_hat": min_pt_hat_map[parameter_config.sqrt_s],
    #    # These are only applicable for JEL, but it's easiest to just write an empty map or list when it's not applicable.
    #    #"centrality": [],
    #    #"parametrization": {},
    #    # These parameters are related to the soft sector. We'll always write them for simplicity.
    #    "index_to_hydro_event": {},
    #    "set_reuse_hydro": None,
    #    "n_reuse_hydro": None,
    #    "skip_afterburner": None,
    #    "number_of_repeated_sampling": None,
    #    "write_qnvector": None
    # }
    """
    ### AA
    calculation_type: jet_energy_loss
    centrality:
    - 0
    - 10
    index_to_hydro_event:
        ....
    min_pt_hat: 4.0
    n_events_per_parquet_file: 5000
    n_events_per_task: 4800
    parametrization:
      design_point_index: 77
      parametrization_values:
        A: 7.85353449260192
        B: 2.00906462523193
        C: 8.19219483058415
        D: 74.5781420242299
        alpha_s: 0.40498248406137
        q_switch: 9.88027913723634
        t_start: 0.914045444231991
      type: binomial
    power_law: 4.0
    run_number: 8096
    sqrt_s: 2760

    # Addition for JEL calculations
    {
        "centrality": list(parameter_config.centrality_range),
        "parametrization": {
            "type": parameter_config.parametrization_type.name,
            "design_point_index": parameter_config.design_point_index,
            "parametrization_values": _parametrization,
        },
    }

    #### pp
    calculation_type: pp_baseline
    centrality: []
    index_to_hydro_event: {}
    """

    # Write the output to appropriate location
    filename = base_path / f"{facility}" / run_number_formatted / f"{run_number_formatted}_info.yaml"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w") as f:
        yaml.dump(run_info, f, sort_keys=False)

    return filename


def generate_many_run_info(
    facility: str,
    start_index: int,
    end_index: int,
    centrality: tuple[int, int] | None,
) -> None:
    """Generate many run info files according to the specification."""
    # Validation
    # Watch out for pp (where centrality is None), but many design points are passed. That's probably a mistake, so notify the user
    if centrality is None and end_index - start_index > 1:
        msg = f"Seems like you're working on pp, but you're provided more than one design points. Provided `{end_index - start_index}`. Did you mean to configure it this way?"
        raise ValueError(msg)

    # Setup
    design_point_values = load_design_point_values(Path("sandbox/StandardDesign_2026_03.dat"))
    base_path = Path("run_info")

    # WARNING:
    # This assumes that the design point aligns with the run number!!
    # RJE arranged for this to be the case, but be sure to confirm that's the case!
    for i, run_number in enumerate(range(start_index, end_index)):
        res = generate_run_info(
            base_path=base_path,
            facility=facility,
            run_number=run_number,
            design_point_index=i,
            centrality=centrality,
            design_point_values=design_point_values,
        )
        logger.info(f"Wrote Run{run_number:04g} to {res}")


def main_entry_point() -> None:
    # Setup
    setup_logging(level=logging.INFO)

    # Arguments
    parser = argparse.ArgumentParser(description="Generate standalone slurm scripts")
    parser.add_argument(
        "-f",
        "--facility",
        action="store",
        type=str,
        required=True,
        metavar="facility",
        help="Name of the facility where the runs were generated.",
    )
    parser.add_argument(
        "-r",
        "--run-numbers",
        nargs=2,
        action="store",
        type=int,
        required=True,
        metavar=("start_index", "end_index"),
        help="Starting (ending) run number. (Implicitly, design point 0 must start with the first run number and count up). n.b. It's [start, end), so be sure to set the end range +1 for the last needed value.",
    )
    parser.add_argument(
        "-c",
        "--centrality",
        nargs=2,
        action="store",
        type=int,
        default=None,
        metavar=("low", "high"),
        help="Centrality range (low, high). Default: None, which assumes pp",
    )

    args = parser.parse_args()

    # Summarize status
    status = f"""Generating Run Info for:
\tfacility: "{args.facility}"
\tRun range: {args.run_numbers[0]} - {args.run_numbers[1]}
\tCentrality: {tuple(args.centrality) if args.centrality is not None else "N/A"}"""
    logger.info(status)

    generate_many_run_info(
        facility=args.facility,
        start_index=args.run_numbers[0],
        end_index=args.run_numbers[1],
        centrality=tuple(args.centrality) if args.centrality is not None else None,
    )


if __name__ == "__main__":
    main_entry_point()
