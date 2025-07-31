"""Write posterior stored in a root file into a text file.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from mammoth import helpers

logger = logging.getLogger(__name__)


def main() -> None:
    """One off script to write the posterior stored in a ROOT file into a text file."""
    helpers.setup_logging(level=logging.INFO)

    _here = Path(__file__).parent
    filename = _here / Path("JETSCAPE_FigureQHatWithQ2_Q0_and_2.7.root")

    with uproot.open(filename) as f:
        # Extract the relevant values
        x_values_median, y_values_median = f["G50_0"].values()
        # NOTE: The 90% wraps around in a circle, so we need to account for this...
        x_values_90_percent, y_values_90_percent = f["G90_0"].values()
        x_values_lower_5_percent = x_values_90_percent[: len(x_values_90_percent) // 2]
        y_values_lower_5_percent = y_values_90_percent[: len(x_values_90_percent) // 2]
        # -1 cuts off the closing of the loop
        # NOTE: Need the reverse because it goes in a loop around the
        x_values_upper_95_percent = x_values_90_percent[len(x_values_90_percent) // 2 : -1][::-1]
        y_values_upper_95_percent = y_values_90_percent[len(x_values_90_percent) // 2 : -1][::-1]

        # test to confirm that all of our values align as expected...
        assert np.allclose(x_values_median, x_values_lower_5_percent)
        assert np.allclose(x_values_lower_5_percent, x_values_upper_95_percent)

        # Restrict to only values with 0.150 <= T <= 0.50, which is what we actually reported
        mask = (x_values_median >= 0.150) & (x_values_median <= 0.50)

        # Write this using numpy as x_values, y_values_lower_5_percent, median, y_values_upper_95_percent
        output_values = np.column_stack(
            (
                x_values_median[mask],
                y_values_lower_5_percent[mask],
                y_values_median[mask],
                y_values_upper_95_percent[mask],
            )
        )
        output_filename = _here / "JETSCAPE_FigureQHatWithQ2_Q0.dat"
        np.savetxt(output_filename, output_values, header="x, y_lower, median, y_upper", fmt="%.6f")
        logger.info(f"Wrote posterior to {output_filename}s")

    # Double check for safety by plotting
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 8),
    )

    ax.plot(output_values[:, 0], output_values[:, 2], linewidth=3)
    ax.fill_between(output_values[:, 0], y1=output_values[:, 1], y2=output_values[:, 3], alpha=0.3)
    fig.savefig(_here / "qhat_confirmation.pdf")
    logger.info(f"Wrote {_here / 'qhat_confirmation.pdf'} to confirm values are reasonable.")


if __name__ == "__main__":
    main()
