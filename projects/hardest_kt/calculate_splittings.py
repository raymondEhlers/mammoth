"""Playground for calculating made up splitting trees

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt

from mammoth import helpers

logger = logging.getLogger(__name__)

@attrs.define
class Splitting:
    z: float
    delta_R: float

    def kt(self, pt_parent: float) -> float:
        # Approx
        #return pt_parent * self.z * self.delta_R  # type: ignore[no-any-return]
        # Exact
        return pt_parent * self.z * np.sin(self.delta_R)  # type: ignore[no-any-return]


def calculate_splittings(splittings: list[Splitting], input_jet_pt: float) -> None:

    pt_parent = input_jet_pt
    for i, splitting in enumerate(splittings):
        logger.info(f"{i}: {splitting}, pt_parent: {pt_parent} -> kt={splitting.kt(pt_parent=pt_parent):.3f}")
        pt_parent = pt_parent * (1 - splitting.z)


if __name__ == "__main__":
    helpers.setup_logging()
    logger.info("Tree 1:")
    calculate_splittings(
        splittings=[
            Splitting(delta_R=0.40, z=0.1),
            Splitting(delta_R=0.35, z=0.2),
            Splitting(delta_R=0.30, z=0.5),
        ],
        input_jet_pt=60.,
    )

    logger.info("Tree 2:")
    calculate_splittings(
        splittings=[
            Splitting(delta_R=0.40, z=0.1),
            Splitting(delta_R=0.30, z=0.2),
            Splitting(delta_R=0.20, z=0.5),
        ],
        input_jet_pt=60.,
    )

    logger.info("Tree 3:")
    calculate_splittings(
        splittings=[
            Splitting(delta_R=0.40, z=0.2),
            Splitting(delta_R=0.35, z=0.1),
            Splitting(delta_R=0.30, z=0.5),
        ],
        input_jet_pt=60.,
    )

    # Selected splittings:
    # 1. DyG Kt
    # 2. SD 0.2
    # 3. SD 0.4, DyG kt z>0.2
    logger.info("Tree 4:")
    calculate_splittings(
        splittings=[
            #Splitting(delta_R=0.40, z=0.19),
            #Splitting(delta_R=0.38, z=0.175),
            Splitting(delta_R=0.40, z=0.175),
            Splitting(delta_R=0.30, z=0.2),
            #Splitting(delta_R=0.225, z=0.4),
            Splitting(delta_R=0.20, z=0.4),
            Splitting(delta_R=0.1, z=0.1),
        ],
        input_jet_pt=60.,
    )