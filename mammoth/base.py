
""" Base analysis functionality

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import operator
from typing import Any, Mapping, Optional, Sequence, TypeVar

import awkward as ak
import numba as nb
import numpy as np
import particle

_T = TypeVar("_T")

"""Lorentz vector functionality will only be maintained until scikit-hep/vector is mature.
Then we will switch over to that, which will be far preferable.
"""

class LorentzVectorCommon:
    """ Basic Lorentz Vector class for convenience.

    Momentum oriented, but of course still works for (x, y, z, t)

    Assumes metric: (-, -, -, +)

    """
    px: Any
    py: Any
    pz: Any
    E: Any

    @property
    def pt(self):
        return np.sqrt(self.pt2)

    @property
    def pt2(self):
        return self.px ** 2 + self.py ** 2

    @property
    def eta(self):
        return np.arcsinh(self.pz / self.pt)

    @property
    def phi(self):
        """

        Appears to be defined from [-pi, pi). PseudoJets are defined from [0, 2pi)
        """
        # NOTE: Could put it within [0, 2pi) with (take from fastjet::PseudoJet):
        #       if (_phi >= twopi) _phi -= twopi;
        #       if (_phi < 0)      _phi += twopi;
        return np.arctan2(self.py, self.px)

    @property
    def mass_squared(self):
        """ Mass squared.

        """
        return ((self.E + self.pz) * (self.E - self.pz)) - self.pt2

    @property
    def rapidity(self):
        """ Rapidity.

        Based on the fastjet::PseudoJet calculation.
        """
        # Get the rapidity in a way that's modestly insensitive to roundoff
        # error when things pz,E are large (actually the best we can do without
        # explicit knowledge of mass)
        # TODO: Fix this line so we can run it safely...
        #effective_m2 = ak.max(np.array(0.0), self.mass_squared)
        effective_m2 = self.mass_squared
        # p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
        sign = ak.where(self.pz > 0, -1, 1)
        return sign * 0.5 * np.log(
            (self.pt2 + effective_m2)/((self.E + np.abs(self.pz)) ** 2)
        )

    def delta_eta(self: _T, other: _T):
        return self.eta - other.eta

    def delta_phi(self: _T, other: _T):
        # TODO: Make this a more comprehensive calculation...
        return self.phi - other.phi

    def delta_R(self: _T, other: _T):
        # TODO: Fully implement...
        ...


class LorentzVector(ak.Record, LorentzVectorCommon):  # type: ignore
    """ Basic Lorentz Vector class for convenience.

    Assumes metric: (+ - - -)

    """
    px: float
    py: float
    pz: float
    E: float


class LorentzVectorArray(ak.Array, LorentzVectorCommon):  # type: ignore
    """ Vector of Lorentz vectors.

    """
    px: ak.Array
    py: ak.Array
    pz: ak.Array
    E: ak.Array

    @staticmethod
    def from_ptetaphim(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, m: np.ndarray) -> ak.Array:
        return ak.zip(
            {
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pt * np.sinh(eta),
                # magnitude of p = pt*cosh(eta)
                "E": np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2),
            },
            with_name="LorentzVector",
        )

    @staticmethod
    def from_awkward_ptetaphim(arrays: ak.Array) -> ak.Array:
        """ Create LorentzVector while maintaining other fields of interest.

        """
        # TODO: Make this more general, such it copies all of the rest of the fields.
        return ak.zip(
            {
                "px": arrays["pt"] * np.cos(arrays["phi"]),
                "py": arrays["pt"] * np.sin(arrays["phi"]),
                "pz": arrays["pt"] * np.sinh(arrays["eta"]),
                # magnitude of p = pt*cosh(eta)
                "E": np.sqrt((arrays["pt"] * np.cosh(arrays["eta"])) ** 2 + arrays["m"] ** 2),
                "status": arrays["status"],
                "particle_ID": arrays["particle_ID"],
            },
            with_name="LorentzVector",
        )

    @staticmethod
    def from_ptetaphie(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, E: np.ndarray) -> ak.Array:
        return ak.zip(
            {
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pt * np.sinh(eta),
                "E": E,
            },
            with_name="LorentzVector",
        )

    @staticmethod
    def from_awkward_ptetaphie(arrays: ak.Array) -> ak.Array:
        """ Create LorentzVector while maintaining other fields of interest.

        """
        # TODO: Make this more general, such it copies all of the rest of the fields.
        return ak.zip(
            {
                "px": arrays["pt"] * np.cos(arrays["phi"]),
                "py": arrays["pt"] * np.sin(arrays["phi"]),
                "pz": arrays["pt"] * np.sinh(arrays["eta"]),
                # magnitude of p = pt*cosh(eta)
                "E": arrays["E"],
                "status": arrays["status"],
                "particle_ID": arrays["particle_ID"],
            },
            with_name="LorentzVector",
        )

# Register behavior
ak.behavior["LorentzVector"] = LorentzVector
ak.behavior["*", "LorentzVector"] = LorentzVectorArray


@functools.lru_cache()
def _pdg_id_to_mass(pdg_id: int) -> float:
    """ Convert PDG ID to mass.

    We cache the result to speed it up.

    Args:
        pdg_id: PDG ID.
    Returns:
        Mass in GeV.
    """
    m = particle.Particle.from_pdgid(pdg_id).mass
    # Apparently neutrino mass returns None...
    if m is None:
        m = 0
    return m / 1000


@nb.njit
def _determine_mass_from_PDG(events: ak.Array, builder: ak.ArrayBuilder, pdg_id_to_mass: Mapping[int, float]) -> ak.Array:
    """ Determine the mass for each particle based on the PID.

    These masses are in the same shape as the given array so they can be stored alongside them.

    Args:
        events: Particles organized in event structure.
        builder: Array builder.
        pdg_id_to_mass: Map from PDG ID to mass. This mapping must be in a form that numba will accept!

    Returns:
        Array of masses matching the given array.
    """
    for event in events:
        builder.begin_list()
        # The conditions for where we can iterate over the entire event vs only one column aren't yet
        # clear to me. It has something to do with record arrays...
        for particle_ID in event["particle_ID"]:
            builder.append(pdg_id_to_mass[particle_ID])
        builder.end_list()

    return builder


def determine_masses_from_events(events: ak.Array) -> ak.Array:
    """ Determine particle masses based on PDG codes.

    Args:
        events: Events containing particles.

    Returns:
        Masses calculated according to the PDG code, in the same shape as events.
    """
    # Add the masses based on the PDG code.
    # First, determine all possible PDG codes, and then retrieve their masses for a lookup table.
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(events["particle_ID"])))
    # NOTE: We need to use this special typed dict with numba.
    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float64,
    )
    # As far as I can tell, we can't fill the dict directly on initialization, so we have to loop over entires.
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(pdg_id)
    # It's important that we snapshot it!
    return _determine_mass_from_PDG(events=events, builder=ak.ArrayBuilder(), pdg_id_to_mass=pdg_id_to_mass).snapshot()


def build_PID_selection_mask(particles: ak.Array, absolute_pids: Optional[Sequence[int]] = None, single_pids: Optional[Sequence[int]] = None):
    """Build selection for particles based on particle ID.

    Args:
        particles: Available particles, over which we will build the mask.
        absolute_pids: PIDs which are selected using their absolute values.
        single_pids: PIDs which are selected only for the given values.

    Returns:
        Mask selecting the requested PIDs.
    """
    # Validation
    if absolute_pids is None:
        absolute_pids = []
    if single_pids is None:
        single_pids = []

    pid_cuts = [
        np.abs(particles["particle_ID"]) == pid for pid in absolute_pids
    ]
    pid_cuts.extend([
        particles["particle_ID"] == pid for pid in single_pids
    ])
    return functools.reduce(operator.and_, pid_cuts)


