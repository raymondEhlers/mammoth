
from typing import Any, TypeVar

import awkward1 as ak
import numpy as np

_T = TypeVar("_T")

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
        sign = -1 if self.pz > 0 else 1
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


# Register behavior
ak.behavior["LorentzVector"] = LorentzVector
ak.behavior["*", "LorentzVector"] = LorentzVectorArray


