""" Tasks related to jet finding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any, Type, TypeVar

import awkward1 as ak
import numpy as np
import pyfastjet as fj


class LorentzVectorCommon:
    """ Basic Lorentz Vector class for conveinence.

    Assumes metric: (+ - - -)

    """
    t: Any
    x: Any
    y: Any
    z: Any

    @property
    def pt(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def eta(self):
        return np.arcsinh(self.z / self.pt)

    @property
    def phi(self):
        """

        Appears to be defined from [-pi, pi). PseudoJets are defined from [0, 2pi)
        """
        # NOTE: Could put it within [0, 2pi) with (take from fastjet::PseudoJet):
        #       if (_phi >= twopi) _phi -= twopi;
        #       if (_phi < 0)      _phi += twopi;
        return np.arctan2(self.y, self.x)


class LorentzVector(ak.Record, LorentzVectorCommon):  # type: ignore
    """ Basic Lorentz Vector class for conveinence.

    Assumes metric: (+ - - -)

    """
    t: float
    x: float
    y: float
    z: float
    ...


class LorentzVectorArray(ak.Array, LorentzVectorCommon):  # type: ignore
    t: ak.Array
    x: ak.Array
    y: ak.Array
    z: ak.Array

    @staticmethod
    def from_ptetaphim(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, m: np.ndarray) -> ak.Array:
        return ak.zip(
            {
                # magnitude of p = pt*cosh(eta)
                "t": np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2),
                "x": pt * np.cos(phi),
                "y": pt * np.sin(phi),
                "z": pt * np.sinh(eta),
            },
            with_name="LorentzVector",
        )


# Register behavior
ak.behavior["LorentzVector"] = LorentzVector
ak.behavior["*", "LorentzVector"] = LorentzVectorArray


def find_jets(array: ak.Array) -> ...:
    """ Find jets.

    """
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    for event in array:
        columns = ["px", "py", "pz", "E"]
        arr = ak.to_numpy(
            event[columns]
        )
        # Convert from recarray to standard array
        arr = arr.view(np.float64).reshape((len(event), len(columns)))

        cs = fj.ClusterSequence(
            pseudojets = arr,
            jet_definition = jet_defintion,
        )
        # Convert from pt, eta, phi, m -> standard LorentzVector
        jets = LorentzVectorArray.from_ptetaphim(
            **dict(zip(["pt", "eta", "phi", "m"], cs.to_numpy()))
        )
        print(jets.layout)
        print(ak.type(jets))
        sorted_by_pt = ak.argsort(jets.pt, ascending=True)
        jets = jets[sorted_by_pt]

        #jets = ak.zip(
        #    dict(zip(["x", "y", "z", "t"], cs.to_numpy())),
        #    with_name="LorentzVector",
        #)
        #jets = fj.sorted_by_pt(cs.inclusive_jets())
        #import IPython; IPython.embed()

        #print(jets.to_numpy())


def find_jets_arr(array: ak.Array) -> ak.Array:
    """ Find jets.

    """
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)

    #jets = fj.find_jets(events=array.layout.Content, settings=settings)
    jets = ak.Array(fj.find_jets(events=array.layout, settings=settings))

    import IPython; IPython.embed()


if __name__ == "__main__":
    input_arrays = ak.from_parquet("skim/jetscape.parquet")
    # We use some very different value to make it clear if something ever goes wrong.
    # NOTE: It's important to do this before constructing anything else. Otherwise it can
    #       mess up the awkward1 behaviors.
    fill_none_value = -9999
    input_arrays = ak.fill_none(input_arrays, fill_none_value)

    arrays = ak.zip(
        {
            "particle_ID": input_arrays["particle_ID"],
            "E": input_arrays["E"],
            "px": input_arrays["px"],
            "py": input_arrays["py"],
            "pz": input_arrays["pz"],
        },
        depth_limit = None,
    )
    #import IPython; IPython.embed()

    find_jets_arr(array=arrays)
