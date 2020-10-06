""" Tasks related to jet finding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
from typing import Any, Mapping, Type, TypeVar

import awkward1 as ak
import numba as nb
import numpy as np
import particle
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



@functools.lru_cache()
def _pdg_id_to_mass(pdg_id: int) -> float:
    """ Convert PDG ID to mass.

    We cache the result to speed it up.

    Args:
        pdg_id: PDG ID.
    Returns:
        Mass in MeV.
    """
    return particle.Particle.from_pdgid(pdg_id).mass

@nb.njit
def determine_mass(events: ak.Array, builder: ak.ArrayBuilder, pdg_id_to_mass: Mapping[int, float]) -> ak.Array:
    for event in events:
        builder.begin_list()
        for particle in event:
            builder.append(pdg_id_to_mass[particle["particle_ID"]])
        builder.end_list()

    #return builder.snapshot()


def find_jets_arr(array: ak.Array) -> ak.Array:
    """ Find jets.

    """
    # Particle selection
    # Drop neutrinos.
    new_array = array[(np.abs(array["particle_ID"]) != 12) & (np.abs(array["particle_ID"]) != 14) & (np.abs(array["particle_ID"]) != 16)]
    # Determine masses
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(new_array["particle_ID"])))

    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float32,
    )
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(np.float32(pdg_id))
    #import IPython; IPython.embed()
    #pdg_id_to_mass = nb.typed.Dict({pdg_id: _pdg_id_to_mass(pdg_id) for pdg_id in all_particle_IDs})

    builder = ak.ArrayBuilder()
    determine_mass(events=new_array, builder=builder, pdg_id_to_mass=pdg_id_to_mass)
    #array["m"] = builder.snapshot()
    mass = builder.snapshot()

    new_array = LorentzVectorArray.from_ptetaphim(new_array["pt"], new_array["eta"], new_array["phi"], mass)

    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)

    # TEMP!! The arguments don't match otherwise!!
    temp_array = ak.zip(
        {
            "E": new_array["t"],
            "px": new_array["x"],
            "py": new_array["y"],
            "pz": new_array["z"],
        },
        #with_name="LorentzVector",
    )
    print(temp_array.type)
    # ENDTEMP

    #jets = fj.find_jets(events=array.layout.Content, settings=settings)
    #jets = ak.Array(fj.find_jets(events=temp_array.layout, settings=settings))

    import IPython; IPython.embed()

    jets = ak.Array(fj.find_jets_awkward_test(events=temp_array.layout))

    #jets = fj.find_jets(events=temp_array, settings=settings)
    import IPython; IPython.embed()


if __name__ == "__main__":
    input_arrays = ak.from_parquet("skim/events_per_chunk_1/JetscapeHadronListBin100_110_00.parquet")
    # We use some very different value to make it clear if something ever goes wrong.
    # NOTE: It's important to do this before constructing anything else. Otherwise it can
    #       mess up the awkward1 behaviors.
    fill_none_value = -9999
    input_arrays = ak.fill_none(input_arrays, fill_none_value)

    # Fully zip the arrays together.
    arrays = ak.zip(dict(zip(ak.fields(input_arrays), ak.unzip(input_arrays))), depth_limit=None)
    #arrays = ak.zip(
    #    {
    #        "particle_ID": input_arrays["particle_ID"],
    #        "E": input_arrays["E"],
    #        "px": input_arrays["px"],
    #        "py": input_arrays["py"],
    #        "pz": input_arrays["pz"],
    #    },
    #    depth_limit = None,
    #)
    import IPython; IPython.embed()

    find_jets_arr(array=arrays)
