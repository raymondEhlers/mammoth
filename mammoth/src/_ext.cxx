#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "mammoth/jetFinding.hpp"

namespace py = pybind11;
// Shorthand for literals
using namespace pybind11::literals;

/**
  * Convert numpy array of px, py, pz, E to a four vector tuple.
  *
  * This is kind of a dumb step, but it makes our lives simpler later. Namely, this means there
  * is a second conversion step to PseudoJets for fastjet, but I think this extra conversion is
  * worth the cost for a cleaner separation of interfaces. To be revised later if it's an issue.
  *
  * Note: The array is required to be c-style, which ensures that it works with other packages.
  *       For example, pandas caused a problem in some cases without that argument.
  *
  * @param[in] pxIn Numpy px array.
  * @param[in] pyIn Numpy py array.
  * @param[in] pzIn Numpy pz array.
  * @param[in] EIn Numpy E array.
  * @returns Column four vectors.
  */
template<typename T>
mammoth::FourVectorTuple<T> & numpyToColumnFourVector(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & EIn
)
{
  // Retrieve array and relevant information
  py::buffer_info infoPx = pxIn.request();
  auto px = pxIn.data();
  auto py = pyIn.data();
  auto pz = pzIn.data();
  auto E = EIn.data();
  // This defines our numpy array shape.
  int nParticles = infoPx.shape[0];

  // Convert the arrays
  std::vector<T> pxOut(nParticles), pyOut(nParticles), pzOut(nParticles), EOut(nParticles);
  for (size_t i = 0; i < nParticles; ++i) {
    /*std::cout << "i: " << i << " inputs: " << inputJets[i * nParams + 0] << " " << inputJets[i * nParams + 1]
      << " " << inputJets[i * nParams + 2] << " " <<  inputJets[i * nParams + 3] << "\n";*/
    pxOut.push_back(px[i]);
    pyOut.push_back(py[i]);
    pzOut.push_back(pz[i]);
    EOut.push_back(E[i]);
  }

  return std::make_tuple(pxOut, pyOut, pzOut, EOut);
}

/**
 * Create PseudoJet objects from a numpy array of px, py, pz, E. Axis 0 is the number of particles,
 * while axis 1 must be the 4 parameters.
 *
 * Note: The array is required to be c-style, which ensures that it works with other packages. For example,
 *       pandas caused a problem in some cases without that argument.
 *
 * @param[jets] Numpy input array.
 * @returns Vector of PseudoJets.
 */
//std::vector<fastjet::PseudoJet> constructPseudojetsFromNumpy(const py::array_t<double, py::array::c_style | py::array::forcecast> & jets)
//{
//  // Retrieve array and relevant information
//  py::buffer_info info = jets.request();
//  // I'm not sure which one of these is better.
//  //auto inputJets = static_cast<double *>(info.ptr);
//  auto inputJets = jets.data();
//  std::vector<fastjet::PseudoJet> outputJets;
//  // This defines our numpy array shape.
//  int nParticles = info.shape[0];
//  int nParams = info.shape[1];
//  //std::cout << "nParams: " << nParams << ", nParticles: " << nParticles << "\n";
//
//  // Validation.
//  if (nParams != 4) {
//    throw std::runtime_error("Number of params is not correct. Should be four per particle.");
//  }
//  // Convert the arrays
//  for (size_t i = 0; i < nParticles; ++i) {
//    /*std::cout << "i: " << i << " inputs: " << inputJets[i * nParams + 0] << " " << inputJets[i * nParams + 1]
//      << " " << inputJets[i * nParams + 2] << " " <<  inputJets[i * nParams + 3] << "\n";*/
//    outputJets.push_back(fastjet::PseudoJet(
//      inputJets[i * nParams + 0], inputJets[i * nParams + 1],
//      inputJets[i * nParams + 2], inputJets[i * nParams + 3]));
//  }
//
//  return outputJets;
//}

//std::vector<fastjet::PseudoJet> numpyToPseudoJet(
//  const py::array_t<double, py::array::c_style | py::array::forcecast> & pxIn,
//  const py::array_t<double, py::array::c_style | py::array::forcecast> & pyIn,
//  const py::array_t<double, py::array::c_style | py::array::forcecast> & pzIn,
//  const py::array_t<double, py::array::c_style | py::array::forcecast> & EIn
//  //const py::array_t<double, py::array::c_style | py::array::forcecast> & particleIndexIn
//)
//{
//  // Retrieve array and relevant information
//  py::buffer_info infoPx = pxIn.request();
//  auto px = pxIn.data();
//  // This defines our numpy array shape.
//  int nParticles = info.shape[0];
//  //int nParams = info.shape[1];
//  // py
//  auto py = pyIn.data();
//  auto pz = pzIn.data();
//  auto E = EIn.data();
//  //auto particleIndex = particleIndexIn.data();
//
//  std::vector<fastjet::PseudoJet> outputJets;
//
//  // Convert the arrays
//  for (size_t i = 0; i < nParticles; ++i) {
//    /*std::cout << "i: " << i << " inputs: " << inputJets[i * nParams + 0] << " " << inputJets[i * nParams + 1]
//      << " " << inputJets[i * nParams + 2] << " " <<  inputJets[i * nParams + 3] << "\n";*/
//    outputJets.emplace_back(fastjet::PseudoJet(px[i], py[i], pz[i], E[i]));
//    //outputJets.back().set_user_index(particleIndex[i]);
//    outputJets.back().set_user_index(i);
//  }
//  return outputJets;
//}

/**
  * Find jets with background subtraction.
  */
template <typename T>
std::tuple<mammoth::FourVectorTuple<T>, std::vector<std::vector<unsigned int>>, std::optional<std::tuple<mammoth::FourVectorTuple<T>, std::vector<unsigned int>>>> findJets(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  double jetR,
  std::tuple<double, double> etaRange = std::make_tuple(-0.9, 0.9)
)
{
    auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
    return mammoth::findJets(fourVectors, jetR, etaRange);
}

void testFunc() {
  std::cout << "Hi!\n";
}

PYBIND11_MODULE(_ext, m) {
  m.def("test_func", &testFunc, "Test function...");
  // Helper functions
  //m.def("dot_product", &dot_product, "jet_1"_a, "jet_2"_a, "Returns the 4-vector dot product of a and b");
  //m.def("have_same_momentum", &have_same_momentum, "jet_1"_a, "jet_2"_a, "Returns true if the momenta of the two input jets are identical");
  //m.def("sorted_by_pt", &sorted_by_pt, "jets"_a, "Return a vector of jets sorted into decreasing transverse momentum");
  //m.def("sorted_by_pz", &sorted_by_pz, "jets"_a, "Return a vector of jets sorted into increasing pz");
  //m.def("sorted_by_rapidity", &sorted_by_rapidity, "jets"_a, "Return a vector of jets sorted into increasing rapidity");
  //m.def("sorted_by_E", &sorted_by_E, "jets"_a, "Return a vector of jets sorted into decreasing energy");
}
