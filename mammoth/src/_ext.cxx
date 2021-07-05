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
  * @tparam T Input data type (usually float or double).
  * @param[in] pxIn Numpy px array.
  * @param[in] pyIn Numpy py array.
  * @param[in] pzIn Numpy pz array.
  * @param[in] EIn Numpy E array.
  * @returns Column four vectors.
  */
template<typename T>
mammoth::FourVectorTuple<T> numpyToColumnFourVector(
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
    // NOTE: Don't emplace back - the size is set above.
    pxOut[i] = px[i];
    pyOut[i] = py[i];
    pzOut[i] = pz[i];
    EOut[i] = E[i];
  }

  return {pxOut, pyOut, pzOut, EOut};
}

 /**
  * @brief Find jets with background subtraction.
  *
  * @tparam T Input data type (usually float or double).
  * @param pxIn px of input particles
  * @param pyIn py of input particles
  * @param pzIn pz of input particles
  * @param EIn energy of input particles
  * @param jetR jet resolution parameter
  * @param jetAlgorithm jet alogrithm
  * @param etaRange Eta range. Tuple of min and max
  * @param minJetPt Minimum jet pt.
  * @return mammoth::OutputWrapper<T> Output from jet finding.
  */
template <typename T>
mammoth::OutputWrapper<T> findJets(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  double jetR,
  std::string jetAlgorithm,
  std::tuple<double, double> etaRange,
  double minJetPt,
  bool backgroundSubtraction,
  std::optional<mammoth::ConstituentSubtractionSettings> constituentSubtraction
)
{
  auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
  return mammoth::findJets(fourVectors, jetR, jetAlgorithm, etaRange, minJetPt, backgroundSubtraction, constituentSubtraction);
}

PYBIND11_MODULE(_ext, m) {
  m.def("find_jets", &findJets<float>, "px"_a, "py"_a, "pz"_a, "E"_a, "jet_R"_a, "jet_algorithm"_a = "anti-kt", "eta_range"_a = std::make_tuple(-0.9, 0.9), "min_jet_pt"_a = 1., "background_subtraction"_a = false, "constituent_subtraction"_a = std::nullopt, "Jet finding function");
  m.def("find_jets", &findJets<double>, "px"_a, "py"_a, "pz"_a, "E"_a, "jet_R"_a, "jet_algorithm"_a = "anti-kt", "eta_range"_a = std::make_tuple(-0.9, 0.9), "min_jet_pt"_a = 1., "background_subtraction"_a = false, "constituent_subtraction"_a = std::nullopt, "Jet finding function");
  // Output wrapper. Just providing access to the fields.
  py::class_<mammoth::OutputWrapper<double>>(m, "OutputWrapper", "Output wrapper")
    .def_readonly("jets", &mammoth::OutputWrapper<double>::jets)
    .def_readonly("constituent_indices", &mammoth::OutputWrapper<double>::constituent_indices)
    .def_readonly("subtracted_info", &mammoth::OutputWrapper<double>::subtracted)
  ;
  // Wrapper for constituent subtraction settings
  py::class_<mammoth::ConstituentSubtractionSettings>(m, "ConstituentSubtractionSettings", "Constituent subtraction settings")
    .def(py::init<double, double>(), "r_max"_a = 0.25, "alpha"_a = 1)
    .def_readwrite("r_max", &mammoth::ConstituentSubtractionSettings::rMax)
    .def_readwrite("alpha", &mammoth::ConstituentSubtractionSettings::alpha)
  ;
}
