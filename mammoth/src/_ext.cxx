#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "mammoth/jetFinding.hpp"
#include "mammoth/aliceFastSim.hpp"

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
  unsigned int nParticles = infoPx.shape[0];

  // Convert the arrays
  std::vector<T> pxOut(nParticles), pyOut(nParticles), pzOut(nParticles), EOut(nParticles);
  for (std::size_t i = 0; i < nParticles; ++i) {
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
  * NOTE: The interface is awkward because we can't optionally pass the background estimator particles.
  *       Instead, we implicitly pass them optionally by reacting if they're empty by passing the input
  *       particles to the background estimator. It would be nicer if it was better, but the only person
  *       who has to actually this interface is me, so it's not the end of the world (it's hidden behind
  *       other functions for all uses).
  *
  * @tparam T Input data type (usually float or double).
  * @param pxIn px of input particles
  * @param pyIn py of input particles
  * @param pzIn pz of input particles
  * @param EIn energy of input particles
  * @param backgroundPxIn px of background estimator particles
  * @param backgroundPyIn py of background estimator particles
  * @param backgroundPzIn pz of background estimator particles
  * @param backgroundEIn energy of background estimator particles
  * @param jetR jet resolution parameter
  * @param jetAlgorithm jet algorithm
  * @param areaSettings Jet area calculation settings
  * @param etaRange Eta range. Tuple of min and max
  * @param minJetPt Minimum jet pt.
  * @param backgroundSubtraction If true, enable rho background subtraction
  * @param constituentSubtraction If provided, configure constituent subtraction according to given settings.
  * @return mammoth::OutputWrapper<T> Output from jet finding.
  */
template <typename T>
mammoth::OutputWrapper<T> findJets(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundEIn,
  double jetR,
  std::string jetAlgorithm,
  mammoth::AreaSettings areaSettings,
  std::tuple<double, double> etaRange,
  bool fiducialAcceptance,
  double minJetPt,
  bool backgroundSubtraction,
  std::optional<mammoth::ConstituentSubtractionSettings> constituentSubtraction
)
{
  auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
  // NOTE: These may be empty. If they are, the input four vectors are used for the background estimator
  auto backgroundFourVectors = numpyToColumnFourVector<T>(backgroundPxIn, backgroundPyIn, backgroundPzIn, backgroundEIn);
  return mammoth::findJets(fourVectors, jetR, jetAlgorithm, areaSettings, etaRange, fiducialAcceptance, minJetPt, backgroundFourVectors, backgroundSubtraction, constituentSubtraction);
}

template <typename T>
mammoth::JetSubstructure::JetSubstructureSplittings reclusterJet(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  double jetR,
  std::string jetAlgorithm,
  std::optional<mammoth::AreaSettings> areaSettings,
  std::tuple<double, double> etaRange,
  bool storeRecursiveSplittings
)
{
  auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
  return mammoth::jetReclustering(fourVectors, jetR, jetAlgorithm, areaSettings, etaRange, storeRecursiveSplittings);
}

 /**
  * @brief Redirect stdout for logging for jet finding functionality.
  *
  * It's a trivial wrapper so we can use call_guard, which makes things simpler but
  * can't pass arguments, so we need to set the defaults.
  */
class JetFindingLoggingStdout : public py::scoped_ostream_redirect {
  public:
    JetFindingLoggingStdout(): py::scoped_ostream_redirect(
        std::cout,                               // std::ostream&
        py::module_::import("mammoth.src.logging").attr("jet_finding_logger_stdout") // Python output
    ) {}
};

 /**
  * @brief Redirect stderr for logging for jet finding functionality.
  *
  * It's a trivial wrapper so we can use call_guard, which makes things simpler but
  * ca't pass arguments, so we need to set the defaults.
  */
class JetFindingLoggingStderr : public py::scoped_ostream_redirect {
  public:
    JetFindingLoggingStderr(): py::scoped_ostream_redirect(
        std::cerr,                               // std::ostream&
        py::module_::import("mammoth.src.logging").attr("jet_finding_logger_stderr") // Python output
    ) {}
};

/**
 * @brief Wrap the output wrapper with pybind11
 *
 * Based on the idea here: https://stackoverflow.com/a/47749076/12907985
 *
 * @tparam T Type to specialize for the output wrapper
 * @param m pybind11 module
 * @param typestr Name of the string, capitalized by convention.
 */
template<typename T>
void wrapOutputWrapper(py::module & m, const std::string & typestr)
{
  using Class = mammoth::OutputWrapper<T>;
  std::string pythonClassName = "OutputWrapper" + typestr;
  py::class_<Class>(m, pythonClassName.c_str(), "Output wrapper")
    .def_readonly("jets", &Class::jets)
    .def_readonly("constituent_indices", &Class::constituent_indices)
    .def_readonly("jets_area", &Class::jetsArea)
    .def_readonly("subtracted_info", &Class::subtracted)
  ;
}

PYBIND11_MODULE(_ext, m) {
  // Output wrapper. Just providing access to the fields.
  wrapOutputWrapper<double>(m, "Double");
  wrapOutputWrapper<float>(m, "Float");
  // Wrapper for area settings
  py::class_<mammoth::AreaSettings>(m, "AreaSettings", "Settings related to jet finding area")
    .def(py::init<std::string, double>(), "area_type"_a = "active_area", "ghost_area"_a = 0.005)
    .def_readwrite("area_type", &mammoth::AreaSettings::areaType)
    .def_readwrite("ghost_area", &mammoth::AreaSettings::ghostArea)
    .def("__repr__", [](const mammoth::AreaSettings &s) {
      return "<AreaSettings area_type='" + s.areaType + "', ghost_area=" + std::to_string(s.ghostArea) + ">";
    })
  ;
  // Wrapper for constituent subtraction settings
  py::class_<mammoth::ConstituentSubtractionSettings>(m, "ConstituentSubtractionSettings", "Constituent subtraction settings")
    .def(py::init<double, double>(), "r_max"_a = 0.25, "alpha"_a = 0)
    .def_readwrite("r_max", &mammoth::ConstituentSubtractionSettings::rMax)
    .def_readwrite("alpha", &mammoth::ConstituentSubtractionSettings::alpha)
    .def("__repr__", [](const mammoth::ConstituentSubtractionSettings &s) {
      return "<ConstituentSubtractionSettings r_max=" + std::to_string(s.rMax) + ", alpha=" + std::to_string(s.alpha) + ">";
    })
  ;
  m.def("find_jets", &findJets<float>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                       "background_px"_a, "background_py"_a, "background_pz"_a, "background_E"_a,
                                       "jet_R"_a, "jet_algorithm"_a, "area_settings"_a,
                                       "eta_range"_a = std::make_tuple(-0.9, 0.9),
                                       "fiducial_acceptance"_a = true,
                                       "min_jet_pt"_a = 1.,
                                       "background_subtraction"_a = false, "constituent_subtraction"_a = std::nullopt,
                                       "Jet finding function", py::call_guard<JetFindingLoggingStdout, JetFindingLoggingStderr>());
  m.def("find_jets", &findJets<double>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                        "background_px"_a, "background_py"_a, "background_pz"_a, "background_E"_a,
                                        "jet_R"_a, "jet_algorithm"_a, "area_settings"_a,
                                        "eta_range"_a = std::make_tuple(-0.9, 0.9),
                                        "fiducial_acceptance"_a = true,
                                        "min_jet_pt"_a = 1.,
                                        "background_subtraction"_a = false, "constituent_subtraction"_a = std::nullopt,
                                        "Jet finding function", py::call_guard<JetFindingLoggingStdout, JetFindingLoggingStderr>());

  // Wrapper for reclustered jet outputs
  py::class_<mammoth::JetSubstructure::ColumnarSplittings>(m, "ColumnarSplittings", "Columnar splittings output")
    .def_readonly("kt", &mammoth::JetSubstructure::ColumnarSplittings::kt)
    .def_readonly("delta_R", &mammoth::JetSubstructure::ColumnarSplittings::deltaR)
    .def_readonly("z", &mammoth::JetSubstructure::ColumnarSplittings::z)
    .def_readonly("parent_index", &mammoth::JetSubstructure::ColumnarSplittings::parentIndex)
  ;
  py::class_<mammoth::JetSubstructure::ColumnarSubjets>(m, "ColumnarSubjest", "Columnar splittings output")
    .def_readonly("splitting_node_index", &mammoth::JetSubstructure::ColumnarSubjets::splittingNodeIndex)
    .def_readonly("part_of_iterative_splitting", &mammoth::JetSubstructure::ColumnarSubjets::partOfIterativeSplitting)
    .def_readonly("constituent_indices", &mammoth::JetSubstructure::ColumnarSubjets::constituentIndices)
  ;

  py::class_<mammoth::JetSubstructure::JetSubstructureSplittings>(m, "JetSubstructureSplittings", "Jet substructure splittings")
    //.def("splittings", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSplittings {
    //  auto && [kt, deltaR, z, parentIndex ] = substructure.GetSplittings().GetSplittings();
    //  return mammoth::JetSubstructure::ColumnarSplittings{kt, deltaR, z, parentIndex};
    //})
    //.def("subjets", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSubjets {
    //  auto && [splittingNodeIndex, partOfIterativeSplitting, constituentIndices] = substructure.GetSubjets().GetSubjets();
    //  return mammoth::JetSubstructure::ColumnarSubjets{splittingNodeIndex, partOfIterativeSplitting, constituentIndices};
    //})
    .def("splittings", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSplittings {
      return substructure.GetSplittings().GetSplittings();
    })
    .def("subjets", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSubjets {
      return substructure.GetSubjets().GetSubjets();
    })
  ;

  // Jet reclustering
  m.def("recluster_jet", &reclusterJet<float>, "px"_a, "py"_a, "pz"_a, "E"_a, "jet_R"_a = 1.0, "jet_algorithm"_a = "CA", "area_settings"_a = std::nullopt, "eta_range"_a = std::make_tuple(-1, 1), "store_recursive_splittings"_a = true, "Recluster the given jet", py::call_guard<JetFindingLoggingStdout, JetFindingLoggingStderr>());
  m.def("recluster_jet", &reclusterJet<double>, "px"_a, "py"_a, "pz"_a, "E"_a, "jet_R"_a = 1.0, "jet_algorithm"_a = "CA", "area_settings"_a = std::nullopt, "eta_range"_a = std::make_tuple(-1, 1), "store_recursive_splittings"_a = true, "Recluster the given jet", py::call_guard<JetFindingLoggingStdout, JetFindingLoggingStderr>());

  // ALICE
  // Fast sim
  py::enum_<alice::fastsim::Period_t>(m, "TrackingEfficiencyPeriod",  py::arithmetic(), "Tracking efficiency periods")
    .value("disabled", alice::fastsim::Period_t::kDisabled, "Disabled. Always return 1")
    .value("LHC11h", alice::fastsim::Period_t::kLHC11h, "Run1 PbPb - LHC11h")
    .value("LHC15o", alice::fastsim::Period_t::kLHC15o, "Run2 PbPb - LHC15o")
    .value("LHC18qr", alice::fastsim::Period_t::kLHC18qr, "Run2 PbPb - LHC18{q,r}")
    .value("LHC11a", alice::fastsim::Period_t::kLHC11a, "Run1 pp - LHC11a (2.76 TeV)")
    .value("pA", alice::fastsim::Period_t::kpA, "Generic pA")
    .value("pp", alice::fastsim::Period_t::kpp, "Generic pp")
    .export_values();
  py::enum_<alice::fastsim::EventActivity_t>(m, "TrackingEfficiencyEventActivity", py::arithmetic(), "Event activity for tracking efficiency")
    .value("inclusive", alice::fastsim::EventActivity_t::kInclusive, "Inclusive event activity, for say, pp, or MB PbPb.")
    .value("central_00_10", alice::fastsim::EventActivity_t::k0010, "0-10% central event activity")
    .value("mid_central_10_30", alice::fastsim::EventActivity_t::k1030, "10-30% mid-central event activity")
    .value("semi_central_30_50", alice::fastsim::EventActivity_t::k3050, "30-50% semi-central event activity")
    .value("peripheral_50_90", alice::fastsim::EventActivity_t::k5090, "50-90% peripheral event activity")
    .value("invalid", alice::fastsim::EventActivity_t::kInvalid, "Invalid event activity")
    .export_values();

  m.def("find_event_activity", &alice::fastsim::findEventActivity, "value"_a,
        "Utility to convert a numerical event activity value to an event activity enumeration value for calling the tracking efficiency.");
  m.def("fast_sim_tracking_efficiency", py::vectorize(alice::fastsim::trackingEfficiencyByPeriod),
        "track_pt"_a, "track_eta"_a, "event_activity"_a, "period"_a,
        "Fast sim via tracking efficiency parametrization", py::call_guard<JetFindingLoggingStdout, JetFindingLoggingStderr>());
}
