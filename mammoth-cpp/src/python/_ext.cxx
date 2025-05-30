#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "mammoth/jetFinding.hpp"
#include "mammoth/jetFindingTools.hpp"
#include "mammoth/aliceFastSim.hpp"
#include "mammoth/analysisUtils.hpp"

namespace py = pybind11;
// Shorthand for literals
using namespace pybind11::literals;

// First up, some convenient constants
constexpr double DEFAULT_RAPIDITY_MAX = 1.0;

namespace mammoth {
namespace python {

/**
 * Minimally modified from the pybind11 class scoped_ostream_redirect.
 *
 * Modified to not rely on python objects, allowing the use of this redirect while releasing the GIL.
 * The easiest approach here is to pass a stringstream for the newStream.
 */
class scoped_ostream_redirect_with_sstream {
  protected:
      std::streambuf *old;
      std::ostream &costream;
      std::ostream &newStream;

  public:
      explicit scoped_ostream_redirect_with_sstream(std::ostream &costream,
                                       std::iostream &newStream)
          : costream(costream), newStream(newStream) {
          old = costream.rdbuf(newStream.rdbuf());
      }

      ~scoped_ostream_redirect_with_sstream() { costream.rdbuf(old); }

      scoped_ostream_redirect_with_sstream(const scoped_ostream_redirect_with_sstream &) = delete;
      scoped_ostream_redirect_with_sstream(scoped_ostream_redirect_with_sstream &&other) = default;
      scoped_ostream_redirect_with_sstream &operator=(const scoped_ostream_redirect_with_sstream &) = delete;
      scoped_ostream_redirect_with_sstream &operator=(scoped_ostream_redirect_with_sstream &&) = delete;
};

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
        py::module_::import("mammoth_cpp.logging").attr("jet_finding_logger_stdout") // Python output
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
        py::module_::import("mammoth_cpp.logging").attr("jet_finding_logger_stderr") // Python output
    ) {}
};

} /* namespace python */
} /* namespace mammoth */

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
  * @param[in] pxIn numpy px array.
  * @param[in] pyIn numpy py array.
  * @param[in] pzIn numpy pz array.
  * @param[in] EIn numpy E array.
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
  * Convert numpy array of user indices to vector.
  *
  * This is an addition to the base FourVectorTuple conversion.
  * This is kind of a dumb step, but it makes our lives simpler later. Namely, this means there
  * is a second conversion step to PseudoJets for fastjet, but I think this extra conversion is
  * worth the cost for a cleaner separation of interfaces. To be revised later if it's an issue.
  *
  * NOTE: The array is required to be c-style, which ensures that it works with other packages.
  *       For example, pandas caused a problem in some cases without that argument.
  *
  * NOTE: With some regularity, this array is empty.
  *
  * @tparam T Input data type (usually int).
  * @param[in] pxIn numpy user index array.
  * @returns User index vector.
  */
template<typename T>
std::vector<T> numpyToUserIndexVector(
  const py::array_t<int, py::array::c_style | py::array::forcecast> & userIndexIn
)
{
  // Retrieve array and relevant information
  py::buffer_info infoUserIndex = userIndexIn.request();
  auto userIndex = userIndexIn.data();
  // This defines our numpy array shape.
  unsigned int nParticles = infoUserIndex.shape[0];

  // Convert the arrays
  std::vector<T> userIndexOut(nParticles);
  for (std::size_t i = 0; i < nParticles; ++i) {
    // NOTE: Don't emplace back - the size is set above.
    userIndexOut[i] = userIndex[i];
  }

  return userIndexOut;
}


 /**
  * @brief Find jets with background subtraction.
  *
  * NOTE: The interface is a bit awkward because we can't optionally pass the background estimator particles.
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
  * @param jetFindingSettings Main jet finding settings
  * @param backgroundPxIn px of background estimator particles
  * @param backgroundPyIn py of background estimator particles
  * @param backgroundPzIn pz of background estimator particles
  * @param backgroundEIn energy of background estimator particles
  * @param backgroundSubtraction Background subtraction settings (including estimator and subtractor settings)
  * @param userIndexIn user provided user index of input particles. Optional.
  * @return mammoth::OutputWrapper<T> Output from jet finding.
  */
template <typename T>
mammoth::OutputWrapper<T> findJets(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  const mammoth::JetFindingSettings & jetFindingSettings,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundPzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & backgroundEIn,
  const mammoth::BackgroundSubtraction & backgroundSubtraction,
  const std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> userIndexIn,
  const bool releaseGil
)
{
  // Preprocessing
  // NOTE: Any cout/cerr in these preprocessing functions won't be forwarded to our usual logging.
  //       We do this so we can release the GIL later. If this becomes an issue, we just need to
  //       define the return values at the function scope, and then scope the logging redirects
  //       (as is done below when running the actual jet finding).
  //       By default, this is fine, since we usually aren't verbose here - it's just copying

  // Convert from python -> cpp types
  auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
  // NOTE: These may be empty. If they are, the user index is generated automatically with the index of the array.
  //       We have to be a bit careful here because we pass a nullptr by default, which will break if passed naively.
  std::vector<int> userIndex;
  if (userIndexIn.has_value()) {
    userIndex = numpyToUserIndexVector<int>(userIndexIn.value());
  }
  // NOTE: These may be empty. If they are, the input four vectors are used for the background estimator
  auto backgroundFourVectors = numpyToColumnFourVector<T>(backgroundPxIn, backgroundPyIn, backgroundPzIn, backgroundEIn);

  if (releaseGil) {
    mammoth::OutputWrapper<T> result;
    std::stringstream ssCout;
    std::stringstream ssCerr;
    {
      // Setup logging to stringstream types. We'll log them after the function is executed.
      // NOTE: We put this in a scope to ensure that it releases this redirect after we reacquire
      //       the gil, allowing us to switch back to the python logging
      mammoth::python::scoped_ostream_redirect_with_sstream outputCout(std::cout, ssCout);
      mammoth::python::scoped_ostream_redirect_with_sstream outputCerr(std::cerr, ssCerr);

      // Now that we're done with grabbing information from python, we can release the GIL and
      // actually run the calculation
      py::gil_scoped_release release;
      result = mammoth::findJets(fourVectors, userIndex, jetFindingSettings, backgroundFourVectors, backgroundSubtraction);
      // Once finished, reacquire the GIL to be able to access python objects
      // NOTE: As of April 2023, I think this may be redundant, but I don't think it will hurt anything.
      py::gil_scoped_acquire acquire;
    }
    {
      // Now that we've gotten the GIL back, we can log as normal
      mammoth::python::JetFindingLoggingStdout outputCoutPythonLogging;
      mammoth::python::JetFindingLoggingStderr outputCerrPythonLogging;
      // NOTE: We only want to log if there's anything to actually log
      // NOTE: Obviously it will be out of order. However, for now, it seems better to have the severity separation
      //       rather than focusing on the order. If this changes, we could always pass the same stringstream to the
      //       redirect. In that case, we need to decide which string to redirect to, because there's only one ss and
      //       we won't be able to differentiate the source of the message anymore.
      std::string logCout = ssCout.str();
      if (logCout.size() > 0) {
        std::cout << ssCout.str() << "\n";
      }
      std::string logCerr= ssCerr.str();
      if (logCerr.size() > 0) {
        std::cerr << ssCerr.str() << "\n";
      }
    }
    // And we're all done
    return result;
  }

  // If not releasing the GIL, then just run the calculation as usual with the standard logging
  mammoth::python::JetFindingLoggingStdout outputCoutPythonLogging;
  mammoth::python::JetFindingLoggingStderr outputCerrPythonLogging;
  return mammoth::findJets(fourVectors, userIndex, jetFindingSettings, backgroundFourVectors, backgroundSubtraction);
}

/**
 * @brief Jet reclustering
 *
 * @tparam T Input data type (usually float or double).
 * @param pxIn px of input particles
 * @param pyIn py of input particles
 * @param pzIn pz of input particles
 * @param EIn energy of input particles
 * @param jetFindingSettings Main jet finding settings
 * @param storeRecursiveSplittings If True, store recursive splittings (as opposed to iterative).
 * @return mammoth::JetSubstructure::JetSubstructureSplittings Data structure containing the requested splittings=
 *                                                            (originally from my AliPhysics task)
 */
template <typename T>
mammoth::JetSubstructure::JetSubstructureSplittings reclusterJet(
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<T, py::array::c_style | py::array::forcecast> & EIn,
  const mammoth::JetFindingSettings & jetFindingSettings,
  const std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> userIndexIn,
  const bool storeRecursiveSplittings,
  const bool releaseGil
)
{
  // Preprocessing
  // NOTE: Any cout/cerr in these preprocessing functions won't be forwarded to our usual logging.
  //       We do this so we can release the GIL later. If this becomes an issue, we just need to
  //       define the return values at the function scope, and then scope the logging redirects
  //       (as is done below when running the actual jet finding).
  //       By default, this is fine, since we usually aren't verbose here - it's just copying

  // Convert from python -> cpp types
  auto fourVectors = numpyToColumnFourVector<T>(pxIn, pyIn, pzIn, EIn);
  // NOTE: These may be empty. If they are, the user index is generated automatically with the index of the array.
  //       We have to be a bit careful here because we pass a nullptr by default, which will break if passed naively.
  std::vector<int> userIndex;
  if (userIndexIn.has_value()) {
    userIndex = numpyToUserIndexVector<int>(userIndexIn.value());
  }

  if (releaseGil) {
    mammoth::JetSubstructure::JetSubstructureSplittings result;
    std::stringstream ssCout;
    std::stringstream ssCerr;
    {
      // Setup logging to stringstream types. We'll log them after the function is executed.
      // NOTE: We put this in a scope to ensure that it releases this redirect after we reacquire
      //       the gil, allowing us to switch back to the python logging
      mammoth::python::scoped_ostream_redirect_with_sstream outputCout(std::cout, ssCout);
      mammoth::python::scoped_ostream_redirect_with_sstream outputCerr(std::cerr, ssCerr);

      // Now that we're done with grabbing information from python, we can release the GIL and
      // actually run the calculation
      py::gil_scoped_release release;
      result = mammoth::jetReclustering(fourVectors, userIndex, jetFindingSettings, storeRecursiveSplittings);
      // Once finished, reacquire the GIL to be able to access python objects
      // NOTE: As of April 2023, I think this may be redundant, but I don't think it will hurt anything.
      py::gil_scoped_acquire acquire;
    }
    {
      // Now that we've gotten the GIL back, we can log as normal
      mammoth::python::JetFindingLoggingStdout outputCoutPythonLogging;
      mammoth::python::JetFindingLoggingStderr outputCerrPythonLogging;
      // NOTE: We only want to log if there's anything to actually log
      // NOTE: Obviously it will be out of order. However, for now, it seems better to have the severity separation
      //       rather than focusing on the order. If this changes, we could always pass the same stringstream to the
      //       redirect. In that case, we need to decide which string to redirect to, because there's only one ss and
      //       we won't be able to differentiate the source of the message anymore.
      std::string logCout = ssCout.str();
      if (logCout.size() > 0) {
        std::cout << ssCout.str() << "\n";
      }
      std::string logCerr= ssCerr.str();
      if (logCerr.size() > 0) {
        std::cerr << ssCerr.str() << "\n";
      }
    }
    // And we're all done
    return result;
  }

  // If not releasing the GIL, then just run the calculation as usual with the standard logging
  mammoth::python::JetFindingLoggingStdout outputCoutPythonLogging;
  mammoth::python::JetFindingLoggingStderr outputCerrPythonLogging;
  return mammoth::jetReclustering(fourVectors, userIndex, jetFindingSettings, storeRecursiveSplittings);
}

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
    .def_readonly("constituents_user_index", &Class::constituents_user_index)
    .def_readonly("jets_area", &Class::jetsArea)
    .def_readonly("rho_value", &Class::rho)
    .def_readonly("subtracted_info", &Class::subtracted)
  ;
}

PYBIND11_MODULE(_ext, m) {
  // Constants
  m.attr("DEFAULT_RAPIDITY_MAX") = py::float_(DEFAULT_RAPIDITY_MAX);
  // Helper
  py::add_ostream_redirect(m, "cpp_redirect_stream");

  // Output wrapper. Just providing access to the fields.
  wrapOutputWrapper<double>(m, "Double");
  wrapOutputWrapper<float>(m, "Float");
  // Area settings
  py::class_<mammoth::AreaSettings, std::shared_ptr<mammoth::AreaSettings>>(m, "AreaSettings", "Settings related to jet finding area")
    .def(py::init<std::string, double, double, int, double, double, double, std::vector<int>>(),
         "area_type"_a = "active_area",
         "ghost_area"_a = 0.005,
         "rapidity_max"_a = DEFAULT_RAPIDITY_MAX,
         "repeat_N_ghosts"_a = 1,
         "grid_scatter"_a = 1.0,
         "kt_scatter"_a = 0.1,
         "kt_mean"_a = 1e-100,
         "random_seed"_a = std::vector<int>{}
      )
    .def_readwrite("area_type", &mammoth::AreaSettings::areaTypeName)
    .def_readwrite("ghost_area", &mammoth::AreaSettings::ghostArea)
    .def_readwrite("rapidity_max", &mammoth::AreaSettings::rapidityMax)
    .def("__repr__", [](const mammoth::AreaSettings &s) {
      return s.to_string();
    })
  ;
  // Base recombiner. Just to make pybind11 aware of it
  py::class_<mammoth::Recombiner, std::shared_ptr<mammoth::Recombiner>>(m, "Recombiner", "Base recombiner");
  // Negative energy recombiner
  py::class_<mammoth::NegativeEnergyRecombiner, mammoth::Recombiner, std::shared_ptr<mammoth::NegativeEnergyRecombiner>>(m, "NegativeEnergyRecombiner", "Negative energy recombiner")
    .def(
      py::init<const int >(),
        "identifier_index"_a
      )
    .def_readonly("identifier_index", &mammoth::NegativeEnergyRecombiner::identifierIndex)
    .def("__repr__", [](const mammoth::NegativeEnergyRecombiner &s) {
      return s.to_string();
    })
  ;
  // Jet finding settings
  py::class_<mammoth::JetFindingSettings, std::shared_ptr<mammoth::JetFindingSettings>>(m, "JetFindingSettings", "Main settings related to jet finding")
    .def(
      py::init<
        double, std::string, std::tuple<double, double>, std::tuple<double, double>, std::string, std::string,
        const std::optional<const mammoth::AreaSettings>,
        const std::shared_ptr<const mammoth::Recombiner>,
        const std::optional<double>
      >(),
        "R"_a,
        "algorithm"_a,
        "pt_range"_a,
        "eta_range"_a,
        "recombination_scheme"_a = "E_scheme",
        "strategy"_a = "Best",
        "area_settings"_a = std::nullopt,
        "recombiner"_a = nullptr,
        "additional_algorithm_parameter"_a = std::nullopt
      )
    .def_readwrite("R", &mammoth::JetFindingSettings::R)
    .def_readonly("area_settings", &mammoth::JetFindingSettings::areaSettings)
    .def_readonly("recombiner", &mammoth::JetFindingSettings::recombiner)
    .def("__repr__", [](const mammoth::JetFindingSettings &s) {
      return s.to_string();
    })
  ;
  // Base background estimator. Just to make pybind11 aware of it
  py::class_<mammoth::BackgroundEstimator, std::shared_ptr<mammoth::BackgroundEstimator>>(m, "BackgroundEstimator", "Base background estimator");
  // Jet median background estimator
  py::class_<mammoth::JetMedianBackgroundEstimator, mammoth::BackgroundEstimator, std::shared_ptr<mammoth::JetMedianBackgroundEstimator>>(m, "JetMedianBackgroundEstimator", "Background estimator based on jet median")
    .def(
      py::init<
        mammoth::JetFindingSettings, bool, bool, int, double
      >(),
        "jet_finding_settings"_a,
        "compute_rho_m"_a = true,
        "use_area_four_vector"_a = true,
        "exclude_n_hardest_jets"_a = 2,
        "constituent_pt_max"_a = 100.0
      )
    .def_readwrite("compute_rho_m", &mammoth::JetMedianBackgroundEstimator::computeRhoM)
    .def_readwrite("use_area_four_vector", &mammoth::JetMedianBackgroundEstimator::useAreaFourVector)
    .def_readwrite("exclude_n_hardest_jets", &mammoth::JetMedianBackgroundEstimator::excludeNHardestJets)
    .def_readwrite("constituent_pt_max", &mammoth::JetMedianBackgroundEstimator::constituentPtMax)
    .def("__repr__", [](const mammoth::JetMedianBackgroundEstimator &s) {
      return s.to_string();
    })
  ;
  // Grid median background estimator
  py::class_<mammoth::GridMedianBackgroundEstimator, mammoth::BackgroundEstimator, std::shared_ptr<mammoth::GridMedianBackgroundEstimator>>(m, "GridMedianBackgroundEstimator", "Background estimator based on a grid")
    .def(
      py::init<double, double>(),
        "rapidity_max"_a = DEFAULT_RAPIDITY_MAX,
        "grid_spacing"_a = 1.0
      )
    .def_readwrite("rapidity_max", &mammoth::GridMedianBackgroundEstimator::rapidityMax)
    .def_readwrite("grid_spacing", &mammoth::GridMedianBackgroundEstimator::gridSpacing)
    .def("__repr__", [](const mammoth::GridMedianBackgroundEstimator &s) {
      return s.to_string();
    })
  ;

  // Background subtraction type
  py::enum_<mammoth::BackgroundSubtraction_t>(m, "BackgroundSubtractionType",  py::arithmetic(), "Background subtraction type")
    .value("disabled", mammoth::BackgroundSubtraction_t::disabled, "Subtraction disabled")
    .value("rho", mammoth::BackgroundSubtraction_t::rho, "Rho subtraction")
    .value("event_wise_constituent_subtraction", mammoth::BackgroundSubtraction_t::eventWiseCS, "Event-wise constituent subtraction")
    .value("jet_wise_constituent_subtraction", mammoth::BackgroundSubtraction_t::jetWiseCS, "Jet-wise constituent subtraction");

  // Base background subtractor. Just to make pybind11 aware of it
  py::class_<mammoth::BackgroundSubtractor, std::shared_ptr<mammoth::BackgroundSubtractor>>(m, "BackgroundSubtractor", "Base background subtractor");
  // Rho background subtractor
  py::class_<mammoth::RhoSubtractor, mammoth::BackgroundSubtractor, std::shared_ptr<mammoth::RhoSubtractor>>(m, "RhoSubtractor", "Rho based background subtraction")
    .def(
      py::init<bool, bool>(),
        "use_rho_M"_a = true,
        "use_safe_mass"_a = true
      )
    .def_readwrite("use_rho_M", &mammoth::RhoSubtractor::useRhoM)
    .def_readwrite("use_safe_mass", &mammoth::RhoSubtractor::useSafeMass)
    .def("__repr__", [](const mammoth::RhoSubtractor &s) {
      return s.to_string();
    })
  ;
  // Constituent subtraction
  py::class_<mammoth::ConstituentSubtractor, mammoth::BackgroundSubtractor, std::shared_ptr<mammoth::ConstituentSubtractor>>(m, "ConstituentSubtractor", "Background subtraction via Constituent Subtraction")
    .def(
      py::init<double, double, double, std::string>(),
        "r_max"_a = 0.25,
        "alpha"_a = 0.0,
        "rapidity_max"_a = DEFAULT_RAPIDITY_MAX,
        "distance_measure"_a = "delta_R"
      )
    .def_readwrite("r_max", &mammoth::ConstituentSubtractor::rMax)
    .def_readwrite("alpha", &mammoth::ConstituentSubtractor::alpha)
    .def_readwrite("rapidity_max", &mammoth::ConstituentSubtractor::rapidityMax)
    .def_readwrite("distance_measure", &mammoth::ConstituentSubtractor::distanceMeasure)
    .def("__repr__", [](const mammoth::ConstituentSubtractor &s) {
      return s.to_string();
    })
  ;
  // Main container for background subtraction configuration
  py::class_<mammoth::BackgroundSubtraction>(m, "BackgroundSubtraction", "Background subtraction settings")
    .def(
      py::init<mammoth::BackgroundSubtraction_t, std::shared_ptr<mammoth::BackgroundEstimator>, std::shared_ptr<mammoth::BackgroundSubtractor> >(),
        "type"_a,
        "estimator"_a = nullptr,
        "subtractor"_a = nullptr
      )
    .def_readwrite("type", &mammoth::BackgroundSubtraction::type)
    .def_readwrite("estimator", &mammoth::BackgroundSubtraction::estimator)
    .def_readwrite("subtractor", &mammoth::BackgroundSubtraction::subtractor)
    .def("__repr__", [](const mammoth::BackgroundSubtraction &s) {
      return s.to_string();
    })
  ;

  m.def("find_jets", &findJets<float>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                          py::kw_only(),
                                          "jet_finding_settings"_a,
                                          "background_px"_a, "background_py"_a, "background_pz"_a, "background_E"_a,
                                          "background_subtraction"_a,
                                          "user_index"_a = std::nullopt,
                                          "release_gil"_a = false,
                                          "Jet finding function"
                                          );
  m.def("find_jets", &findJets<double>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                           py::kw_only(),
                                           "jet_finding_settings"_a,
                                           "background_px"_a, "background_py"_a, "background_pz"_a, "background_E"_a,
                                           "background_subtraction"_a,
                                           "user_index"_a = std::nullopt,
                                           "release_gil"_a = false,
                                           "Jet finding function"
                                           );

  // Wrapper for reclustered jet outputs
  py::class_<mammoth::JetSubstructure::ColumnarSplittings>(m, "ColumnarSplittings", "Columnar splittings output")
    .def_readonly("kt", &mammoth::JetSubstructure::ColumnarSplittings::kt)
    .def_readonly("delta_R", &mammoth::JetSubstructure::ColumnarSplittings::deltaR)
    .def_readonly("z", &mammoth::JetSubstructure::ColumnarSplittings::z)
    .def_readonly("tau", &mammoth::JetSubstructure::ColumnarSplittings::tau)
    .def_readonly("parent_index", &mammoth::JetSubstructure::ColumnarSplittings::parentIndex)
  ;
  py::class_<mammoth::JetSubstructure::ColumnarSubjets>(m, "ColumnarSubjets", "Columnar splittings output")
    .def_readonly("splitting_node_index", &mammoth::JetSubstructure::ColumnarSubjets::splittingNodeIndex)
    .def_readonly("part_of_iterative_splitting", &mammoth::JetSubstructure::ColumnarSubjets::partOfIterativeSplitting)
    .def_readonly("constituent_indices", &mammoth::JetSubstructure::ColumnarSubjets::constituentIndices)
  ;

  py::class_<mammoth::JetSubstructure::JetSubstructureSplittings>(m, "JetSubstructureSplittings", "Jet substructure splittings")
    .def("splittings", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSplittings {
      return substructure.GetSplittings().GetSplittings();
    })
    .def("subjets", [](mammoth::JetSubstructure::JetSubstructureSplittings & substructure) -> mammoth::JetSubstructure::ColumnarSubjets {
      return substructure.GetSubjets().GetSubjets();
    })
  ;

  // Jet reclustering
  m.def("recluster_jet", &reclusterJet<float>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                               py::kw_only(),
                                               "jet_finding_settings"_a,
                                               "user_index"_a = std::nullopt,
                                               "store_recursive_splittings"_a = true,
                                               "release_gil"_a = false,
                                               "Recluster the given jet");
  m.def("recluster_jet", &reclusterJet<double>, "px"_a, "py"_a, "pz"_a, "E"_a,
                                               py::kw_only(),
                                               "jet_finding_settings"_a,
                                               "user_index"_a = std::nullopt,
                                               "store_recursive_splittings"_a = true,
                                               "release_gil"_a = false,
                                               "Recluster the given jet");

  // ALICE
  // Fast sim
  py::enum_<alice::fastsim::Period_t>(m, "TrackingEfficiencyPeriod",  py::arithmetic(), "Tracking efficiency periods")
    .value("disabled", alice::fastsim::Period_t::kDisabled, "Disabled. Always return 1")
    .value("LHC11h", alice::fastsim::Period_t::kLHC11h, "Run1 PbPb - LHC11h")
    .value("LHC15o", alice::fastsim::Period_t::kLHC15o, "Run2 PbPb - LHC15o")
    .value("LHC18qr", alice::fastsim::Period_t::kLHC18qr, "Run2 PbPb - LHC18{q,r}")
    .value("LHC11a", alice::fastsim::Period_t::kLHC11a, "Run1 pp - LHC11a (2.76 TeV)")
    .value("pA", alice::fastsim::Period_t::kpA, "Generic pA")
    .value("pp", alice::fastsim::Period_t::kpp, "Generic pp");
  py::enum_<alice::fastsim::EventActivity_t>(m, "TrackingEfficiencyEventActivity", py::arithmetic(), "Event activity for tracking efficiency")
    .value("inclusive", alice::fastsim::EventActivity_t::kInclusive, "Inclusive event activity, for say, pp, or MB PbPb.")
    .value("central_00_10", alice::fastsim::EventActivity_t::k0010, "0-10% central event activity")
    .value("mid_central_10_30", alice::fastsim::EventActivity_t::k1030, "10-30% mid-central event activity")
    .value("semi_central_30_50", alice::fastsim::EventActivity_t::k3050, "30-50% semi-central event activity")
    .value("peripheral_50_90", alice::fastsim::EventActivity_t::k5090, "50-90% peripheral event activity")
    .value("invalid", alice::fastsim::EventActivity_t::kInvalid, "Invalid event activity");

  m.def("find_event_activity", &alice::fastsim::findEventActivity, "value"_a,
        "Utility to convert a numerical event activity value to an event activity enumeration value for calling the tracking efficiency.");
  m.def("fast_sim_tracking_efficiency", py::vectorize(alice::fastsim::trackingEfficiencyByPeriod),
        "track_pt"_a, "track_eta"_a, "event_activity"_a, "period"_a,
        "Fast sim via tracking efficiency parametrization", py::call_guard<mammoth::python::JetFindingLoggingStdout, mammoth::python::JetFindingLoggingStderr>());

  // Analysis utilities
  m.def("smooth_array", [](std::vector<double> array, int nTimes){
    auto res = mammoth::analysis::smoothArray<double>(array, nTimes);
    // NOTE: RJE: I believe this copies the output. It's a bit inefficient, but it's not performance critical, so good enough.
    return py::array_t<double>(res.size(), res.data());
    },
    "array"_a,
    "n_times"_a = 1,
    "Smooth an array according to the ROOT TH1::Smooth formulation.");
  m.def("smooth_array_f", [](std::vector<float> array, int nTimes){
    auto res = mammoth::analysis::smoothArray<float>(array, nTimes);
    // NOTE: RJE: I believe this copies the output. It's a bit inefficient, but it's not performance critical, so good enough.
    return py::array_t<float>(res.size(), res.data());
    },
    "array"_a,
    "n_times"_a = 1,
    "Smooth an array according to the ROOT TH1::Smooth formulation (float version).");
}
