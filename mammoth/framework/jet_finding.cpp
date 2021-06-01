#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/AreaDefinition.hh>
#include <fastjet/GhostedAreaSpec.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>
#include <fastjet/tools/Subtractor.hh>
#include <fastjet/contrib/ConstituentSubtractor.hh>

namespace py = pybind11;
// Shorthand for literals
using namespace pybind11::literals;

// Convenience
template<class T>
using FourVectorTuple = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>

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
std::vector<fastjet::PseudoJet> constructPseudojetsFromNumpy(const py::array_t<double, py::array::c_style | py::array::forcecast> & jets)
{
  // Retrieve array and relevant information
  py::buffer_info info = jets.request();
  // I'm not sure which one of these is better.
  //auto inputJets = static_cast<double *>(info.ptr);
  auto inputJets = jets.data();
  std::vector<fastjet::PseudoJet> outputJets;
  // This defines our numpy array shape.
  int nParticles = info.shape[0];
  int nParams = info.shape[1];
  //std::cout << "nParams: " << nParams << ", nParticles: " << nParticles << "\n";

  // Validation.
  if (nParams != 4) {
    throw std::runtime_error("Number of params is not correct. Should be four per particle.");
  }
  // Convert the arrays
  for (size_t i = 0; i < nParticles; ++i) {
    /*std::cout << "i: " << i << " inputs: " << inputJets[i * nParams + 0] << " " << inputJets[i * nParams + 1]
      << " " << inputJets[i * nParams + 2] << " " <<  inputJets[i * nParams + 3] << "\n";*/
    outputJets.push_back(fastjet::PseudoJet(
      inputJets[i * nParams + 0], inputJets[i * nParams + 1],
      inputJets[i * nParams + 2], inputJets[i * nParams + 3]));
  }

  return outputJets;
}

std::vector<fastjet::PseudoJet> numpyToPseudoJet(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & EIn
  //const py::array_t<double, py::array::c_style | py::array::forcecast> & particleIndexIn
)
{
  // Retrieve array and relevant information
  py::buffer_info infoPx = pxIn.request();
  auto px = pxIn.data();
  // This defines our numpy array shape.
  int nParticles = info.shape[0];
  //int nParams = info.shape[1];
  // py
  auto py = pyIn.data();
  auto pz = pzIn.data();
  auto E = EIn.data();
  auto particleIndex = particleIndexIn.data();

  std::vector<fastjet::PseudoJet> inputVectors;

  // Convert the arrays
  for (size_t i = 0; i < nParticles; ++i) {
    /*std::cout << "i: " << i << " inputs: " << inputJets[i * nParams + 0] << " " << inputJets[i * nParams + 1]
      << " " << inputJets[i * nParams + 2] << " " <<  inputJets[i * nParams + 3] << "\n";*/
    outputJets.emplace_back(fastjet::PseudoJet(px[i], py[i], pz[i], E[i]));
    //outputJets.back().set_user_index(particleIndex[i]);
    outputJets.back().set_user_index(i);
  }
  return outputJets;
}

std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> pseudoJetsToNumpy(
    const & std::vector<fastjet::PseudoJet> jets
)
{
  std::size_t nJets = jets.size();
  std::vector<T> px(nJets), py(nJets), pz(nJets), E(nJets);

  for (auto pseudoJet : pseudoJets) {
    px.emplace_back(pseudoJet.px());
    py.emplace_back(pseudoJet.py());
    pz.emplace_back(pseudoJet.pz());
    E.emplace_back(pseudoJet.e());
  }
  return std::make_tuple(px, py, pz, E);
}

std::vector<std::vector<unsigned int>> jetsToConstituentIndices(
  const & std::vector<fastjet::PseudoJet> jets
)
{
  std::vector<std::vector<unsigned int>> indices;
  for (auto jet : jets) {
    std::vector<unsigned int> constituentIndicesInJet;
    for (auto constituent : jet.constituents()) {
      constituentIndicesInJet.push_back(constituent.user_index());
    }
    indices.emplace_back(constituentIndicesInJet);
  }
  return indices;
}

std::vector<unsigned int> & updateSubtractedConstituentIndices(
  const & std::vector<fastjet::PseudoJet> pseudoJets
)
{
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  for (unsigned int i = 0; i < pseudoJets.size(); ++i) {
    subtractedToUnsubtractedIndices.push_back(pseudoJets[i].user_index());
    // The indexing may be different due to the subtraction. So we reset it be certain.
    pseudoJets[i].set_user_index(i);
  }

  return subtractedToUnsubtractedIndices;
}

std::tuple<FourVectorTuple<T>, std::vector<std::vector<unsigned int>, std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>> findJets(
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pxIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pyIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & pzIn,
  const py::array_t<double, py::array::c_style | py::array::forcecast> & EIn,
  //const py::array_t<double, py::array::c_style | py::array::forcecast> & indexIn
  double jetR,
  std::tuple<double, double> etaRange = std::make_tuple(-0.9, 0.9)
)
{
  // General settings
  double etaMin = etaRange.get<0>();
  double etaMax = etaRange.get<1>();
  // Ghost settings
  double ghostEtaMin = etaMin;
  double ghostEtaMax = etaMax;
  double ghostArea = 0.005;
  int ghostRepeatN = 1;
  double ghostktMean = 1e-100;
  double gridScatter = 1.0;
  double ktScatter = 0.1;
  fastjet::GhostedAreaSpec ghostAreaSpec(ghostEtaMax, ghostRepeatN, ghostArea, gridScatter, ktScatter, ghostktMean);

  // Background settings
  double backgroundJetR = 0.2;
  double backgroundJetEtaMin = etaMin;
  double backgroundJetEtaMax = etaMax;
  double backgroundJetPhiMin = 0;
  double backgroundJetPhiMax = 2 * M_PI;
  // Fastjet background settings
  fastjet::JetAlgorithm backgroundJetAlgorithm(fastjet::kt_algorithm);
  fastjet::RecombinationScheme backgroundRecombinationScheme(fastjet::E_scheme);
  fastjet::Strategy backgroundStrategy(fastjet::Best);
  fastjet::AreaType backgroundAreaType(fastjet::active_area);
  // Derived fastjet settings
  fastjet::JetDefinition backgroundJetDefinition(backgroundJetAlgorithm, backgroundJetR, backgroundRecombinationScheme, backgroundStrategy);
  fastjet::AreaDefinition backgroundAreaDefinition(backgroundAreaType, ghostAreaSpec);
  fastjet::Selector selRho = fastjet::SelectorRapRange(backgroundJetEtaMin, backgroundJetEtaMax) && fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax) && !fastjet::SelectorNHardest(2);
  // Constituent subtraction options (if used)
  double constituentSubAlpha = 1.0;
  double constituentSubRMax = 0.25;

  // Finally, define the background estimator
  // This is needed for all background subtraction cases.
  fastjet::JetMedianBackgroundEstimator backgroundEstimator(selRho, backgroundJetDefinition, backgroundAreaDefinition);

  // Signal jet settings
  // Again, these should all be settable, but I wanted to keep the signature simple, so I just define them here with some reasonable(ish) defaults.
  double jetPtMin = 0;
  double jetPtMax = 1000;
  // Would often set as abs(eta - R), but should be configurable.
  double jetEtaMin = etaMin + jetR;
  double jetEtaMax = etaMax - jetR;
  double jetPhiMin = 0;
  double jetPhiMax = 2 * M_PI;
  // Fastjet settings
  fastjet::JetAlgorithm jetAlgorithm(fastjet::antikt_algorithm);
  fastjet::RecombinationScheme recombinationScheme(fastjet::E_scheme);
  fastjet::Strategy strategy(fastjet::Best);
  fastjet::AreaType areaType(fastjet::active_area);
  // Derived fastjet settings
  fastjet::JetDefinition jetDefinition(jetAlgorithm, jetR, recombinationScheme, strategy);
  fastjet::AreaDefinition areaDefinition(areaType, ghostAreaSpec);
  fastjet::Selector selJets = fastjet::SelectorPtRange(jetPtMin, jetPtMax) && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax) && fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);

  // Convert numpy input to pseudo jets.
  auto inputVectors = numpyToPseudoJet(pxIn, pyIn, pzIn, EIn, indexIn);

  // Setup the background estimator to be able to make the estimation.
  backgroundEstimator.set_particles(inputPseudoJets);

  // Now, deal with applying the background subtraction.
  // The subtractor will subtract the background from jets. It's not used in the case of constituent subtraction.
  std::shared_ptr<fastjet::Subtractor> subtractor = nullptr;
  // The constituent subtraction (here, it's implemented as event-wise subtraction, but that doesn't matter) takes
  // a different approach to background subtraction. It's used here to illustrate a different work flow.
  std::shared_ptr<fastjet::contrib::ConstituentSubtractor> constituentSubtraction = nullptr;
  // Now, set them up as necessary.
  if (!useConstituentSubtraction) {
    subtractor = std::make_shared<fastjet::Subtractor>(&backgroundEstimator);
  }
  else {
    constituentSubtraction = std::make_shared<fastjet::contrib::ConstituentSubtractor>(&backgroundEstimator);
    constituentSubtraction->set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR);
    constituentSubtraction->set_max_distance(constituentSubRMax);
    constituentSubtraction->set_alpha(constituentSubAlpha);
    constituentSubtraction->set_ghost_area(ghostArea);
    constituentSubtraction->set_max_eta(backgroundJetEtaMax);
    constituentSubtraction->set_background_estimator(&backgroundEstimator);
  }

  // For constituent subtraction, we subtract the input particles
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  if (useConstituentSubtraction) {
    inputVectors = constituentSubtraction->subtract_event(inputVectors);
    subtractedToUnsubtractedIndices = updateSubtractedConstituentIndices(inputVectors);
  }

  // Perform jet finding on signal
  fastjet::ClusterSequenceArea cs(inputVectors, jetDefinition, areaDefinition);
  auto jets = cs->inclusive_jets(0);
  // Apply the subtractor when appropriate
  if (!useConstituentSubtraction) {
    jets = (*subtractor)(jets);
  }

  // It's also not uncommon to apply a sorting by E or pt.
  jets = fastjet::sorted_by_pt(jets);

  // Now, handle returning the values.
  // First, we need to extract the constituents.
  //auto & [px, py, pz, E] = pseudoJetsToNumpy(jets);
  auto & numpyJets = pseudoJetsToNumpy(jets);
  // Then, we convert the jets themselves into vectors to return.
  auto constituentIndices = jetsToConstituentIndices(jets);

  if (useConstituentSubtraction) {
    return std::make_tuple(numpyJets, constituentIndices, std::make_tuple(pseudoJetsToNumpy(inputPseudoJets), subtractedToUnsubtractedIndices);
  }
  return std::make_tuple(numpyJets, constituentIndices);

}
