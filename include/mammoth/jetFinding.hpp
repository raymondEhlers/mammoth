#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/AreaDefinition.hh>
#include <fastjet/GhostedAreaSpec.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>
#include <fastjet/tools/Subtractor.hh>
#include <fastjet/contrib/ConstituentSubtractor.hh>

namespace mammoth {

// Convenience
template<typename T>
using FourVectorTuple = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>;

template<typename T>
struct OutputWrapper {
  FourVectorTuple<T> jets;
  std::vector<std::vector<unsigned int>> constituent_indices;
  std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>> subtracted;
};

template<typename T>
std::vector<fastjet::PseudoJet> vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors
);

template<typename T>
FourVectorTuple<T> pseudoJetsToVectors(
    const std::vector<fastjet::PseudoJet> & jets
);

std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
);

std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
);

template<typename T>
//std::tuple<FourVectorTuple<T>, std::vector<std::vector<unsigned int>>, std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>>> findJets(
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  std::tuple<double, double> etaRange = std::make_tuple(-0.9, 0.9),
  double minJetPt = 1
);

/********************
  * Implementations *
  *******************/

template<typename T>
std::vector<fastjet::PseudoJet> vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors
)
{
    std::vector<fastjet::PseudoJet> particles;
    const auto & [px, py, pz, E] = fourVectors;
    for (std::size_t i = 0; i < px.size(); ++i) {
        particles.emplace_back(fastjet::PseudoJet(px[i], py[i], pz[i], E[i]));
        particles.back().set_user_index(i);
    }
    return particles;
}

template<typename T>
FourVectorTuple<T> pseudoJetsToVectors(
    const std::vector<fastjet::PseudoJet> & jets
)
{
  std::size_t nJets = jets.size();
  std::cout << "nJets: " << nJets << "\n";
  std::vector<T> px(nJets), py(nJets), pz(nJets), E(nJets);

  std::size_t i = 0;
  for (const auto & pseudoJet : jets) {
    px[i] = pseudoJet.px();
    py[i] = pseudoJet.py();
    pz[i] = pseudoJet.pz();
    E[i] = pseudoJet.e();
    ++i;
  }
  return std::make_tuple(px, py, pz, E);
}

//template<typename T>
//std::tuple<FourVectorTuple<T>, std::vector<std::vector<unsigned int>>, std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>>> findJets(
template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  std::tuple<double, double> etaRange,
  double minJetPt
)
{
  // Validation
  std::map<std::string, fastjet::JetAlgorithm> jetAlgorithms = {
    {"anti-kt", fastjet::JetAlgorithm::antikt_algorithm},
    {"kt", fastjet::JetAlgorithm::kt_algorithm},
    {"CA", fastjet::JetAlgorithm::cambridge_algorithm},
  };
  fastjet::JetAlgorithm jetAlgorithm(jetAlgorithms.at(jetAlgorithmStr));

  // General settings
  double etaMin = std::get<0>(etaRange);
  double etaMax = std::get<1>(etaRange);
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
  // TODO: Add these options...
  bool useConstituentSubtraction = false;
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
  //fastjet::JetAlgorithm jetAlgorithm(fastjet::antikt_algorithm);
  fastjet::RecombinationScheme recombinationScheme(fastjet::E_scheme);
  fastjet::Strategy strategy(fastjet::Best);
  fastjet::AreaType areaType(fastjet::active_area);
  // Derived fastjet settings
  fastjet::JetDefinition jetDefinition(jetAlgorithm, jetR, recombinationScheme, strategy);
  fastjet::AreaDefinition areaDefinition(areaType, ghostAreaSpec);
  fastjet::Selector selJets = fastjet::SelectorPtRange(jetPtMin, jetPtMax) && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax) && fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);

  // Convert numpy input to pseudo jets.
  //auto particlePseudoJets = numpyToPseudoJet(pxIn, pyIn, pzIn, EIn, indexIn);
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors);

  // Setup the background estimator to be able to make the estimation.
  backgroundEstimator.set_particles(particlePseudoJets);

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
    particlePseudoJets = constituentSubtraction->subtract_event(particlePseudoJets);
    subtractedToUnsubtractedIndices = updateSubtractedConstituentIndices(particlePseudoJets);
  }

  // Perform jet finding on signal
  fastjet::ClusterSequenceArea cs(particlePseudoJets, jetDefinition, areaDefinition);
  auto jets = cs.inclusive_jets(minJetPt);
  for (auto j : jets) {
    std::cout << "j pt=" << j.perp() << "\n";
  }
  // Apply the subtractor when appropriate
  if (!useConstituentSubtraction) {
    jets = (*subtractor)(jets);
  }
  for (auto j : jets) {
    std::cout << "j pt=" << j.perp() << "\n";
  }

  // It's also not uncommon to apply a sorting by E or pt.
  jets = fastjet::sorted_by_pt(jets);
  for (auto j : jets) {
    std::cout << "j pt=" << j.perp() << "\n";
  }

  // Now, handle returning the values.
  // First, we need to extract the constituents.
  //auto & [px, py, pz, E] = pseudoJetsToNumpy(jets);
  std::cout << "jets.size() before: " << jets.size() << "\n";
  auto numpyJets = pseudoJetsToVectors<T>(jets);
  std::cout << "jets.size()  after: " << jets.size() << "\n";
  auto & [px, py, pz, E] = numpyJets;
  for (std::size_t i = 0; i < px.size(); ++i) {
    std::cout << "j[" << i << "] pt=" << std::sqrt(std::pow(px[i], 2) + std::pow(py[i], 2)) << "\n";
  }

  // Then, we convert the jets themselves into vectors to return.
  auto constituentIndices = constituentIndicesFromJets(jets);

  if (useConstituentSubtraction) {
    // NOTE: particlePseudoJets are actually the subtracted constituents now.
    return OutputWrapper<T>{
      numpyJets, constituentIndices, std::make_tuple(pseudoJetsToVectors<T>(particlePseudoJets), subtractedToUnsubtractedIndices)
    };
  }
  return OutputWrapper<T>{numpyJets, constituentIndices, {}};
}

}
