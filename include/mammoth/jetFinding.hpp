#include <string>
#include <utility>

#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include <fastjet/JetDefinition.hh>
#include <fastjet/AreaDefinition.hh>
#include <fastjet/GhostedAreaSpec.hh>
#include <fastjet/tools/JetMedianBackgroundEstimator.hh>
#include <fastjet/tools/Subtractor.hh>
#include <fastjet/contrib/ConstituentSubtractor.hh>

// operator<< has to be forward declared carefully to stay in the global namespace so that it works with CINT.
// For generally how to keep the operator in the global namespace, See: https://stackoverflow.com/a/38801633
// NOTE: This probably isn't necessary for mammoth, but I'm copying my code from AliPhysics, and trying to modify
//       it as little as possible. So since this doesn't cause an issue, I will leave it as is.
namespace mammoth {
namespace SubstructureTree {
  class Subjets;
  class JetSplittings;
  class JetConstituents;
  class JetSubstructureSplittings;
}
}
std::ostream& operator<<(std::ostream& in, const mammoth::SubstructureTree::Subjets& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::SubstructureTree::JetSplittings& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::SubstructureTree::JetConstituents& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::SubstructureTree::JetSubstructureSplittings& myTask);
void swap(mammoth::SubstructureTree::Subjets& first,
     mammoth::SubstructureTree::Subjets& second);
void swap(mammoth::SubstructureTree::JetSplittings& first,
     mammoth::SubstructureTree::JetSplittings& second);
void swap(mammoth::SubstructureTree::JetConstituents& first,
     mammoth::SubstructureTree::JetConstituents& second);
void swap(mammoth::SubstructureTree::JetSubstructureSplittings& first,
     mammoth::SubstructureTree::JetSubstructureSplittings& second);

namespace mammoth {

// Convenience
template<typename T>
using FourVectorTuple = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>;

/**
 * @brief Jet finding output wrapper
 *
 * @tparam T Input data type (usually float or double)
 */
template<typename T>
struct OutputWrapper {
  FourVectorTuple<T> jets;
  std::vector<std::vector<unsigned int>> constituent_indices;
  std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>> subtracted;
};

/**
 * @brief Constituent subtraction settings
 *
 * Just a simple container for area related settings
 */
struct AreaSettings {
  std::string areaType{"active_area"};
  double ghostArea{0.005};
};

/**
 * @brief Constituent subtraction settings
 *
 * Just a simple container for constituent subtraction related settings.
 *
 * @param rMax Delta R max parameter
 * @param alpha Alpha parameter
 */
struct ConstituentSubtractionSettings {
  double rMax{0.25};
  double alpha{1.0};
};

/**
 * @brief Convert column vectors to a vector of PseudoJets
 *
 * @tparam T Input data type (usually float or double)
 * @param fourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @return std::vector<fastjet::PseudoJet> Vector of PseudoJets containing the same information.
 */
template<typename T>
std::vector<fastjet::PseudoJet> vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors
);

/**
 * @brief Convert vector of PseudoJets to a column of vectors.
 *
 * @tparam T Input data type (usually float or double)
 * @param jets Input pseudo jets
 * @return FourVectorTuple<T> Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 */
template<typename T>
FourVectorTuple<T> pseudoJetsToVectors(
    const std::vector<fastjet::PseudoJet> & jets
);

/**
 * @brief Extract constituent indices from jets.
 *
 * @param jets Jets with constituents.
 * @return std::vector<std::vector<unsigned int>> The indices of all constituents in all jets.
 */
std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
);

/**
 * @brief Update the indices in subtracted constituents.
 *
 * Updating this indexing ensures that we can keep track of everything.
 *
 * @param pseudoJets Subtracted input particles.
 * @return std::vector<unsigned int> Map of indices from subtracted constituents to unsubtracted constituents.
 */
std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
);

/**
  * @brief Implementatino of main jet finder.
  *
  * @tparam T Input data type (usually float or double)
  * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
  * @param jetR jet resolution parameter
  * @param jetAlgorithmStr jet alogrithm
  * @param areaSettings Area settings
  * @param etaRange Eta range. Tuple of min and max. Default: (-0.9, 0.9)
  * @param minJetPt Minimum jet pt. Default: 1.
  * @param backgroundSubtraction If true, enable rho background subtraction
  * @param constituentSubtraction If provided, configure constituent subtraction according to given settings.
  * @return OutputWrapper<T> Output from jet finding.
  */
template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  AreaSettings areaSettings,
  std::tuple<double, double> etaRange = {-0.9, 0.9},
  double minJetPt = 1,
  bool backgroundSubtraction = false,
  std::optional<ConstituentSubtractionSettings> constituentSubtraction = std::nullopt
);

namespace SubstructureTree {

/**
 * @class Subjets
 * @brief Subjets of a jet.
 *
 * Store the subjets as determined by declustering a jet.
 *
 * @author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
 * @date 9 Feb 2020
 */
class Subjets {
 public:
  Subjets();
  // Additional constructors
  Subjets(const Subjets & other);
  Subjets& operator=(Subjets other);
  friend void ::swap(Subjets & first, Subjets & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~Subjets() {}

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const Subjets &myTask);
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<unsigned short> fSplittingNodeIndex;        ///<  Index of the parent splitting node.
  std::vector<bool> fPartOfIterativeSplitting;            ///<  True if the splitting is follow an iterative splitting.
  std::vector<std::vector<unsigned short>> fConstituentIndices;        ///<  Constituent jet indices (ie. indexed by the stored jet constituents, not the global index).
};

/**
 * @class JetSplittings
 * @brief Properties of jet splittings.
 *
 * Store the properties of jet splittings determined by declustering a jet.
 *
 * @author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
 * @date 9 Feb 2020
 */
class JetSplittings {
 public:
  JetSplittings();
  // Additional constructors
  JetSplittings(const JetSplittings & other);
  JetSplittings& operator=(JetSplittings other);
  friend void ::swap(JetSplittings & first, JetSplittings & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetSplittings() {}

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  std::tuple<float, float, float, short> GetSplitting(int i) const;
  unsigned int GetNumberOfSplittings() const { return fKt.size(); }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetSplittings &myTask);
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<float> fKt;             ///<  kT between the subjets.
  std::vector<float> fDeltaR;         ///<  Delta R between the subjets.
  std::vector<float> fZ;              ///<  Momentum sharing of the splitting.
  std::vector<short> fParentIndex;    ///<  Index of the parent splitting.
};

/**
 * @class JetConstituents
 * @brief Jet constituents.
 *
 * Store the constituents associated with a jet.
 *
 * @author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
 * @date 9 Feb 2020
 */
class JetConstituents
{
 public:
  JetConstituents();
  // Additional constructors
  JetConstituents(const JetConstituents & other);
  JetConstituents& operator=(JetConstituents other);
  friend void ::swap(JetConstituents & first, JetConstituents & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetConstituents() {}

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Getters and setters
  void AddJetConstituent(const AliVParticle* part, const int & id);
  std::tuple<float, float, float, int> GetJetConstituent(int i) const;
  static const int GetGlobalIndexOffset() { return fgkGlobalIndexOffset; }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetConstituents &myTask);
  std::ostream & Print(std::ostream &in) const;

 protected:
  static const int fgkGlobalIndexOffset;  ///<  Offset for GlobalIndex values in the ID to ensure it never conflicts with the label.

  std::vector<float> fPt;                 ///<  Jet constituent pt
  std::vector<float> fEta;                ///<  Jet constituent eta
  std::vector<float> fPhi;                ///<  Jet constituent phi
  std::vector<int> fID;                   ///<  Jet constituent identifier. MC label (via GetLabel()) or global index (with offset defined above).
};

/**
 * @class JetSubstructureSplittings
 * @brief Jet substructure splittings.
 *
 * Jet substructure splitting properties. There is sufficient information to calculate any
 * additional splitting properties.
 *
 * @author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
 * @date 9 Feb 2020
 */
class JetSubstructureSplittings {
 public:
  JetSubstructureSplittings();
  // Additional constructors
  JetSubstructureSplittings(const JetSubstructureSplittings & other);
  JetSubstructureSplittings& operator=(JetSubstructureSplittings other);
  friend void ::swap(JetSubstructureSplittings & first, JetSubstructureSplittings & second);
  // Avoid implementing move since c++11 is not allowed in the header
  virtual ~JetSubstructureSplittings() {}

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Setters
  void SetJetPt(float pt) { fJetPt = pt; }
  void AddJetConstituent(const AliVParticle* part, const int & id);
  void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  // Getters
  float GetJetPt() { return fJetPt; }
  std::tuple<float, float, float, int> GetJetConstituent(int i) const;
  std::tuple<float, float, float, short> GetSplitting(int i) const;
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;
  unsigned int GetNumberOfSplittings() { return fJetSplittings.GetNumberOfSplittings(); }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetSubstructureSplittings &myTask);
  std::ostream & Print(std::ostream &in) const;

 private:
  // Jet properties
  float fJetPt;                                           ///<  Jet pt.
  SubstructureTree::JetConstituents fJetConstituents;     ///<  Jet constituents
  SubstructureTree::JetSplittings fJetSplittings;         ///<  Jet splittings.
  SubstructureTree::Subjets fSubjets;                     ///<  Subjets within the jet.
};

} /* namespace SubstructureTree */

/**
  * @brief Implementation of jet reclustering
  *
  * @tparam T Input data type (usually float or double)
  * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
  * @param jetR jet resolution parameter. Default: 1.
  * @param jetAlgorithmStr jet alogrithm. Default: "CA".
  * @param areaSettings Area settings. Default: None.
  * @param etaRange Eta range. Tuple of min and max. Default: (-1, 1)
  * @return OutputWrapper<T> Output from reclustering
  */
template<typename T>
OutputWrapper<T> jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  double jetR = 1,
  std::string jetAlgorithmStr = "CA",
  std::optional<AreaSettings> areaSettings = std::nullopt,
  std::tuple<double, double> etaRange = {-1, 1}
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
  // Setup
  std::size_t nJets = jets.size();
  std::vector<T> px(nJets), py(nJets), pz(nJets), E(nJets);

  // Fill column vectors
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

fastjet::JetAlgorithm getJetAlgorithm(std::string jetAlgorithmStr)
{
  // Jet algorithm name
  std::map<std::string, fastjet::JetAlgorithm> jetAlgorithms = {
    {"anti-kt", fastjet::JetAlgorithm::antikt_algorithm},
    {"kt", fastjet::JetAlgorithm::kt_algorithm},
    {"CA", fastjet::JetAlgorithm::cambridge_algorithm},
  };
  return jetAlgorithms.at(jetAlgorithmStr);
}

fastjet::AreaType getAreaType(const AreaSettings & areaSettings)
{
  // Area type
  std::map<std::string, fastjet::AreaType> areaTypes = {
    {"active_area", fastjet::AreaType::active_area},
    {"passive_area", fastjet::AreaType::passive_area},
  };
  return areaTypes.at(areaSettings.areaType);
}

template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  AreaSettings areaSettings,
  std::tuple<double, double> etaRange,
  double minJetPt,
  bool backgroundSubtraction,
  std::optional<ConstituentSubtractionSettings> constituentSubtraction
)
{
  // Validation
  // Main jet algorithm
  fastjet::JetAlgorithm jetAlgorithm = getJetAlgorithm(jetAlgorithmStr);
  // Main Area type
  fastjet::AreaType areaType = getAreaType(areaSettings);

  // Convert column vector input to pseudo jets.
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors);

  // Notify about the settings for the jet finding.
  // NOTE: This can be removed eventually. For now (July 2021), it wll be routed to debug level
  //       so we can be 100% sure about what is being calculated.
  std::cout << std::boolalpha << "Settings:\n"
    << "\tGhost area: " << areaSettings.ghostArea << "\n"
    << "\tBackground subtraction: " << backgroundSubtraction << "\n"
    << "\tConstituent subtraction: " << static_cast<bool>(constituentSubtraction) << "\n";

  // General settings
  double etaMin = std::get<0>(etaRange);
  double etaMax = std::get<1>(etaRange);
  // Ghost settings
  double ghostEtaMin = etaMin;
  double ghostEtaMax = etaMax;
  int ghostRepeatN = 1;
  double ghostktMean = 1e-100;
  double gridScatter = 1.0;
  double ktScatter = 0.1;
  fastjet::GhostedAreaSpec ghostAreaSpec(ghostEtaMax, ghostRepeatN, areaSettings.ghostArea, gridScatter, ktScatter, ghostktMean);

  // Background settings
  // We need to define these basic settings for both rho subtraction and constituent subtraction.
  std::shared_ptr<fastjet::JetMedianBackgroundEstimator> backgroundEstimator = nullptr;
  // We'll also define the constituent subtractor here too.
  std::shared_ptr<fastjet::contrib::ConstituentSubtractor> constituentSubtractor = nullptr;
  if (backgroundSubtraction || constituentSubtraction) {
    double backgroundJetR = 0.2;
    double backgroundJetEtaMin = etaMin;
    double backgroundJetEtaMax = etaMax;
    double backgroundJetPhiMin = 0;
    double backgroundJetPhiMax = 2 * M_PI;
    // Fastjet background settings
    fastjet::JetAlgorithm backgroundJetAlgorithm(fastjet::JetAlgorithm::kt_algorithm);
    fastjet::RecombinationScheme backgroundRecombinationScheme(fastjet::RecombinationScheme::E_scheme);
    fastjet::Strategy backgroundStrategy(fastjet::Strategy::Best);
    fastjet::AreaType backgroundAreaType(fastjet::AreaType::active_area);
    // Derived fastjet settings
    fastjet::JetDefinition backgroundJetDefinition(backgroundJetAlgorithm, backgroundJetR, backgroundRecombinationScheme, backgroundStrategy);
    fastjet::AreaDefinition backgroundAreaDefinition(backgroundAreaType, ghostAreaSpec);
    fastjet::Selector selRho = fastjet::SelectorRapRange(backgroundJetEtaMin, backgroundJetEtaMax) && fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax) && !fastjet::SelectorNHardest(2);

    // Finally, define the background estimator
    // This is needed for all background subtraction cases.
    backgroundEstimator = std::make_shared<fastjet::JetMedianBackgroundEstimator>(selRho, backgroundJetDefinition, backgroundAreaDefinition);

    // Setup the background estimator to be able to make the estimation.
    backgroundEstimator->set_particles(particlePseudoJets);

    // Specific setup for event-wise constituent subtraction
    if (constituentSubtraction) {
      constituentSubtractor = std::make_shared<fastjet::contrib::ConstituentSubtractor>(backgroundEstimator.get());
      constituentSubtractor->set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR);
      constituentSubtractor->set_max_distance(constituentSubtraction->rMax);
      constituentSubtractor->set_alpha(constituentSubtraction->alpha);
      constituentSubtractor->set_ghost_area(areaSettings.ghostArea);
      constituentSubtractor->set_max_eta(backgroundJetEtaMax);
      constituentSubtractor->set_background_estimator(backgroundEstimator.get());
    }
  }

  // Now, setup the subtractor object (when needed), which will subtract the background from jets.
  // NOTE: It's not used in the case of event-wise constituent subtraction, since the subtraction
  //       is applied separately.
  std::shared_ptr<fastjet::Subtractor> subtractor = nullptr;
  // Now, set it up as necessary.
  if (backgroundSubtraction) {
    subtractor = std::make_shared<fastjet::Subtractor>(backgroundEstimator.get());
  }

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
  // NOTE: Jet algorithm defined at the beginning
  // NOTE: Jet area type defined at the beginning
  fastjet::RecombinationScheme recombinationScheme(fastjet::RecombinationScheme::E_scheme);
  fastjet::Strategy strategy(fastjet::Strategy::Best);
  // Derived fastjet settings
  fastjet::JetDefinition jetDefinition(jetAlgorithm, jetR, recombinationScheme, strategy);
  fastjet::AreaDefinition areaDefinition(areaType, ghostAreaSpec);
  fastjet::Selector selJets = fastjet::SelectorPtRange(jetPtMin, jetPtMax) && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax) && fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);

  // For constituent subtraction, we perform event-wise subtraction on the input particles
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  if (constituentSubtractor) {
    particlePseudoJets = constituentSubtractor->subtract_event(particlePseudoJets);
    subtractedToUnsubtractedIndices = updateSubtractedConstituentIndices(particlePseudoJets);
  }

  // Perform jet finding on signal
  fastjet::ClusterSequenceArea cs(particlePseudoJets, jetDefinition, areaDefinition);
  auto jets = cs.inclusive_jets(minJetPt);
  // Apply the subtractor when appropriate
  if (backgroundSubtraction) {
    jets = (*subtractor)(jets);
  }

  // It's also not uncommon to apply a sorting by E or pt.
  jets = fastjet::sorted_by_pt(jets);

  // Now, handle returning the values.
  // First, we need to extract the constituents.
  //auto & [px, py, pz, E] = pseudoJetsToNumpy(jets);
  auto numpyJets = pseudoJetsToVectors<T>(jets);

  // Then, we convert the jets themselves into vectors to return.
  auto constituentIndices = constituentIndicesFromJets(jets);

  if (constituentSubtraction) {
    // NOTE: particlePseudoJets are actually the subtracted constituents now.
    return OutputWrapper<T>{
      numpyJets, constituentIndices, std::make_tuple(pseudoJetsToVectors<T>(particlePseudoJets), subtractedToUnsubtractedIndices)
    };
  }
  return OutputWrapper<T>{numpyJets, constituentIndices, {}};
}

template<typename T>
OutputWrapper<T> jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  std::optional<AreaSettings> areaSettings,
  std::tuple<double, double> etaRange
)
{
  // Jet algorithm
  fastjet::JetAlgorithm jetAlgorithm = getJetAlgorithm(jetAlgorithmStr);
  fastjet::RecombinationScheme recombinationScheme(fastjet::RecombinationScheme::E_scheme);
  fastjet::Strategy strategy(fastjet::Strategy::Best);
  fastjet::JetDefinition jetDefinition(jetAlgorithm, jetR, recombinationScheme, strategy);
  // For area calculation (when desired)
  // Area type
  std::unique_ptr<fastjet::AreaType> areaType = nullptr;
  std::unique_ptr<fastjet::GhostedAreaSpec> ghostSpec = nullptr;
  std::unique_ptr<fastjet::AreaDefinition> areaDefinition = nullptr;
  if (areaSettings) {
    double etaMax = std::get<1>(etaRange);
    int ghostRepeatN = 1;
    areaType = std::make_unique<fastjet::AreaType>(getAreaType(*areaSettings));
    //fastjet::GhostedAreaSpec ghost_spec(1, 1, 0.05);
    //fastjet::AreaDefinition areaDef(areaType, ghost_spec);
    ghostSpec = std::make_unique<fastjet::GhostedAreaSpec>(etaMax, ghostRepeatN, areaSettings->ghostArea);
    areaDefinition = std::make_unique<fastjet::AreaDefinition>(areaType, ghostSpec);
  }

  // Convert column vector input to pseudo jets.
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors);

  // If we use the area definition, we need to create a ClusterSequenceArea.
  // NOTE: The CS has to stay in scope while we explore the splitting history.
  std::unique_ptr<fastjet::ClusterSequence> cs = nullptr;
  if (areaDefinition) {
    cs = std::make_unique<fastjet::ClusterSequenceArea>(particlePseudoJets, jetDefinition, areaDefinition);
  }
  else {
    cs = std::make_unique<fastjet::ClusterSequence>(particlePseudoJets, jetDefinition);
  }
  std::vector<fastjet::PseudoJet> outputJets = cs->inclusive_jets(0);

  fastjet::PseudoJet jj;
  jj = outputJets[0];

  // TODO: Use the declustering results...
  ExtractJetSplittings(jj);
}

}
