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
namespace JetSubstructure {
  class Subjets;
  class JetSplittings;
  class JetSubstructureSplittings;
}
}
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::Subjets& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSplittings& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSubstructureSplittings& myTask);
void swap(mammoth::JetSubstructure::Subjets& first,
     mammoth::JetSubstructure::Subjets& second);
void swap(mammoth::JetSubstructure::JetSplittings& first,
     mammoth::JetSubstructure::JetSplittings& second);
void swap(mammoth::JetSubstructure::JetSubstructureSplittings& first,
     mammoth::JetSubstructure::JetSubstructureSplittings& second);

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
  std::vector<float> jetsArea;
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
  double alpha{0.0};
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
 * @brief Extract jet area for given jets.
 *
 * Recall that the cluster sequence must still exist when extracting the jet area.
 *
 * @param jets Jets to grab the area.
 * @return std::vector<float> Jet area for the given jets.
 */
std::vector<float> extractJetsArea(
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

/// Functionality related to jet substructure
/// Much of it is based on code that I originally wrote for AliPhysics
/// (namely, AliAnalysisTaskJetDynamicalGrooming.{cxx,h})
namespace JetSubstructure {

/**
 * @class ColumnarSubjets
 * @brief Columnar subjets
 *
 * Container for columnar subjets info. It's mainly for convenience in moving over to python.
 */
struct ColumnarSubjets {
  std::vector<unsigned short> splittingNodeIndex;                     ///<  Index of the parent splitting node.
  std::vector<bool> partOfIterativeSplitting;                         ///<  True if the splitting is follow an iterative splitting.
  std::vector<std::vector<unsigned short>> constituentIndices;        ///<  Constituent jet indices (ie. indexed by the stored jet constituents, not the global index).
};

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
  ColumnarSubjets GetSubjets() { return std::move(ColumnarSubjets{fSplittingNodeIndex, fPartOfIterativeSplitting, fConstituentIndices}); }
  //std::tuple<std::vector<unsigned short> &, std::vector<bool> &, std::vector<std::vector<unsigned short>> &> GetSubjets() { return ; }

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
 * @class ColumnarSplittings
 * @brief Columnar jet splittings
 *
 * Container for columnar jet splitings info. It's mainly for convenience in moving over to python.
 */
struct ColumnarSplittings {
  std::vector<float> kt;             ///<  kT between the subjets.
  std::vector<float> deltaR;         ///<  Delta R between the subjets.
  std::vector<float> z;              ///<  Momentum sharing of the splitting.
  std::vector<short> parentIndex;    ///<  Index of the parent splitting.
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
  ColumnarSplittings GetSplittings() { return std::move(ColumnarSplittings{fKt, fDeltaR, fZ, fParentIndex}); }
  //std::tuple<std::vector<float> &, std::vector<float> &, std::vector<float> &, std::vector<short> &> GetSplittings() { return {fKt, fDeltaR, fZ, fParentIndex}; }

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
  virtual ~JetSubstructureSplittings() {}

  /// Reset the properties for the next filling of the tree.
  bool Clear();

  // Setters
  void AddSplitting(float kt, float deltaR, float z, short parentIndex);
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  // Getters
  std::tuple<float, float, float, short> GetSplitting(int i) const;
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;
  unsigned int GetNumberOfSplittings() { return fJetSplittings.GetNumberOfSplittings(); }
  JetSubstructure::JetSplittings & GetSplittings() { return fJetSplittings; }
  JetSubstructure::Subjets & GetSubjets() { return fSubjets; }

  // Printing
  std::string toString() const;
  friend std::ostream & ::operator<<(std::ostream &in, const JetSubstructureSplittings &myTask);
  std::ostream & Print(std::ostream &in) const;

 private:
  // Jet properties
  JetSubstructure::JetSplittings fJetSplittings;         ///<  Jet splittings.
  JetSubstructure::Subjets fSubjets;                     ///<  Subjets within the jet.
};

} /* namespace JetSubstructure */

/**
 * @brief Extract jet splittings recursively.
 *
 * @param jetSplittings Container for storing the jet splittings. We pass as an argument so we can update recursively.
 * @param inputJet Reclustered jet (may be one of the subjets).
 * @param splittingNodeIndex Index for the splitting node.
 * @param followingIterativeSplitting If true, we're following an iterative splitting.
 * @param storeRecursiveSplittings If true, store recursive splittings (in addition to iterative splittings).
 */
void ExtractJetSplittings(
  JetSubstructure::JetSubstructureSplittings & jetSplittings,
  fastjet::PseudoJet & inputJet,
  int splittingNodeIndex,
  bool followingIterativeSplitting,
  const bool storeRecursiveSplittings
);

/**
  * @brief Implementation of jet reclustering
  *
  * @tparam T Input data type (usually float or double)
  * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
  * @param jetR jet resolution parameter. Default: 1.
  * @param jetAlgorithmStr jet alogrithm. Default: "CA".
  * @param areaSettings Area settings. Default: None.
  * @param etaRange Eta range. Tuple of min and max. Default: (-1, 1)
  * @return Jet substructure splittings container
  */
template<typename T>
JetSubstructure::JetSubstructureSplittings jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  double jetR = 1.0,
  std::string jetAlgorithmStr = "CA",
  std::optional<AreaSettings> areaSettings = std::nullopt,
  std::tuple<double, double> etaRange = {-1, 1},
  bool storeRecursiveSplittings = true
);

/************************************************
  * Implementations for templated functionality *
  ***********************************************/

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
    {"active_area_explicit_ghosts", fastjet::AreaType::active_area_explicit_ghosts},
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
  // NOTE: This can be removed eventually. For now (July 2021), it will be routed to debug level
  //       so we can be 100% sure about what is being calculated.
  std::cout << std::boolalpha
    << "Cuts:\n"
    << "\tMin jet pt=" << minJetPt << "\n"
    << "Settings:\n"
    << "\tGhost area: " << areaSettings.ghostArea << "\n"
    << "\tBackground subtraction: " << backgroundSubtraction << "\n"
    << "\tConstituent subtraction: " << static_cast<bool>(constituentSubtraction) << "\n";

  // General settings
  double etaMin = std::get<0>(etaRange);
  double etaMax = std::get<1>(etaRange);
  // Ghost settings
  // ghost eta edges are expected to be symmetric, so we don't actually need the min
  //double ghostEtaMin = etaMin;
  // TODO: This seems wayyyyyyy to sensitive to this value.
  // Use the ALICE value for now.
  double ghostEtaMax = 1.0;
  int ghostRepeatN = 1;
  double gridScatter = 1.0;
  double ktScatter = 0.1;
  double ghostktMean = 1e-100;
  fastjet::GhostedAreaSpec ghostAreaSpec(ghostEtaMax, ghostRepeatN, areaSettings.ghostArea, gridScatter, ktScatter, ghostktMean);

  // Background settings
  // We need to define these basic settings for both rho subtraction and constituent subtraction.
  std::shared_ptr<fastjet::JetMedianBackgroundEstimator> backgroundEstimator = nullptr;
  // We'll also define the constituent subtractor here too.
  std::shared_ptr<fastjet::contrib::ConstituentSubtractor> constituentSubtractor = nullptr;
  if (backgroundSubtraction || constituentSubtraction) {
    double backgroundJetR = 0.2;
    // Use fiducial cut for jet background.
    double backgroundJetEtaMin = etaMin + backgroundJetR;
    double backgroundJetEtaMax = etaMax - backgroundJetR;
    //double backgroundJetPhiMin = 0;
    //double backgroundJetPhiMax = 2 * M_PI;
    // Fastjet background settings
    fastjet::JetAlgorithm backgroundJetAlgorithm(fastjet::JetAlgorithm::kt_algorithm);
    fastjet::RecombinationScheme backgroundRecombinationScheme(fastjet::RecombinationScheme::E_scheme);
    fastjet::Strategy backgroundStrategy(fastjet::Strategy::Best);
    // NOTE: Must include the explicit ghosts - otherwise, excluding the 2 hardest jets won't work!
    //       As described in footnote 27 in the fastjet 3.4 manual:
    //       "If you use non-geometric selectors such as [n hardest] in determining [rho], the area must
    //       have explicit ghosts in order to simplify the determination of the empty area. If it does
    //       not, an error will be thrown"
    fastjet::AreaType backgroundAreaType(fastjet::AreaType::active_area_explicit_ghosts);
    // Derived fastjet settings
    fastjet::JetDefinition backgroundJetDefinition(backgroundJetAlgorithm, backgroundJetR, backgroundRecombinationScheme, backgroundStrategy);
    fastjet::AreaDefinition backgroundAreaDefinition(backgroundAreaType, ghostAreaSpec);
    // NOTE: This applies eta cuts, phi cuts, and removes the two hardest jets (as is standard for rho).
    // NOTE: We want to apply the two hardest removal _after_ the acceptance cuts, so we use "*"
    //       Be aware that the order of the selectors really matters. It applies the right most first if we use "*"
    //
    // Considerations for possible tweaks:
    // - Including or not including the two hardest can have a big impact on the number of jets that are accepted.
    //   Removing them includes more jets (because the median background is smaller)
    // - Since we're using explicit ghosts, we could consider removing ghost only jets with !SelectorIsPureGhost().
    //   Currently, this seems pretty aggressive, but perhaps it makes sense given that the ALICE example doesn't include them.
    fastjet::Selector selRho = !fastjet::SelectorIsPureGhost() * !fastjet::SelectorNHardest(2) * fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax); //* fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax);
    //fastjet::Selector selRho = !fastjet::SelectorNHardest(2) * fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) * fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax);

    //fastjet::Selector selRho = (fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) && fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax)) * !fastjet::SelectorNHardest(2);
    //fastjet::Selector selRho = (fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) && fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax));
    //fastjet::Selector selRho = (fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax)) * !fastjet::SelectorNHardest(2);
    //fastjet::Selector selRho = (!fastjet::SelectorNHardest(2)) * fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) * (fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax));
    //fastjet::Selector selRho = fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) * fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax);
    //fastjet::Selector selRho = fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax) * (fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax));
    //fastjet::Selector selRho = (fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax));
    //fastjet::Selector selRho = (fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax));
    // NOTE: We don't remove jets with tracks > 100 GeV here because it would require copying all of
    //       the input particles (which could have a measurable performance impact) for what I expect
    //       is quite a small impact. I think it will be small because we're concerned with the median
    //       and we exclude the two leading. So unless there are many fake tracks in a single event,
    //       it's unlikely to have a meaningful effect on the median.

    // Finally, define the background estimator
    // This is needed for all background subtraction cases.
    backgroundEstimator = std::make_shared<fastjet::JetMedianBackgroundEstimator>(selRho, backgroundJetDefinition, backgroundAreaDefinition);
    // Ensure rho_m is calculated (should be by default, but just being sure).
    // NOTE: The background estimator should calculate rho_m by default, but it's not used by default
    //       in the standard subtractor, so we explicitly enable it below (CS is a whole separate topic)
    backgroundEstimator->set_compute_rho_m(true);

    // Setup the background estimator to be able to make the estimation.
    backgroundEstimator->set_particles(particlePseudoJets);

    // Specific setup for event-wise constituent subtraction
    if (constituentSubtraction) {
      constituentSubtractor = std::make_shared<fastjet::contrib::ConstituentSubtractor>();
      constituentSubtractor->set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR);
      constituentSubtractor->set_max_distance(constituentSubtraction->rMax);
      constituentSubtractor->set_alpha(constituentSubtraction->alpha);
      // ALICE doesn't appear to set this value, so we skip it here and use the default of 0.01
      //constituentSubtractor->set_ghost_area(areaSettings.ghostArea);
      // Since this is event wise, this should be the track eta, not the fiducial eta
      constituentSubtractor->set_max_eta(etaMax);
      constituentSubtractor->set_background_estimator(backgroundEstimator.get());
      // Use the same estimator for rho_m (by default, I think it won't be used, but better to
      // provide it in case we change our mind later).
      // NOTE: This needs to be set after setting the background estimator.
      constituentSubtractor->set_common_bge_for_rho_and_rhom();
      // Apparently this is new and now required for event-wise CS.
      // From some tests, constituent subtraction gives some crazy results if it's not called!
      // NOTE: ALICE gets away with skipping this because we have the old call for event-wise
      //       subtraction where we pass the max rapidity. But better if we use the newer version.
      constituentSubtractor->initialize();
    }
  }

  // Now, setup the subtractor object (when needed), which will subtract the background from jets.
  // NOTE: It's not used in the case of event-wise constituent subtraction, since the subtraction
  //       is applied separately.
  std::shared_ptr<fastjet::Subtractor> subtractor = nullptr;
  // Now, set it up as necessary.
  if (backgroundSubtraction) {
    subtractor = std::make_shared<fastjet::Subtractor>(backgroundEstimator.get());
    // Use rho_m from the estimator.
    subtractor->set_use_rho_m(true);
    // Handle negative masses by adjusting the 4-vector to maintain the pt and phi, which leaves
    // the rapidity the same as the unsubtracted jet. The fj manual describes this as "a sensible
    // behavior" for most applications, so good enough for us.
    subtractor->set_safe_mass(true);
  }

  // Signal jet settings
  // Again, these should all be settable, but I wanted to keep the signature simple, so I just define them here with some reasonable(ish) defaults.
  double jetPtMax = 1000;
  // Would often set as abs(eta - R), but should be configurable.
  double jetEtaMin = etaMin + jetR;
  double jetEtaMax = etaMax - jetR;
  //double jetPhiMin = 0;
  //double jetPhiMax = 2 * M_PI;
  // Fastjet settings
  // NOTE: Jet algorithm defined at the beginning
  // NOTE: Jet area type defined at the beginning
  fastjet::RecombinationScheme recombinationScheme(fastjet::RecombinationScheme::E_scheme);
  fastjet::Strategy strategy(fastjet::Strategy::Best);
  // Derived fastjet settings
  fastjet::JetDefinition jetDefinition(jetAlgorithm, jetR, recombinationScheme, strategy);
  fastjet::AreaDefinition areaDefinition(areaType, ghostAreaSpec);
  fastjet::Selector selectJets = !fastjet::SelectorIsPureGhost() * (fastjet::SelectorPtRange(minJetPt, jetPtMax) && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax)); //&& fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);

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
  // Apply the jet selector after all subtraction is completed.
  // NOTE: It's okay that we already applied the min jet pt cut above because any additional
  //       subtraction will just remove more jets (ie. the first cut is _less_ restrictive
  //       than the second)
  jets = selectJets(jets);

  // It's also not uncommon to apply a sorting by E or pt.
  jets = fastjet::sorted_by_pt(jets);

  // Now, handle returning the values.
  // First, we need to extract the constituents.
  //auto & [px, py, pz, E] = pseudoJetsToNumpy(jets);
  auto numpyJets = pseudoJetsToVectors<T>(jets);
  auto columnarJetsArea = extractJetsArea(jets);

  // Then, we convert the jets themselves into vectors to return.
  auto constituentIndices = constituentIndicesFromJets(jets);

  if (constituentSubtraction) {
    // NOTE: particlePseudoJets are actually the subtracted constituents now.
    return OutputWrapper<T>{
      numpyJets, constituentIndices, columnarJetsArea, std::make_tuple(pseudoJetsToVectors<T>(particlePseudoJets), subtractedToUnsubtractedIndices)
    };
  }
  return OutputWrapper<T>{numpyJets, constituentIndices, columnarJetsArea, {}};
}

template<typename T>
JetSubstructure::JetSubstructureSplittings jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  std::optional<AreaSettings> areaSettings,
  std::tuple<double, double> etaRange,
  bool storeRecursiveSplittings
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
    areaDefinition = std::make_unique<fastjet::AreaDefinition>(*areaType, *ghostSpec);
  }

  // Convert column vector input to pseudo jets.
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors);

  // If we use the area definition, we need to create a ClusterSequenceArea.
  // NOTE: The CS has to stay in scope while we explore the splitting history.
  std::unique_ptr<fastjet::ClusterSequence> cs = nullptr;
  if (areaDefinition) {
    cs = std::make_unique<fastjet::ClusterSequenceArea>(particlePseudoJets, jetDefinition, *areaDefinition);
  }
  else {
    cs = std::make_unique<fastjet::ClusterSequence>(particlePseudoJets, jetDefinition);
  }
  std::vector<fastjet::PseudoJet> outputJets = cs->inclusive_jets(0);

  fastjet::PseudoJet jj;
  jj = outputJets[0];

  // Store the jet splittings.
  JetSubstructure::JetSubstructureSplittings jetSplittings;
  int splittingNodeIndex = -1;
  ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

  return jetSplittings;
}

}
