#pragma once

#include <algorithm>
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

// Custom fastjet selectors
/**
 * @brief A selector requiring jets to have area greater than a minimum area
 *
 * @param areaMin Minimum area
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaMin(double areaMin);
/**
 * @brief A selector requiring jets to have area less than a maximum area
 *
 * @param areaMax Maximum area
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaMax(double areaMax);
/**
 * @brief A selector requiring jets to be within an area range
 *
 * @param areaMin Minimum area
 * @param areaMax Maximum area
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaRange(double areaMin, double areaMax);

/**
 * @brief A selector requiring jets to have area greater than a percentage of the jet parameter.
 *
 * @param jetParameter Jet R
 * @param percentageMin Minimum percentage (0-100%)
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaPercentageMin(double jetParameter, double percentageMin);
/**
 * @brief A selector requiring jets to have area less than a percentage of the jet parameter.
 *
 * @param jetParameter Jet R
 * @param percentageMax Maximum percentage (0-100%)
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaPercentageMax(double jetParameter, double percentageMax);
/**
 * @brief A selector requiring jets to have area within a percentage range of the jet parameter.
 *
 * @param jetParameter Jet R
 * @param percentageMin Minimum percentage (0-100%)
 * @param percentageMax Maximum percentage (0-100%)
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorAreaPercentageRange(double jetParameter, double percentageMin, double percentageMax);

/**
 * @brief Selector requiring that no constituent have a pt greater than or equal to a maximum
 *
 * @param constituentPtMax Maximum constituent pt
 * @return fastjet::Selector The selector
 */
fastjet::Selector SelectorConstituentPtMax(double constituentPtMax);


// Convenience
template<typename T>
using FourVectorTuple = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>;

/**
 * @brief Jet finding output wrapper
 *
 * @tparam T Output data type (usually float or double, set by the input type)
 */
template<typename T>
struct OutputWrapper {
  FourVectorTuple<T> jets;
  std::vector<std::vector<unsigned int>> constituent_indices;
  std::vector<T> jetsArea;
  T rho;
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
 * @return std::vector<T> Jet area for the given jets.
 */
template<typename T>
std::vector<T> extractJetsArea(
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
 * @return std::vector<unsigned int> Map indices of subtracted constituents to unsubtracted constituents.
 */
std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
);

/**
  * @brief Implementation of main jet finder.
  *
  * @tparam T Input data type (usually float or double)
  * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
  * @param jetR jet resolution parameter
  * @param jetAlgorithmStr jet algorithm
  * @param areaSettings Area settings
  * @param etaRange Eta range. Tuple of min and max. Default: (-0.9, 0.9)
  * @param minJetPt Minimum jet pt. Default: 1.
  * @param backgroundEstimatorFourVectors Four vectors to provide to the background estimator. If they're empty
  *                                       the column (ie. input) four vectors are used.
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
  FourVectorTuple<T> & backgroundEstimatorFourVectors = {{}, {}, {}, {}},
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
  ColumnarSubjets GetSubjets() { return ColumnarSubjets{fSplittingNodeIndex, fPartOfIterativeSplitting, fConstituentIndices}; }
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
 * Container for columnar jet splittings info. It's mainly for convenience in moving over to python.
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
  ColumnarSplittings GetSplittings() { return ColumnarSplittings{fKt, fDeltaR, fZ, fParentIndex}; }
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
  * @param jetAlgorithmStr jet algorithm. Default: "CA".
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
  const bool storeRecursiveSplittings = true
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

template<typename T>
std::vector<T> extractJetsArea(
  const std::vector<fastjet::PseudoJet> & jets
)
{
  std::size_t nJets = jets.size();
  std::vector<T> jetsArea(nJets);
  for (std::size_t i = 0; i < nJets; ++i) {
    // According to the fj manual, taking the transverse component of the area four vector
    // provides a more accurate determination of rho. So we take it here.
    jetsArea.at(i) = jets.at(i).area_4vector().pt();
  }
  return jetsArea;
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

// From: https://stackoverflow.com/a/39487448/12907985
template <typename T = double, typename C>
inline const T median(const C &the_container)
{
    std::vector<T> tmp_array(std::begin(the_container),
                             std::end(the_container));
    size_t n = tmp_array.size() / 2;
    std::nth_element(tmp_array.begin(), tmp_array.begin() + n, tmp_array.end());

    if(tmp_array.size() % 2){ return tmp_array[n]; }
    else
    {
        // even sized vector -> average the two middle values
        auto max_it = std::max_element(tmp_array.begin(), tmp_array.begin() + n);
        return (*max_it + tmp_array[n]) / 2.0;
    }
}

template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  AreaSettings areaSettings,
  std::tuple<double, double> etaRange,
  bool fiducialAcceptance,
  double minJetPt,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
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
    << "\tBackground estimator using " << (std::get<0>(backgroundEstimatorFourVectors).size() > 0 ? "background" : "input") << " particles\n"
    << "\tBackground subtraction: " << backgroundSubtraction << "\n"
    << "\tConstituent subtraction: " << static_cast<bool>(constituentSubtraction) << "\n";


  // TODO: Clean this up...
  // Needed for PbPb validation
  std::vector<int> fixedSeeds = {12345, 67890};
  //

  // General settings
  double etaMin = std::get<0>(etaRange);
  double etaMax = std::get<1>(etaRange);
  // Ghost settings
  // NOTE: ghost rapidity edges are expected to be symmetric
  // NOTE: The jets which are found seems to be super sensitive to this max rapidity.
  //       For now, we use the ALICE value for now.
  double ghostRapidityMax = 1.0;
  int ghostRepeatN = 1;
  double gridScatter = 1.0;
  double ktScatter = 0.1;
  double ghostktMean = 1e-100;
  fastjet::GhostedAreaSpec ghostAreaSpec(ghostRapidityMax, ghostRepeatN, areaSettings.ghostArea, gridScatter, ktScatter, ghostktMean);
  ghostAreaSpec.set_random_status(fixedSeeds);

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
    // NOTE: No restriction on phi, since we're focused on charged jets
    //double backgroundJetPhiMin = 0;
    //double backgroundJetPhiMax = 2 * M_PI;
    // Fastjet background settings
    fastjet::JetAlgorithm backgroundJetAlgorithm(fastjet::JetAlgorithm::kt_algorithm);
    fastjet::RecombinationScheme backgroundRecombinationScheme(fastjet::RecombinationScheme::E_scheme);
    fastjet::Strategy backgroundStrategy(fastjet::Strategy::Best);
    //std::vector<int> fixedSeeds = {1709161381, 1637757752};
    fastjet::GhostedAreaSpec backgroundGhostAreaSpec(ghostRapidityMax, ghostRepeatN, areaSettings.ghostArea, gridScatter, ktScatter, ghostktMean);
    backgroundGhostAreaSpec.set_random_status(fixedSeeds);
    // NOTE: Must include the explicit ghosts - otherwise, excluding the 2 hardest jets won't work!
    //       As described in footnote 27 in the fastjet 3.4 manual:
    //       "If you use non-geometric selectors such as [n hardest] in determining [rho], the area must
    //       have explicit ghosts in order to simplify the determination of the empty area. If it does
    //       not, an error will be thrown"
    fastjet::AreaType backgroundAreaType(fastjet::AreaType::active_area_explicit_ghosts);
    // Derived fastjet settings
    fastjet::JetDefinition backgroundJetDefinition(backgroundJetAlgorithm, backgroundJetR, backgroundRecombinationScheme, backgroundStrategy);
    fastjet::AreaDefinition backgroundAreaDefinition(backgroundAreaType, backgroundGhostAreaSpec);

    // Select jets for calculating the background
    // As of September 2021, this includes (ordered from top to bottom):
    // - Fiducial eta selection
    // - Remove the two hardest jets
    // - Remove pure ghost jets (since they are included with explicit ghosts)

    // NOTES:
    // - This is more or less the standard rho procedure
    // - We want to apply the two hardest removal _after_ the acceptance cuts, so we use "*"
    //   Be aware that the order of the selectors really matters. It applies the **right** most first if we use "*"
    //   This is super important! Otherwise, you'll get unexpected selections!
    // - Including or removing the two hardest can have a big impact on the number of jets that are accepted.
    //   Removing them includes more jets (because the median background is smaller). We remove them because
    //   that's what is done in ALICE.
    // - We skip phi selection since we're looking at charged jets, so we take the full [0, 2pi).
    //   [0, 2pi) is the standard PseudoJet phi range. If you want to restrict the phi range, it should be
    //   applied at the right most with
    //   `* fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax)`, or via the combined
    //   EtaPhi selector (see possible tweaks below).
    // - We don't remove jets with tracks > 100 GeV here because:
    //     - It is technically complicated with the current setup because I don't see any way to select
    //       constituents with a selector. It looks like I'd have to make an additional copy.
    //     - I think it will be a small effect on the background because we're concerned with the median
    //       and we exclude the two leading jets. So unless there are many fake tracks in a single event, it's
    //       unlikely to have a meaningful effect on the median.
    //
    // Some notes for possible tweaks (not saying that they necessarily should be done):
    // - `GridMedianBackgroundEstimator` may be usable here, and much faster. However, it needs to be validated.
    // - If one goes back to applying phi cuts, one could use `* fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax)`
    //   However, if using this combined selector, be careful about the difference between rapidity and eta!
    //fastjet::Selector selRho = !fastjet::SelectorNHardest(2) * !fastjet::SelectorIsPureGhost() * fastjet::SelectorAbsRapMax(ghostRapidityMax);
    //fastjet::Selector selRho = !fastjet::SelectorNHardest(2) * !fastjet::SelectorIsPureGhost() * fastjet::SelectorRapRange(backgroundJetEtaMin, backgroundJetEtaMax);
    //fastjet::Selector selRho = !fastjet::SelectorNHardest(2) * !fastjet::SelectorIsPureGhost() * fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax);
    fastjet::Selector selRho = !fastjet::SelectorNHardest(2)
                               //* !fastjet::SelectorIsPureGhost()  // NB: This selector doesn't make a difference...
                               * fastjet::SelectorEtaRange(backgroundJetEtaMin, backgroundJetEtaMax)
                               * SelectorConstituentPtMax(100);
    // TODO: Try estimating without area_4vector, try disabling rho_m, since it may be contributing...

    // Try to calculate rho by hand...
    fastjet::ClusterSequenceArea csBkg(particlePseudoJets,
                                       backgroundJetDefinition, backgroundAreaDefinition);

    // Try use JetMedianBackgroundEstimator w/ existing ClusterSequence
    fastjet::JetMedianBackgroundEstimator bgeWithExisting(selRho, csBkg);
    bgeWithExisting.set_compute_rho_m(false);
    bgeWithExisting.set_use_area_4vector(true);

    // For the calculation by hand
    auto backgroundJets = selRho(csBkg.inclusive_jets(0.));
    std::vector<T> rhoValues;
    for (auto & _j : backgroundJets) {
      rhoValues.push_back(_j.pt() / _j.area_4vector().perp());
    }
    T rhoValueByHand = median(rhoValues);

    std::cerr << "rhoByHand=" << rhoValueByHand << ", bgeWithExisting=" << bgeWithExisting.rho() << "\n";

    //std::vector<int> fixedSeeds;
    // NOTE: Need to retrieve it for the CSA because we pass copies of objects, not by reference
    //csBkg.area_def().ghost_spec().get_last_seed(fixedSeeds);

    std::cout << "Fixed seeds: ";
    for (auto & v : fixedSeeds) {
      std::cout << " " << v;
    }
    std::cout << "\n";

    // Finally, define the background estimator
    // This is needed for all background subtraction cases.
    backgroundEstimator = std::make_shared<fastjet::JetMedianBackgroundEstimator>(
      selRho, backgroundJetDefinition, backgroundAreaDefinition.with_fixed_seed(fixedSeeds)
    );
    // Ensure rho_m is calculated (should be by default, but just to be sure).
    // NOTE: The background estimator should calculate rho_m by default, but it's not used by default
    //       in the standard subtractor, so we explicitly enable it in the next block down
    //       (CS is handled separately)
    backgroundEstimator->set_compute_rho_m(true);

    // According to the fj manual, taking the transverse component of the area four vector
    // provides a more accurate determination of rho. So we take it here.
    // NOTE: ALICE also takes the transverse component.
    backgroundEstimator->set_use_area_4vector(true);

    // Setup the input particles for the estimator so we it calculate the background.
    // If we have background estimator four vectors, we need to make sure we use them here.
    // In the case that they weren't provided, the arrays are empty, so it doesn't really cost anything
    // to create a new (empty) vector. So we just do it regardless.
    auto possibleBackgroundEstimatorParticles = vectorsToPseudoJets(backgroundEstimatorFourVectors);
    // Then, we actually decide on what to pass depending on if there are passed background estimator particles or not.
    // NOTE: In principle, this would get us in trouble if the estimator is supposed to have no particles. But in that case,
    //       we would just turn off the background estimator. So it should be fine.
    backgroundEstimator->set_particles(possibleBackgroundEstimatorParticles.size() > 0
                                       ? possibleBackgroundEstimatorParticles
                                       : particlePseudoJets);

    // Specific setup for event-wise constituent subtraction
    if (constituentSubtraction) {
      constituentSubtractor = std::make_shared<fastjet::contrib::ConstituentSubtractor>();
      constituentSubtractor->set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR);
      constituentSubtractor->set_max_distance(constituentSubtraction->rMax);
      constituentSubtractor->set_alpha(constituentSubtraction->alpha);
      // ALICE doesn't appear to set the ghost area, so we skip it here and use the default of 0.01
      //constituentSubtractor->set_ghost_area(areaSettings.ghostArea);
      // NOTE: Since this is event wise, the max eta should be the track eta, not the fiducial eta
      //constituentSubtractor->set_max_eta(etaMax);
      constituentSubtractor->set_max_eta(ghostRapidityMax);
      constituentSubtractor->set_background_estimator(backgroundEstimator.get());
      // Use the same estimator for rho_m (by default, I think it won't be used in CS, but better
      // to provide it in case we change our mind later).
      // NOTE: This needs to be set after setting the background estimator.
      constituentSubtractor->set_common_bge_for_rho_and_rhom();
      // Apparently this is new and now required for event-wise CS.
      // From some tests, constituent subtraction gives some crazy results if it's not called!
      // NOTE: ALICE gets away with skipping this because we have the old call for event-wise
      //       subtraction where we pass the max rapidity. But better if we use the newer version here.
      constituentSubtractor->initialize();
    }
    //std::cerr << "rho=" << backgroundEstimator->rho() << ", rho_m=" << backgroundEstimator->rho_m() << "\n";
    std::cerr << "rho=" << backgroundEstimator->rho() << "\n";
  }

  // Now, setup the subtractor object (when needed), which will subtract the background from jets.
  // NOTE: It's not used in the case of event-wise constituent subtraction, since the subtraction
  //       is applied separately.
  std::shared_ptr<fastjet::Subtractor> subtractor = nullptr;
  // Now, set it up as necessary.
  if (backgroundSubtraction) {
    subtractor = std::make_shared<fastjet::Subtractor>(backgroundEstimator.get());
    // Use rho_m from the estimator.
    // NOTE: This causes the subtractor to subtract rhom in the four vector subtraction.
    //       This is done to account for nonzero hadron mass
    subtractor->set_use_rho_m(true);
    // Handle negative masses by adjusting the 4-vector to maintain the pt and phi, which leaves
    // the rapidity the same as the unsubtracted jet. The fj manual describes this as "a sensible
    // behavior" for most applications, so good enough for us.
    subtractor->set_safe_mass(true);
  }

  // Signal jet settings
  // Again, these should all be settable, but I wanted to keep the signature simple, so I just define them here with some reasonable(ish) defaults.
  double jetPtMax = 1000;
  // Jet acceptance
  double jetEtaMin = etaMin;
  double jetEtaMax = etaMax;
  // Allow to select jets in fiducial acceptance (ie . abs(eta - R))
  if (fiducialAcceptance) {
    jetEtaMin = etaMin + jetR;
    jetEtaMax = etaMax - jetR;
  }
  // Since we're using charged jets over the full acceptance, we don't both with setting the phi range.
  // NOTE: If we wanted to, we would use combine it with the pt and eta selector below using `&&`.
  //       eg. `&& fastjet::SelectorPhiRange(jetPhiMin, jetPhiMax);` One could also use the combined RapPhi selector
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
  fastjet::Selector selectJets = !fastjet::SelectorIsPureGhost()
                                 * (
                                   fastjet::SelectorPtRange(minJetPt, jetPtMax)
                                   && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax)
                                  );

  // For constituent subtraction, we perform event-wise subtraction on the input particles
  // We also keep track of a map from the subtracted constituents to the unsubtracted constituents
  // (both of which are based on the user_index that we assign during the jet finding).
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  if (constituentSubtractor) {
    particlePseudoJets = constituentSubtractor->subtract_event(particlePseudoJets);
    subtractedToUnsubtractedIndices = updateSubtractedConstituentIndices(particlePseudoJets);
  }

  // Perform jet finding on signal
  fastjet::ClusterSequenceArea cs(particlePseudoJets, jetDefinition, areaDefinition.with_fixed_seed(fixedSeeds));
  auto jets = cs.inclusive_jets(minJetPt);

  std::vector<int> fixedSeed;
  // NOTE: Need to retrieve it for the CSA because we pass copies of objects, not by reference
  cs.area_def().ghost_spec().get_last_seed(fixedSeed);
  std::cout << "Fixed seeds (main jet finding): ";
  for (auto & v : fixedSeed) {
    std::cout << " " << v;
  }
  std::cout << "\n";

  // Subtract scalar jet pt
  T rhoValue = backgroundEstimator ? backgroundEstimator->rho() : 0;
  //for (std::size_t i = 0; i < jets.size(); i++) {
  //  double ptToSubtract = rhoValue * jets.at(i).area_4vector().perp();
  //  fastjet::PseudoJet rhoToSubtract(ptToSubtract * std::cos(jets.at(i).phi()),
  //                                   ptToSubtract * std::sin(jets.at(i).phi()),
  //                                   0, 0);
  //  std::cerr << ", jetPtUnsub=" << jets.at(i).pt()
  //            << ", area=" << jets.at(i).area_4vector().perp()
  //            << ", phi=" << jets.at(i).phi()
  //            << ", ptToSubtract=" << ptToSubtract
  //            << ", rhoToSubtract.pt()=" << rhoToSubtract.pt()
  //            << "\n";
  //  fastjet::PseudoJet result = jets.at(i) - rhoToSubtract;
  //  //std::cerr << "after subtraction=" << result.pt() << "\n";
  //  // Using four vectors...
  //  fastjet::PseudoJet toSubtract = rhoValue * jets.at(i).area_4vector();

  //  std::cerr << "after subtraction by hand=" << result.pt() << ", with 4 vector=" << (jets.at(i) - toSubtract).pt() << "\n";
  //}
  // Apply the subtractor when appropriate
  if (backgroundSubtraction) {
    jets = (*subtractor)(jets);
  }

  // Apply the jet selector after all subtraction is completed.
  // NOTE: It's okay that we already applied the min jet pt cut when we take the inclusive_jets above
  //       because any additional subtraction will just remove more jets (ie. the first cut is _less_
  //       restrictive than the second)
  jets = selectJets(jets);

  // Sort by pt for convenience
  jets = fastjet::sorted_by_pt(jets);

  // Now, handle returning the values.
  // First, we grab the jets themselves, converting the four vectors into column vector to return them.
  auto numpyJets = pseudoJetsToVectors<T>(jets);
  // Next, we grab whatever other properties we desire
  auto columnarJetsArea = extractJetsArea<T>(jets);
  // Finally, we need to associate the constituents with the jets. To do so, we store one vector per jet,
  // with the vector containing the user_index assigned earlier in the jet finding process.
  auto constituentIndices = constituentIndicesFromJets(jets);
  // TODO: Moved up
  //T rhoValue = backgroundEstimator ? backgroundEstimator->rho() : 0;

  if (constituentSubtraction) {
    // NOTE: particlePseudoJets are actually the subtracted constituents now.
    return OutputWrapper<T>{
      numpyJets, constituentIndices, columnarJetsArea, rhoValue, std::make_tuple(
        pseudoJetsToVectors<T>(particlePseudoJets), subtractedToUnsubtractedIndices
      )
    };
  }
  return OutputWrapper<T>{numpyJets, constituentIndices, columnarJetsArea, rhoValue, {}};
}

template<typename T>
JetSubstructure::JetSubstructureSplittings jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::string jetAlgorithmStr,
  std::optional<AreaSettings> areaSettings,
  std::tuple<double, double> etaRange,
  const bool storeRecursiveSplittings
)
{
  // Jet algorithm
  fastjet::JetAlgorithm jetAlgorithm = getJetAlgorithm(jetAlgorithmStr);
  fastjet::RecombinationScheme recombinationScheme(fastjet::RecombinationScheme::E_scheme);
  fastjet::Strategy strategy(fastjet::Strategy::BestFJ30);
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
  //for (auto & temp_j : particlePseudoJets) {
  //  std::cerr << "constituent " << temp_j.user_index() << ", pt=" << temp_j.pt() << ", rapidity=" << temp_j.rapidity() << ", mass=" << temp_j.m() << "\n";
  //}
  //std::cerr << "delta_R\n";
  //for (auto & temp_i : particlePseudoJets) {
  //  for (auto & temp_j: particlePseudoJets) {
  //    if (temp_i.user_index() >= temp_j.user_index()) {
  //      continue;
  //    }
  //    std::cerr << "(" << temp_i.user_index() << ", " << temp_j.user_index() << "): " << temp_i.delta_R(temp_j) << "\n";
  //  }
  //}
  //std::cerr << "phi\n";
  //for (auto & temp_i : particlePseudoJets) {
  //  for (auto & temp_j: particlePseudoJets) {
  //    if (temp_i.user_index() >= temp_j.user_index()) {
  //      continue;
  //    }
  //    std::cerr << "(" << temp_i.user_index() << ", " << temp_j.user_index() << "): " << temp_i.delta_phi_to(temp_j) << "\n";
  //  }
  //}
  //std::cerr << "eta\n";
  //for (auto & temp_i : particlePseudoJets) {
  //  for (auto & temp_j: particlePseudoJets) {
  //    if (temp_i.user_index() >= temp_j.user_index()) {
  //      continue;
  //    }
  //    std::cerr << "(" << temp_i.user_index() << ", " << temp_j.user_index() << "): " << temp_i.eta() - temp_j.eta() << "\n";
  //  }
  //}
  //std::cerr << "y\n";
  //for (auto & temp_i : particlePseudoJets) {
  //  for (auto & temp_j: particlePseudoJets) {
  //    if (temp_i.user_index() >= temp_j.user_index()) {
  //      continue;
  //    }
  //    std::cerr << "(" << temp_i.user_index() << ", " << temp_j.user_index() << "): " << temp_i.rap() - temp_j.rap() << "\n";
  //  }
  //}

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
  //std::cerr << "output jets\n";
  //for (auto & temp_j : outputJets){
  //  std::cerr << temp_j.pt() << "\n";
  //}

  fastjet::PseudoJet jj;
  jj = outputJets[0];

  // Store the jet splittings.
  JetSubstructure::JetSubstructureSplittings jetSplittings;
  int splittingNodeIndex = -1;
  ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

  return jetSplittings;
}

}
