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
#include <fastjet/tools/GridMedianBackgroundEstimator.hh>
#include <fastjet/tools/Subtractor.hh>
#include <fastjet/contrib/ConstituentSubtractor.hh>

namespace mammoth {
namespace JetSubstructure {
  class Subjets;
  class JetSplittings;
  class JetSubstructureSplittings;
}
}

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

/**
 * @brief Wrapper for momentum four vectors.
 *
 * Assume px, py, pz, E
 *
 * @tparam T Storage type for four vector - usually double or float.
 */
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
 * Just a simple container for area related settings. Create via brace initialization.
 */
struct AreaSettings {
  std::string areaTypeName;
  double ghostArea;
  double rapidityMax;
  int repeatN;
  double gridScatter;
  double ktScatter;
  double ktMean;
  std::vector<int> randomSeed;

  /**
   * @brief Create AreaType object according to the settings.
   *
   * @return fastjet::AreaType
   */
  fastjet::AreaType areaType() const { return areaTypes.at(this->areaTypeName); }

  /**
   * @brief Create GhostedAreaSpec object according to the settings.
   *
   * @return fastjet::GhostedAreaSpec
   */
  fastjet::GhostedAreaSpec ghostedAreaSpec() const;

  /**
   * @brief Create AreaDefinition object based on the settings.
   *
   * @return fastjet::AreaDefinition
   */
  fastjet::AreaDefinition areaDefinition() const;

  /**
   * Prints information about the settings.
   *
   * @return std::string containing information about the task.
   */
  std::string to_string() const;

  protected:
    static const std::map<std::string, fastjet::AreaType> areaTypes;
};

/**
 * @brief Jet finding settings
 *
 * Contains the essential jet finding settings. Also contains helpers to create the relevant fastjet
 * objects based on the provided settings. Create via brace initialization.
 */
struct JetFindingSettings {
  double R;
  std::string algorithmName;
  std::string recombinationSchemeName;
  std::string strategyName;
  std::tuple<double, double> ptRange;
  std::tuple<double, double> etaRange;
  const std::optional<const AreaSettings> areaSettings{std::nullopt};

  /**
   * @brief Helper to provide convenient access to the minimum jet pt.
   *
   * @return double The minimum jet pt
   */
  double minJetPt() const { return std::get<0>(this->ptRange); }

  /**
   * @brief Create jet algorithm from name stored in settings.
   *
   * @return fastjet::JetAlgorithm
   */
  fastjet::JetAlgorithm algorithm() const { return this->algorithms.at(this->algorithmName); }
  /**
   * @brief Create jet recombination scheme from name stored in the settings.
   *
   * @return fastjet::RecombinationScheme
   */
  fastjet::RecombinationScheme recombinationScheme() const { return this->recombinationSchemes.at(this->recombinationSchemeName); }
  /**
   * @brief Create jet clustering strategy from name stored in the settings.
   *
   * @return fastjet::Strategy
   */
  fastjet::Strategy strategy() const { return this->strategies.at(this->strategyName); }

  /**
   * @brief Create jet selector based on the stored properties.
   *
   * The name is specific because I don't want to get confused at what it does.
   * As a simple selector on pt and eta (and removing pure ghost jets), this should be
   * sufficient to at least get started (most of the time) for the main jet finder.
   *
   * @return fastjet::Selector Selector to be applied to the jets that are found by the CS.
   */
  fastjet::Selector selectorPtEtaNonGhost() const;

  /**
   * @brief Create jet definition based on the stored settings.
   *
   * @return fastjet::JetDefinition
   */
  fastjet::JetDefinition definition() const {
    return fastjet::JetDefinition(this->algorithm(), this->R, this->recombinationScheme(), this->strategy());
  }

  /**
   * @brief Create cluster sequence based on the stored settings.
   *
   * It will only create a ClusterSequenceArea if AreaSettings were provided.
   *
   * @param particlePseudoJets Particles to be clustered together.
   * @return std::unique_ptr<fastjet::ClusterSequence> The cluster sequence, ready to cluster
   *                                                   particles into jets.
   */
  std::unique_ptr<fastjet::ClusterSequence> create(std::vector<fastjet::PseudoJet> particlePseudoJets) const;

  /**
   * Prints information about the settings.
   *
   * @return std::string containing information about the task.
   */
  std::string to_string() const;

 protected:
  /// Map from name of jet algorithm to jet algorithm object.
  static const std::map<std::string, fastjet::JetAlgorithm> algorithms;
  /// Map from name of jet recombination scheme to jet recombination scheme object.
  static const std::map<std::string, fastjet::RecombinationScheme> recombinationSchemes;
  /// Map from name of jet clustering strategy to jet strategy object.
  static const std::map<std::string, fastjet::Strategy> strategies;
};

/**
 * @brief Abstract case class for background estimator
 *
 * Provides a simple interface for creating background estimators.
 */
struct BackgroundEstimator {
  /**
   * @brief Create the background estimator based on the stored settings.
   *
   * @return std::unique_ptr<fastjet::BackgroundEstimatorBase> The background estimator.
   */
  virtual std::unique_ptr<fastjet::BackgroundEstimatorBase> create() const = 0;

  /**
   * Prints information about the estimator.
   *
   * @return std::string containing information about the estimator.
   */
  virtual std::string to_string() const = 0;
};

/**
 * @brief Background estimator based on the median jet pt
 *
 * This is the standard method used by ALICE, etc.
 */
struct JetMedianBackgroundEstimator : BackgroundEstimator {
  JetFindingSettings settings;
  bool computeRhoM;
  bool useAreaFourVector;
  int excludeNHardestJets;
  double constituentPtMax;

  /**
   * @brief Construct a new Jet Median Background Estimator object.
   *
   * This is equivalent to the brace initialization that I usually use, but that doesn't work with pybind11
   * (even though the base class is abstract), so we have to write it by hand.
   *
   * @param _settings Jet finding settings
   * @param _computeRhoM If True, compute Rho_M
   * @param _useAreaFourVector If True, use area four vector
   * @param _excludeNHardestJets Number of hardest jets to exclude
   * @param _constituentPtMax Maximum constitunet pt to allow in a selected jet
   */
  JetMedianBackgroundEstimator(JetFindingSettings _settings, bool _computeRhoM, bool _useAreaFourVector, int _excludeNHardestJets, double _constituentPtMax):
      settings(_settings), computeRhoM(_computeRhoM), useAreaFourVector(_useAreaFourVector),
      excludeNHardestJets(_excludeNHardestJets), constituentPtMax(_constituentPtMax) {}

  /**
   * @brief Standard selector for the Jet Median Background Estimator
   *
   * In principle, this could vary. But in practice, this will usually be what we want to use.
   * It applies the following selections:
   *
   * - Removing jets with a constituent with pt greater than `constituentMaxPt`
   * - Selecting within the eta range
   * - Removing the N hardest jets, set by `excludeNHardestJets`
   *
   * @return fastjet::Selector
   */
  fastjet::Selector selector() const;

  /**
   * @brief Create the background estimator based on the stored settings.
   *
   * @return std::unique_ptr<fastjet::BackgroundEstimatorBase> The background estimator.
   */
  std::unique_ptr<fastjet::BackgroundEstimatorBase> create() const override;

  /**
   * Prints information about the estimator.
   *
   * @return std::string containing information about the estimator.
   */
  std::string to_string() const;
};

/**
 * @brief Background estimator based on values estimated on a grid.
 *
 * This is supposed to be much faster than the JetMedian approach, and is used by heppy et al,
 * but needs to be validated (optimize the parameters, as well as verify the actual implementation here).
 */
struct GridMedianBackgroundEstimator : BackgroundEstimator {
  double rapidityMax;
  double gridSpacing;

  /**
   * @brief Construct a new Grid Median Background Estimator object
   *
   * This is equivalent to the brace initialization that I usually use, but that doesn't work with pybind11
   * (even though the base class is abstract), so we have to write it by hand.
   *
   * @param _rapidityMax Max rapidity to consider
   * @param _gridSpacing Size of a grid cell
   */
  GridMedianBackgroundEstimator(double _rapidityMax, double _gridSpacing):
    rapidityMax(_rapidityMax), gridSpacing(_gridSpacing) {}

  /**
   * @brief Create the background estimator based on the stored settings.
   *
   * @return std::unique_ptr<fastjet::BackgroundEstimatorBase> The background estimator.
   */
  std::unique_ptr<fastjet::BackgroundEstimatorBase> create() const override {
    return std::make_unique<fastjet::GridMedianBackgroundEstimator>(this->rapidityMax, this->gridSpacing);
  }

  /**
   * Prints information about the estimator.
   *
   * @return std::string containing information about the estimator.
   */
  std::string to_string() const;
};

/**
 * @brief Background subtraction type / mode
 *
 * Controls the background subtraction that is utilized.
 */
enum class BackgroundSubtractionType {
  /// Disable background subtraction (also disables background estimation, since it's not needed in this case)
  disabled = 0,
  /// Standard rho subtraction
  rho= 1,
  /// Event-wise constituent subtraction
  eventWiseCS = 2,
  /// Jet-wise consituent subtraction (never tested as of Feb 2022, so it should be verified)
  jetWiseCS = 3,
};

/**
 * @brief Abstract case class for background subtractor
 *
 * Provides a simple interface for creating background subtractors.
 */
struct BackgroundSubtractor {
  /**
   * @brief Create the background subtractor based on the stored settings.
   *
   * @param backgroundEstimator Background estimator
   * @return std::unique_ptr<fastjet::Transformer> Subtractor based on the stored settings.
   */
  virtual std::unique_ptr<fastjet::Transformer> create(std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator) const = 0;

  /**
   * Prints information about the subtractor.
   *
   * @return std::string containing information about the subtractor.
   */
  virtual std::string to_string() const = 0;
};

/**
 * @brief Background subtraction using rho (with jet area)
 */
struct RhoSubtractor : BackgroundSubtractor {
  bool useRhoM;
  bool useSafeMass;

  /**
   * @brief Construct a new Rho Subtractor object
   *
   * This is equivalent to the brace initialization that I usually use, but that doesn't work with pybind11
   * (even though the base class is abstract), so we have to write it by hand.
   *
   * @param _useRhoM Use rho_m for subtraction
   * @param _useSafeMass Use safe mass during subtraction
   */
  RhoSubtractor(bool _useRhoM, bool _useSafeMass): useRhoM(_useRhoM), useSafeMass(_useSafeMass) {}

  /**
   * @brief Create the rho subtractor based on the stored settings.
   *
   * @param backgroundEstimator Background estimator
   * @return std::unique_ptr<fastjet::Transformer> Subtractor based on the stored settings.
   */
  std::unique_ptr<fastjet::Transformer> create(std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator) const override;

  /**
   * Prints information about the subtractor.
   *
   * @return std::string containing information about the subtractor.
   */
  std::string to_string() const;
};

/**
 * @brief Background subtraction using constituent subtraction
 */
struct ConstituentSubtractor : BackgroundSubtractor {
  double rMax;
  double alpha;
  double rapidityMax;
  std::string distanceMeasure;

  /**
   * @brief Construct a new Constituent Subtractor object
   *
   * This is equivalent to the brace initialization that I usually use, but that doesn't work with pybind11
   * (even though the base class is abstract), so we have to write it by hand.
   *
   * @param _rMax CS R_max parameter
   * @param _alpha CS alpha parameter
   * @param _rapidityMax CS rapidity (technically, eta here) max
   * @param _distanceMeasure CS distance measure
   */
  ConstituentSubtractor(double _rMax, double _alpha, double _rapidityMax, std::string _distanceMeasure):
    rMax(_rMax), alpha(_alpha), rapidityMax(_rapidityMax), distanceMeasure(_distanceMeasure) {}

  /**
   * @brief Create the constituent subtractor based on the stored settings.
   *
   * @param backgroundEstimator Background estimator
   * @return std::unique_ptr<fastjet::Transformer> Subtractor based on the stored settings.
   */
  std::unique_ptr<fastjet::Transformer> create(std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator) const override;

  /**
   * Prints information about the subtractor.
   *
   * @return std::string containing information about the subtractor.
   */
  std::string to_string() const;

 protected:
  /// Map from name of constituent subtractor distance measure to distance enumeration value.
  static const std::map<std::string, fastjet::contrib::ConstituentSubtractor::Distance> distanceTypes;
};

/**
 * @brief Main container for background subtraction settings.
 *
 * Used to keep track of all background subtraction settings in one place. Create via brace initialization.
 */
struct BackgroundSubtraction {
  BackgroundSubtractionType type;
  std::shared_ptr<BackgroundEstimator> estimator;
  std::shared_ptr<BackgroundSubtractor> subtractor;

  /**
   * Prints information about the background subtraction.
   *
   * @return std::string containing information about the background subtraction.
   */
  std::string to_string() const;
};

/**
 * @brief Output wrapper for the internal findJets implementation
 *
 */
struct FindJetsImplementationOutputWrapper {
  std::shared_ptr<fastjet::ClusterSequence> cs;
  std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator;
  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> particles;
  std::vector<unsigned int> subtractedToUnsubtractedIndices;

  /**
   * @brief Construct a new Find Jets Implementation Output Wrapper object
   *
   * Created an explicit constructor to ensure that we pass objects by reference when appropriate.
   *
   * @param _cs Cluster sequence
   * @param _backgroundEstimator Background estimator
   * @param _jets Jets found by cluster sequence
   * @param _particles Particles used for jet finding. May be just the input particles, or
   *                   those subtracted by event-wise constituent subtraction. Depends on the setttings.
   * @param _subtractedToUnsubtractedIndices Map from subtracted to unsubtracted indicies. Only propulated
   *                    if using event-wise constituent subtraction.
   */
  FindJetsImplementationOutputWrapper(std::shared_ptr<fastjet::ClusterSequence> _cs,
                                      std::shared_ptr<fastjet::BackgroundEstimatorBase> _backgroundEstimator,
                                      std::vector<fastjet::PseudoJet> & _jets,
                                      std::vector<fastjet::PseudoJet> & _particles,
                                      std::vector<unsigned int> & _subtractedToUnsubtractedIndices):
    cs(_cs), backgroundEstimator(_backgroundEstimator), jets(_jets), particles(_particles), subtractedToUnsubtractedIndices(_subtractedToUnsubtractedIndices) {}
};

/**
 * @brief Constituent subtraction settings
 *
 * Just a simple container for constituent subtraction related settings.
 *
 * TODO: Delete this once we've moved to the new interface.
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
 * @brief Generic function to handle the implementation of jet finding.
 *
 * @tparam T Input data type (usually float or double)
 * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param mainJetFinder Settings for jet finding, including R, algorithm, acceptance, area settings, etc
 * @param backgroundEstimatorFourVectors Four vectors to provide to the background estimator. If they're empty
 *                                       the column (ie. input) four vectors are used.
 * @param backgroundSubtraction Settings for background subtraction, including the option to specify the
 *                              background estimator and background subtractor.
 * @return FindJetsImplementationOutputWrapper Output from jet finding, including the CS, background estimator, jets, etc.
 */
template<typename T>
FindJetsImplementationOutputWrapper findJetsImplementation(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
);

/**
 * @brief Find jets based on the input particles and settings.
 *
 * @tparam T Input data type (usually float or double)
 * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param mainJetFinder Settings for jet finding, including R, algorithm, acceptance, area settings, etc
 * @param backgroundEstimatorFourVectors Four vectors to provide to the background estimator. If they're empty
 *                                       the column (ie. input) four vectors are used.
 * @param backgroundSubtraction Settings for background subtraction, including the option to specify the
 *                              background estimator and background subtractor.
 * @return OutputWrapper<T> Output from jet finding.
 */
template<typename T>
OutputWrapper<T> findJetsNew(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
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
 * @brief Main function for jet reclustering
 *
 * @tparam T Input data type (usually float or double)
 * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param mainJetFinder Settings for jet finding, including R, algorithm, acceptance, area settings, etc
 * @return JetSubstructure::JetSubstructureSplittings Jet substructure splittings container
 */
template<typename T>
JetSubstructure::JetSubstructureSplittings jetReclusteringNew(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder
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

// TODO: Remove this function once done with validation
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

// TODO: Remove this function once done with validation
fastjet::AreaType getAreaType(const AreaSettings & areaSettings)
{
  // Area type
  std::map<std::string, fastjet::AreaType> areaTypes = {
    {"active_area", fastjet::AreaType::active_area},
    {"active_area_explicit_ghosts", fastjet::AreaType::active_area_explicit_ghosts},
    {"passive_area", fastjet::AreaType::passive_area},
  };
  return areaTypes.at(areaSettings.areaTypeName);
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
FindJetsImplementationOutputWrapper findJetsImplementation(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
)
{
  // TODO: The seed needs to get moved out...
  // Needed for PbPb validation
  std::vector<int> fixedSeeds = {12345, 67890};

  // Determine if we're doing validation based on whether there is a fixed seed provided for the AreaSettings
  bool validationMode = false;
  if (mainJetFinder.areaSettings) {
    validationMode = (mainJetFinder.areaSettings->randomSeed.size() > 0);
  }

  // Convert column vector input to pseudo jets.
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors);

  // Notify about the settings for the jet finding.
  // NOTE: This can be removed eventually. For now (July 2021), it will be routed to debug level
  //       so we can be 100% sure about what is being calculated.
  std::cout << std::boolalpha
    << "Cuts:\n"
    //<< "\tMin jet pt=" << minJetPt << "\n"
    << "Settings:\n"
    << "\tValidation mode" << validationMode << "\n"
    //<< "\tGhost area: " << areaSettings.ghostArea << "\n"
    << "\tBackground estimator using " << (std::get<0>(backgroundEstimatorFourVectors).size() > 0 ? "background" : "input") << " particles\n"
    //<< "\tBackground subtraction: " << backgroundSubtraction << "\n"
    //<< "\tConstituent subtraction: " << static_cast<bool>(constituentSubtraction)
    << "\n";

  // First start with a background estimator, if we're running one.
  std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator;
  std::shared_ptr<fastjet::Transformer> subtractor;
  if (backgroundSubtraction.type != BackgroundSubtractionType::disabled) {
    // First, we need to create the background estimator
    if (!backgroundSubtraction.estimator) {
      throw std::runtime_error("Background estimator is required, but not defined. Please check settings!");
    }
    // All of the settings are specified in the background estimator
    backgroundEstimator = backgroundSubtraction.estimator->create();

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

    // We can't directly access the cluster sequence of the background estimator.
    // Consequently, to validate the seed, we create a separate cluster sequence based estimator
    // using the same settings, and then compare the rho values. They should agree.
    if (validationMode) {
      // This is specific to the JetMedianBackgroundEstimator, so we need to cast to it.
      std::shared_ptr<JetMedianBackgroundEstimator> jetMedianSettings = std::dynamic_pointer_cast<JetMedianBackgroundEstimator>(backgroundSubtraction.estimator);
      // Create the CS and convert to a CSA. We need to do this in two steps to avoid running
      // into issues due to returning a unique_ptr
      std::shared_ptr<fastjet::ClusterSequence> tempCSBkg = jetMedianSettings->settings.create(
        possibleBackgroundEstimatorParticles.size() > 0
        ? possibleBackgroundEstimatorParticles
        : particlePseudoJets
      );
      auto csBkg = std::dynamic_pointer_cast<fastjet::ClusterSequenceArea>(tempCSBkg);
      // Finally, create the JetMedianBackgroundEstimator separately
      fastjet::JetMedianBackgroundEstimator bgeWithExistingCS(jetMedianSettings->selector(), *csBkg);
      bgeWithExistingCS.set_compute_rho_m(jetMedianSettings->computeRhoM);
      bgeWithExistingCS.set_use_area_4vector(jetMedianSettings->useAreaFourVector);
      // And check the values
      assert(
        bgeWithExistingCS.rho() == backgroundEstimator->rho() &&
        ("estimator rho=" + std::to_string(backgroundEstimator->rho()) + ", validation rho=" + std::to_string(bgeWithExistingCS.rho())).c_str()
      );
      // TODO: Remove the printout...
      std::cerr << "rhoWithCS=" << bgeWithExistingCS.rho() << ", standard rho=" << backgroundEstimator->rho()  << "\n";
      // ENDTODO
    }

    // Next up, create the subtractor
    // All of the settings are specified in the background subtractor
    subtractor = backgroundSubtraction.subtractor->create(backgroundEstimator);
  }

  // For constituent subtraction, we perform event-wise subtraction on the input particles
  // We also keep track of a map from the subtracted constituents to the unsubtracted constituents
  // (both of which are based on the user_index that we assign during the jet finding).
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  if (backgroundSubtraction.type == BackgroundSubtractionType::eventWiseCS) {
    // Need to cast to CS object so we can actually do the event-wise subtraction
    auto constituentSubtractor = std::dynamic_pointer_cast<fastjet::contrib::ConstituentSubtractor>(subtractor);
    particlePseudoJets = constituentSubtractor->subtract_event(particlePseudoJets);
    subtractedToUnsubtractedIndices = updateSubtractedConstituentIndices(particlePseudoJets);
  }

  // Perform jet finding on signal
  std::shared_ptr<fastjet::ClusterSequence> cs = mainJetFinder.create(particlePseudoJets);
  auto jets = cs->inclusive_jets(mainJetFinder.minJetPt());

  // Validate that the seed is fixed
  // NOTE: We don't want to do this all of the time because the seed needs to be set very carefully,
  //       or it can lead to very confusing results. It's also a waste of cycles if not needed.
  if (validationMode) {
    std::vector<int> checkFixedSeed;
    // NOTE: Need to retrieve it from the CSA because we pass copies of objects, not by reference
    auto csa = std::dynamic_pointer_cast<fastjet::ClusterSequenceArea>(cs);
    csa->area_def().ghost_spec().get_last_seed(checkFixedSeed);
    if (checkFixedSeed != mainJetFinder.areaSettings->randomSeed) {
      std::string values = "";
      for (const auto & v : checkFixedSeed) {
        values += " " + v;
      }
      throw std::runtime_error("Seed mismatch in validation mode! Retrieved: " + values);
    }
    // TODO: Comment this out when done...
    std::cout << "Fixed seeds (main jet finding): ";
    for (auto & v : checkFixedSeed) {
      std::cout << " " << v;
    }
    std::cout << "\n";
    // ENDTEMP
  }

  // Apply the subtractor when appropriate
  if (backgroundSubtraction.type != BackgroundSubtractionType::eventWiseCS) {
    jets = (*subtractor)(jets);
  }

  // Apply the jet selector after all subtraction is completed.
  // NOTE: It's okay that we already applied the min jet pt cut when we take the inclusive_jets above
  //       because any additional subtraction will just remove more jets (ie. the first cut is _less_
  //       restrictive than the second)
  jets = mainJetFinder.selectorPtEtaNonGhost()(jets);

  // Sort by pt for convenience
  jets = fastjet::sorted_by_pt(jets);

  return FindJetsImplementationOutputWrapper{
    std::move(cs), backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices
  };
}

template<typename T>
OutputWrapper<T> findJetsNew(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
)
{
  // Use jet finding implementation to do most of the work
  auto && [cs, backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices] = findJetsImplementation(
    columnFourVectors, mainJetFinder, backgroundEstimatorFourVectors, backgroundSubtraction
  );

  // Now, handle returning the values.
  // First, we grab the jets themselves, converting the four vectors into column vector to return them.
  auto numpyJets = pseudoJetsToVectors<T>(jets);
  // Next, we grab whatever other properties we desire:
  // Jet area
  auto columnarJetsArea = extractJetsArea<T>(jets);
  // Next, grab event wide properties
  // Rho value (storing 0 if not available)
  T rhoValue = backgroundEstimator ? backgroundEstimator->rho() : 0;
  // Finally, we need to associate the constituents with the jets. To do so, we store one vector per jet,
  // with the vector containing the user_index assigned earlier in the jet finding process.
  auto constituentIndices = constituentIndicesFromJets(jets);

  if (backgroundSubtraction.type == BackgroundSubtractionType::eventWiseCS ||
      backgroundSubtraction.type == BackgroundSubtractionType::jetWiseCS) {
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
JetSubstructure::JetSubstructureSplittings jetReclusteringNew(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  bool storeRecursiveSplittings
)
{
  // Use jet finding implementation to do most of the work
  // We need to disable background subtraction, so create a simple container to disable it
  FourVectorTuple<T> backgroundEstimatorFourVectors = {{}, {}, {}, {}};
  BackgroundSubtraction backgroundSubtraction{BackgroundSubtractionType::disabled, nullptr, nullptr};
  auto && [cs, backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices] = findJetsImplementation(
    columnFourVectors, mainJetFinder, backgroundEstimatorFourVectors, backgroundSubtraction
  );

  // Now that we're done, just need to handle formatting the output
  // TODO: Remove this print out afeter validation...
  std::cerr << "output jets\n";
  for (auto & temp_j : jets){
    std::cerr << temp_j.pt() << "\n";
  }

  // Extract the reclustered jets
  fastjet::PseudoJet jj = jets.at(0);
  // And store the jet splittings.
  JetSubstructure::JetSubstructureSplittings jetSplittings;
  int splittingNodeIndex = -1;
  ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

  return jetSplittings;
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
  for (auto & temp_j : particlePseudoJets) {
    std::cerr << "constituent " << temp_j.user_index() << ", pt=" << temp_j.pt() << ", rapidity=" << temp_j.rapidity() << ", mass=" << temp_j.m() << "\n";
  }
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

  // TODO: Tie into main jet finding function...

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
  std::cerr << "output jets\n";
  for (auto & temp_j : outputJets){
    std::cerr << temp_j.pt() << "\n";
  }

  fastjet::PseudoJet jj;
  jj = outputJets[0];

  // Store the jet splittings.
  JetSubstructure::JetSubstructureSplittings jetSplittings;
  int splittingNodeIndex = -1;
  ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

  return jetSplittings;
}

}

std::ostream& operator<<(std::ostream& in, const mammoth::AreaSettings & c);
std::ostream& operator<<(std::ostream& in, const mammoth::JetFindingSettings & c);
std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const mammoth::JetMedianBackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const mammoth::GridMedianBackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtractionType& c);
std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtractor& c);
std::ostream& operator<<(std::ostream& in, const mammoth::RhoSubtractor& c);
std::ostream& operator<<(std::ostream& in, const mammoth::ConstituentSubtractor& c);
std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtraction& c);
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::Subjets& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSplittings& myTask);
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSubstructureSplittings& myTask);