#pragma once

#include <algorithm>
#include <memory>
#include <optional>
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
  std::vector<std::vector<int>> constituents_user_index;
  std::vector<T> jetsArea;
  T rho;
  std::optional<std::tuple<FourVectorTuple<T>, std::vector<int>>> subtracted;
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
 * @brief Abstract case class for jet recombiner
 *
 * Provides a simple interface for creating recombiner classes.
 */
struct Recombiner {
  /**
   * @brief Create the recombiner on the stored settings.
   *
   * Note:
   *    We can't use smart pointers here because they'll go out of scope here and deallocate when we're still
   *    using the recombiner. Thus, we make the jet definition responsible for deallocating the memory of the recombiner
   *
   * @return fastjet::JetDefinition::Recombiner* The jet recombiner.
   */
  virtual fastjet::JetDefinition::Recombiner* create() const = 0;

  /**
   * Prints information about the recombiner.
   *
   * @return std::string containing information about the recombiner.
   */
  virtual std::string to_string() const = 0;

  /**
    * virtual destructor needs to be explicitly defined to avoid compiler warnings.
    */
  virtual ~Recombiner() = default;
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
  std::tuple<double, double> ptRange;
  std::tuple<double, double> etaRange;
  // NOTE: It would be more natural for the recombination scheme and strategy to be listed with the algorithm.
  //       However, we will set default values for the reco scheme and strategy, but not the pt and eta ranges,
  //       So to enable brace initialization, I reordered it. I could have also written a constructor, but this
  //       is fine and has no impact beyond the order in which I bind the arguments.
  std::string recombinationSchemeName;
  std::string strategyName;
  const std::optional<const AreaSettings> areaSettings{std::nullopt};
  // NOTE: We create the recombiner as a shared_ptr rather than an optional because the Recombiner is an abstract class
  //       (which allows us to create derived types for different recombiners). We allow it to default to a nullptr
  //       because if we don't explicitly set it, the default one will automatically be created, so we don't have to worry
  //       about the details in that case.
  const std::shared_ptr<const Recombiner> recombiner{nullptr};
  const std::optional<double> additionalAlgorithmParameter{std::nullopt};

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
    fastjet::JetDefinition jetDefinition = this->createJetDefinition();
    if (this->recombiner) {
      jetDefinition.set_recombiner(this->recombiner->create());
      // We can't use smart pointers here because they'll go out of scope here and deallocate when we're still
      // using the recombiner. Thus, we make the jet definition responsible for deallocating the memory of the recombiner
      jetDefinition.delete_recombiner_when_unused();
    }
    return jetDefinition;
  }

  /**
   * @brief Create cluster sequence based on the stored settings.
   *
   * It will only create a ClusterSequenceArea if AreaSettings were provided.
   * For some reason, casting the returned object to ClusterSequenceArea will fail
   * unless this method is defined in the header. Perhaps it loses visibility into the
   * types somehow?
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
  /// @brief Create the jet definition. Added additional indirection to ease creation of object.
  /// @return JetDefinition
  fastjet::JetDefinition createJetDefinition() const {
    if (this->additionalAlgorithmParameter) {
      return fastjet::JetDefinition(
        this->algorithm(),
        this->R,
        *this->additionalAlgorithmParameter,
        this->recombinationScheme(),
        this->strategy()
      );
    }
    return fastjet::JetDefinition(
      this->algorithm(),
      this->R,
      this->recombinationScheme(),
      this->strategy()
    );
  }

  /// Map from name of jet algorithm to jet algorithm object.
  static const std::map<std::string, fastjet::JetAlgorithm> algorithms;
  /// Map from name of jet recombination scheme to jet recombination scheme object.
  static const std::map<std::string, fastjet::RecombinationScheme> recombinationSchemes;
  /// Map from name of jet clustering strategy to jet strategy object.
  static const std::map<std::string, fastjet::Strategy> strategies;
};

// As noted above, this _must_ be defined in the header, or we will lose the ability
// to dynamic_cast into ClusterSequenceArea. I don't understand why, but won't overthink it
// NOTE: Needs to be inline to avoid being doubly defined. See: https://stackoverflow.com/a/3319310/12907985
inline std::unique_ptr<fastjet::ClusterSequence> JetFindingSettings::create(std::vector<fastjet::PseudoJet> particlePseudoJets) const {
  if (this->areaSettings) {
    return std::make_unique<fastjet::ClusterSequenceArea>(
      particlePseudoJets,
      this->definition(),
      this->areaSettings->areaDefinition()
    );
  }
  return std::make_unique<fastjet::ClusterSequence>(
    particlePseudoJets,
    this->definition()
  );
}

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

  /**
    * virtual destructor needs to be explicitly defined to avoid compiler warnings.
    */
  virtual ~BackgroundEstimator() = default;
};

/**
 * @brief Background estimator based on the median jet pt
 *
 * This is the standard method used by ALICE, etc.
 */
struct JetMedianBackgroundEstimator : public BackgroundEstimator {
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
   * @param _constituentPtMax Maximum constituent pt to allow in a selected jet
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
  std::string to_string() const override;
};

/**
 * @brief Background estimator based on values estimated on a grid.
 *
 * This is supposed to be much faster than the JetMedian approach, and is used by heppy et al,
 * but needs to be validated (optimize the parameters, as well as verify the actual implementation here).
 */
struct GridMedianBackgroundEstimator : public BackgroundEstimator {
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
  std::string to_string() const override;
};

/**
 * @brief Background subtraction type / mode
 *
 * Controls the background subtraction that is utilized.
 */
enum class BackgroundSubtraction_t {
  /// Disable background subtraction (also disables background estimation, since it's not needed in this case)
  disabled = 0,
  /// Standard rho subtraction
  rho= 1,
  /// Event-wise constituent subtraction
  eventWiseCS = 2,
  /// Jet-wise constituent subtraction (never tested as of Feb 2022, so it should be verified)
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

  /**
    * virtual destructor needs to be explicitly defined to avoid compiler warnings.
    */
  virtual ~BackgroundSubtractor() = default;
};

/**
 * @brief Background subtraction using rho (with jet area)
 */
struct RhoSubtractor : public BackgroundSubtractor {
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
  std::string to_string() const override;
};

/**
 * @brief Background subtraction using constituent subtraction
 */
struct ConstituentSubtractor : public BackgroundSubtractor {
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
  std::string to_string() const override;

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
  BackgroundSubtraction_t type;
  std::shared_ptr<BackgroundEstimator> estimator;
  std::shared_ptr<BackgroundSubtractor> subtractor;

  BackgroundSubtraction(BackgroundSubtraction_t _type, std::shared_ptr<BackgroundEstimator> _estimator, std::shared_ptr<BackgroundSubtractor> _subtractor):
    type{_type}, estimator{_estimator}, subtractor{_subtractor} {}

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
  std::vector<int> subtractedToUnsubtractedIndices;

  /**
   * @brief Construct a new Find Jets Implementation Output Wrapper object
   *
   * Created an explicit constructor to ensure that we pass objects by reference when appropriate.
   *
   * @param _cs Cluster sequence
   * @param _backgroundEstimator Background estimator
   * @param _jets Jets found by cluster sequence
   * @param _particles Particles used for jet finding. May be just the input particles, or
   *                   those subtracted by event-wise constituent subtraction. Depends on the settings.
   * @param _subtractedToUnsubtractedIndices Map from subtracted to unsubtracted indices. Only populated
   *                    if using event-wise constituent subtraction.
   */
  FindJetsImplementationOutputWrapper(std::shared_ptr<fastjet::ClusterSequence> _cs,
                                      std::shared_ptr<fastjet::BackgroundEstimatorBase> _backgroundEstimator,
                                      std::vector<fastjet::PseudoJet> & _jets,
                                      std::vector<fastjet::PseudoJet> & _particles,
                                      std::vector<int> & _subtractedToUnsubtractedIndices):
    cs(_cs), backgroundEstimator(_backgroundEstimator), jets(_jets), particles(_particles), subtractedToUnsubtractedIndices(_subtractedToUnsubtractedIndices) {}
};

/**
 * @brief Convert column vectors to a vector of PseudoJets
 *
 * @tparam T Input data type (usually float or double)
 * @param fourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param userIndices User indices to include along with the four vectors. Optional.
 * @return std::vector<fastjet::PseudoJet> Vector of PseudoJets containing the same information.
 */
template<typename T>
std::vector<fastjet::PseudoJet> vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors,
    const std::vector<int> & userIndices
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
 * @param jetFindingSettings Jet finding settings
 * @return std::vector<T> Jet area for the given jets.
 */
template<typename T>
std::vector<T> extractJetsArea(
  const std::vector<fastjet::PseudoJet> & jets,
  const JetFindingSettings & jetFindingSettings
);

/**
 * @brief Extract constituent indices from jets.
 *
 * @param jets Jets with constituents.
 * @return std::vector<std::vector<int>> The indices of all constituents in all jets.
 */
std::vector<std::vector<int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
);

/**
 * @brief Update the indices in subtracted constituents.
 *
 * Updating this indexing ensures that we can keep track of everything.
 *
 * @param pseudoJets Subtracted input particles.
 * @return std::vector<int> Map indices of subtracted constituents to unsubtracted constituents.
 */
std::vector<int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
);

/**
 * @brief Generic function to handle the implementation of jet finding.
 *
 * @tparam T Input data type (usually float or double)
 * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param columnFourVectorsUserIndices Containing user provided user indices. If empty, we will generate them ourselves.
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
  std::vector<int> & columnFourVectorUserIndices,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
);

/**
 * @brief Find jets based on the input particles and settings.
 *
 * @tparam T Input data type (usually float or double)
 * @param columnFourVectors Column four vectors, with the columns ordered ["px", "py", "pz", "E"]
 * @param userIndices User provided user indices for identifying particles in fastjet. If provided an
 *                    empty vector, we will generate indices automatically.
 * @param mainJetFinder Settings for jet finding, including R, algorithm, acceptance, area settings, etc
 * @param backgroundEstimatorFourVectors Four vectors to provide to the background estimator. If they're empty
 *                                       the column (ie. input) four vectors are used.
 * @param backgroundSubtraction Settings for background subtraction, including the option to specify the
 *                              background estimator and background subtractor.
 * @return OutputWrapper<T> Output from jet finding.
 */
template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  std::vector<int> & columnUserIndices,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
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
  std::string to_string() const;
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
  std::vector<float> tau;            ///<  Formation time between the subjets.
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
  void AddSplitting(float kt, float deltaR, float z, float tau, short parentIndex);
  std::tuple<float, float, float, float, short> GetSplitting(int i) const;
  unsigned int GetNumberOfSplittings() const { return fKt.size(); }
  ColumnarSplittings GetSplittings() { return ColumnarSplittings{fKt, fDeltaR, fZ, fTau, fParentIndex}; }
  //std::tuple<std::vector<float> &, std::vector<float> &, std::vector<float> &, std::vector<float> &, std::vector<short> &> GetSplittings() { return {fKt, fDeltaR, fZ, fTau, fParentIndex}; }

  // Printing
  std::string to_string() const;
  std::ostream & Print(std::ostream &in) const;

 protected:
  std::vector<float> fKt;             ///<  kT between the subjets.
  std::vector<float> fDeltaR;         ///<  Delta R between the subjets.
  std::vector<float> fZ;              ///<  Momentum sharing of the splitting.
  std::vector<float> fTau;            ///<  Formation time between the subjets.
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
  void AddSplitting(float kt, float deltaR, float z, float tau, short parentIndex);
  void AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
          const std::vector<unsigned short>& constituentIndices);
  // Getters
  std::tuple<float, float, float, float, short> GetSplitting(int i) const;
  std::tuple<unsigned short, bool, const std::vector<unsigned short>> GetSubjet(int i) const;
  unsigned int GetNumberOfSplittings() { return fJetSplittings.GetNumberOfSplittings(); }
  JetSubstructure::JetSplittings & GetSplittings() { return fJetSplittings; }
  JetSubstructure::Subjets & GetSubjets() { return fSubjets; }

  // Printing
  std::string to_string() const;
  std::ostream & Print(std::ostream &in) const;

 private:
  // Jet properties
  JetSubstructure::JetSplittings fJetSplittings;         ///<  Jet splittings.
  JetSubstructure::Subjets fSubjets;                     ///<  Subjets within the jet.
};

std::ostream& operator<<(std::ostream& in, const Subjets& myTask);
std::ostream& operator<<(std::ostream& in, const JetSplittings& myTask);
std::ostream& operator<<(std::ostream& in, const JetSubstructureSplittings& myTask);

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
 * @param storeRecursiveSplittings If True, store recursive splittings
 * @return JetSubstructure::JetSubstructureSplittings Jet substructure splittings container
 */
template<typename T>
JetSubstructure::JetSubstructureSplittings jetReclustering(
  FourVectorTuple<T> & columnFourVectors,
  const JetFindingSettings & mainJetFinder,
  const bool storeRecursiveSplittings
);


/************************************************
  * Implementations for templated functionality *
  ***********************************************/

template<typename T>
std::vector<fastjet::PseudoJet> vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors,
    const std::vector<int> & userIndices
)
{
    // Setup
    std::vector<fastjet::PseudoJet> particles;
    const auto & [px, py, pz, E] = fourVectors;

    // Validation for user index
    // Use px as proxy, since the size will be the same for all fields
    bool providedUserIndices = (px.size() == userIndices.size());

    // Convert
    for (std::size_t i = 0; i < px.size(); ++i) {
        particles.emplace_back(fastjet::PseudoJet(px[i], py[i], pz[i], E[i]));
        particles.back().set_user_index(providedUserIndices ? userIndices[i] : i);
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
  const std::vector<fastjet::PseudoJet> & jets,
  const JetFindingSettings & jetFindingSettings
)
{
  // Validation
  // Avoid requesting area properties if we haven't defined an area
  if (!jetFindingSettings.areaSettings) {
    return {};
  }

  std::size_t nJets = jets.size();
  std::vector<T> jetsArea(nJets);
  for (std::size_t i = 0; i < nJets; ++i) {
    // According to the fj manual, taking the transverse component of the area four vector
    // provides a more accurate determination of rho. So we take it here.
    jetsArea.at(i) = jets.at(i).area_4vector().pt();
  }
  return jetsArea;
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
  std::vector<int> & columnUserIndices,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
)
{
  // Determine if we're doing validation based on whether there is a fixed seed provided for the AreaSettings
  bool validationMode = false;
  if (mainJetFinder.areaSettings) {
    validationMode = (mainJetFinder.areaSettings->randomSeed.size() > 0);
  }

  // Convert column vector input to pseudo jets.
  auto particlePseudoJets = vectorsToPseudoJets(columnFourVectors, columnUserIndices);

  // Notify about the settings for the jet finding.
  // NOTE: This can be removed eventually. For now (July 2021), it will be routed to debug level
  //       so we can be 100% sure about what is being calculated.
  std::cout << std::boolalpha
    << "Settings:\n"
    << "\tValidation mode=" << validationMode << "\n"
    // For whatever reason, the mainJetFinder streamer doesn't seem to work here. I'm sure I'm doing
    // something dumb, but it's easily worked around via to_string, so just let it go.
    << "\tMain jet finding settings: " << mainJetFinder.to_string() << "\n"
    << "\tBackground estimator using " << (std::get<0>(backgroundEstimatorFourVectors).size() > 0 ? "background" : "input") << " particles\n"
    << "\tBackground subtraction: " << backgroundSubtraction
    << "\n";

  // First start with a background estimator, if we're running one.
  std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator;
  std::shared_ptr<fastjet::Transformer> subtractor;
  if (backgroundSubtraction.type != BackgroundSubtraction_t::disabled) {
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
    auto possibleBackgroundEstimatorParticles = vectorsToPseudoJets(backgroundEstimatorFourVectors, {});
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
      // This should always work, but double check for safety
      if (!jetMedianSettings) {
        throw std::runtime_error("Failed to retrieve jet median settings! What?");
      }
      // Create the CS and convert to a CSA. We need to do this in two steps to avoid running
      // into issues due to returning a unique_ptr
      std::shared_ptr<fastjet::ClusterSequence> tempCSBkg = jetMedianSettings->settings.create(
        possibleBackgroundEstimatorParticles.size() > 0
        ? possibleBackgroundEstimatorParticles
        : particlePseudoJets
      );
      auto csBkg = std::dynamic_pointer_cast<fastjet::ClusterSequenceArea>(tempCSBkg);
      // This should always work, but double check for safety
      if (!csBkg) {
        throw std::runtime_error("Failed to cast to ClusterSequenceArea for background subtraction! What?");
      }
      // Finally, create the JetMedianBackgroundEstimator separately
      fastjet::JetMedianBackgroundEstimator bgeWithExistingCS(jetMedianSettings->selector(), *csBkg);
      bgeWithExistingCS.set_compute_rho_m(jetMedianSettings->computeRhoM);
      bgeWithExistingCS.set_use_area_4vector(jetMedianSettings->useAreaFourVector);
      // And check the values
      assert(
        bgeWithExistingCS.rho() == backgroundEstimator->rho() &&
        ("estimator rho=" + std::to_string(backgroundEstimator->rho()) + ", validation rho=" + std::to_string(bgeWithExistingCS.rho())).c_str()
      );
      // NOTE: This is usually too noisy, but if visual confirmation is needed beyond the check above, uncomment the line below.
      //std::cout << "rhoWithClusterSequence=" << bgeWithExistingCS.rho() << ", rhoStandard=" << backgroundEstimator->rho()  << "\n";
    }

    // Next up, create the subtractor
    // All of the settings are specified in the background subtractor
    subtractor = backgroundSubtraction.subtractor->create(backgroundEstimator);
  }

  // For constituent subtraction, we perform event-wise subtraction on the input particles
  // We also keep track of a map from the subtracted constituents to the unsubtracted constituents
  // (both of which are based on the user_index that we assign during the jet finding).
  std::vector<int> subtractedToUnsubtractedIndices;
  if (backgroundSubtraction.type == BackgroundSubtraction_t::eventWiseCS) {
    // Need to cast to CS object so we can actually do the event-wise subtraction
    auto constituentSubtractor = std::dynamic_pointer_cast<fastjet::contrib::ConstituentSubtractor>(subtractor);
    if (!constituentSubtractor) {
      throw std::runtime_error("Failed to cast to subtractor to ConstituentSubtraction object! What?");
    }
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
    if (!csa) {
      // We use a ss here because we want to see the address of the base CS to ensure that it isn't
      // null too. This is probably redundant since it seems unlikely to fully fail to define a CS,
      // but it helps me think about the debugging conceptually, so I will leave it.
      std::stringstream ss;
      ss << "ClusterSequenceArea for validation is invalid! csa=" << static_cast<void*>(csa.get()) << ", cs: " << static_cast<void*>(cs.get());
      throw std::runtime_error(ss.str());
    }
    csa->area_def().ghost_spec().get_last_seed(checkFixedSeed);
    if (checkFixedSeed != mainJetFinder.areaSettings->randomSeed) {
      std::string values = "";
      for (const auto & v : checkFixedSeed) {
        values += " " + std::to_string(v);
      }
      throw std::runtime_error("Seed mismatch in validation mode! Retrieved: " + values);
    }
    // NOTE: This is usually too noisy, but if visual confirmation is needed beyond the check above, uncomment the lines below.
    //std::cout << "Fixed seeds (main jet finding): ";
    //for (auto & v : checkFixedSeed) {
    //  std::cout << " " << v;
    //}
    //std::cout << "\n";
  }

  // Apply the subtractor when appropriate
  if (subtractor && backgroundSubtraction.type != BackgroundSubtraction_t::eventWiseCS) {
    jets = (*subtractor)(jets);
  }

  // Apply the jet selector after all subtraction is completed.
  // NOTE: It's okay that we already applied the min jet pt cut when we take the inclusive_jets above
  //       because any additional subtraction will just remove more jets (ie. the first cut is _less_
  //       restrictive than the second)
  jets = mainJetFinder.selectorPtEtaNonGhost()(jets);

  // Sort by pt for convenience
  // NOTE: For embed pythia validation (and only there), we should disable this line since apparently
  //       the AliPhysics sorting isn't stable in that case...
  jets = fastjet::sorted_by_pt(jets);

  return FindJetsImplementationOutputWrapper{
    cs, backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices
  };
}

template<typename T>
OutputWrapper<T> findJets(
  FourVectorTuple<T> & columnFourVectors,
  std::vector<int> & columnUserIndices,
  const JetFindingSettings & mainJetFinder,
  FourVectorTuple<T> & backgroundEstimatorFourVectors,
  const BackgroundSubtraction & backgroundSubtraction
)
{
  // Use jet finding implementation to do most of the work
  auto && [cs, backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices] = findJetsImplementation(
    columnFourVectors, columnUserIndices, mainJetFinder, backgroundEstimatorFourVectors, backgroundSubtraction
  );

  // Now, handle returning the values.
  // First, we grab the jets themselves, converting the four vectors into column vector to return them.
  auto numpyJets = pseudoJetsToVectors<T>(jets);
  // Next, we grab whatever other properties we desire:
  // Jet area
  auto columnarJetsArea = extractJetsArea<T>(jets, mainJetFinder);
  // Next, grab event wide properties
  // Rho value (storing 0 if not available)
  T rhoValue = backgroundEstimator ? backgroundEstimator->rho() : 0;
  // Finally, we need to associate the constituents with the jets. To do so, we store one vector per jet,
  // with the vector containing the user_index assigned earlier in the jet finding process.
  auto constituentIndices = constituentIndicesFromJets(jets);

  if (backgroundSubtraction.type == BackgroundSubtraction_t::eventWiseCS ||
      backgroundSubtraction.type == BackgroundSubtraction_t::jetWiseCS) {
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
  const JetFindingSettings & mainJetFinder,
  const bool storeRecursiveSplittings
)
{
  // Use jet finding implementation to do most of the work
  // We don't want background subtraction here. To disable it, we create a simple empty container
  // and set the setting to disabled
  FourVectorTuple<T> backgroundEstimatorFourVectors = {{}, {}, {}, {}};
  BackgroundSubtraction backgroundSubtraction{BackgroundSubtraction_t::disabled, nullptr, nullptr};
  // We don't care about passing a custom user index for the background subtraction calculation because
  // the need to pass them is related to MC, where we shouldn't need background subtraction
  std::vector<int> columnFourVectorsUserIndices = {};
  auto && [cs, backgroundEstimator, jets, particlePseudoJets, subtractedToUnsubtractedIndices] = findJetsImplementation(
    columnFourVectors, columnFourVectorsUserIndices, mainJetFinder, backgroundEstimatorFourVectors, backgroundSubtraction
  );

  // Now that we're done with the jet finding, we just need to extract the splittings and
  // put them into the expected output format.
  // First, extract the reclustered jet
  fastjet::PseudoJet jj = jets.at(0);
  std::cout << "Reclustered jet pt=" << jj.pt() << "\n";
  // And store the jet splittings.
  JetSubstructure::JetSubstructureSplittings jetSplittings;
  int splittingNodeIndex = -1;
  ExtractJetSplittings(jetSplittings, jj, splittingNodeIndex, true, storeRecursiveSplittings);

  return jetSplittings;
}

std::ostream& operator<<(std::ostream& in, const AreaSettings & c);
std::ostream& operator<<(std::ostream& in, const JetFindingSettings & c);
std::ostream& operator<<(std::ostream& in, const BackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const JetMedianBackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const GridMedianBackgroundEstimator & c);
std::ostream& operator<<(std::ostream& in, const BackgroundSubtraction_t& c);
std::ostream& operator<<(std::ostream& in, const BackgroundSubtractor& c);
std::ostream& operator<<(std::ostream& in, const RhoSubtractor& c);
std::ostream& operator<<(std::ostream& in, const ConstituentSubtractor& c);
std::ostream& operator<<(std::ostream& in, const BackgroundSubtraction& c);

} /* namespace mammoth */
