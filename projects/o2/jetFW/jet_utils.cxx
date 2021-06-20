
//
// Initial code based on AliEmcalJetTaggerTaskFast in AliPhysics by Markus Fasel
//

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <TKDTree.h>


/**
  * Duplicates jets around the phi boundary which are within the matching distance.
  *
  * NOTE: Assumes, but does not validate, that 0 <= phi < 2pi.
  *
  * @param jetsPhi Jets phi
  * @param jetsEta Jets eta
  * @param maxMatchingDistance Maximum matching distance. Only duplicate jets within this distance of the boundary.
  */
template<typename T>
std::tuple<std::vector<std::size_t>, std::vector<T>, std::vector<T>> DuplicateJetsAroundPhiBoundary(
    std::vector<T> & jetsPhi,
    std::vector<T> & jetsEta,
    double maxMatchingDistance,
    // TODO: Remove additional margin after additional testing.
    double additionalMargin = 0.05
)
{
    const std::size_t nJets = jetsPhi.size();
    std::vector<std::size_t> jetsMapToJetIndex(nJets);
    // We need to keep track of the map from the duplicated vector to the existing jets.
    // To start, we fill this map (practically, it maps from vector index to an index, so we
    // just use a standard vector) with the existing jet indices, which range from 0..nJets.
    std::iota(jetsMapToJetIndex.begin(), jetsMapToJetIndex.end(), 0);

    // The full duplicated jets will be stored in a new vector to avoid disturbing the input data.
    std::vector<T> jetsPhiComparison(jetsPhi);
    std::vector<T> jetsEtaComparison(jetsEta);

    // Duplicate the jets
    // When a jet is outside of the desired phi range, we make a copy of the jet, shifting it by 2pi
    // as necessary. The eta value is copied directly, as is the index of the original jet, so we can
    // map from index in duplicated data -> index in original collection.
    // NOTE: Assumes, but does not validate, that 0 <= phi < 2pi.
    for (std::size_t i = 0; i < nJets; i++) {
        // Handle lower edge
        if (jetsPhi[i] <= (maxMatchingDistance + additionalMargin)) {
            jetsPhiComparison.emplace_back(jetsPhi[i] + 2 * M_PI);
            jetsEtaComparison.emplace_back(jetsEta[i]);
            jetsMapToJetIndex.emplace_back(jetsMapToJetIndex[i]);
        }
        // Handle upper edge
        if (jetsPhi[i] >= (2 * M_PI - (maxMatchingDistance + additionalMargin))) {
            jetsPhiComparison.emplace_back(jetsPhi[i] - 2 * M_PI);
            jetsEtaComparison.emplace_back(jetsEta[i]);
            jetsMapToJetIndex.emplace_back(jetsMapToJetIndex[i]);
        }
    }

    return std::move(std::make_tuple(jetsMapToJetIndex, jetsPhiComparison, jetsEtaComparison));
}

/**
  * Implementation of geometrical jet matching.
  *
  * Jets are required to match uniquely - namely: base <-> tag. Only one direction of matching isn't enough.
  * Unless special conditions are required, it's better to use `MatchJetsGeometrically`, which has an
  * easier to use interface.
  *
  * NOTE: The vectors for matching could all be const, except that SetData() doesn't take a const.
  *
  * @param jetsBasePhi Base jet collection phi.
  * @param jetsBaseEta Base jet collection eta.
  * @param jetsBasePhiForMatching Base jet collection phi to use for matching.
  * @param jetsBaseEtaForMatching Base jet collection eta to use for matching.
  * @param jetMapBaseToJetIndex Base jet collection index map from duplicated jets to original jets.
  * @param jetsTagPhi Tag jet collection phi.
  * @param jetsTagEta Tag jet collection eta.
  * @param jetsTagPhiForMatching Tag jet collection phi to use for matching.
  * @param jetsTagEtaForMatching Tag jet collection eta to use for matching.
  * @param jetMapTagToJetIndex Tag jet collection index map from duplicated jets to original jets.
  * @param maxMatchingDistance Maximum matching distance.
  *
  * @returns (Base to tag index map, tag to base index map) for uniquely matched jets.
  */
template<typename T>
std::tuple<std::vector<int>, std::vector<int>> MatchJetsGeometricallyImpl(
        const std::vector<T> & jetsBasePhi,
        const std::vector<T> & jetsBaseEta,
        std::vector<T> jetsBasePhiForMatching,
        std::vector<T> jetsBaseEtaForMatching,
        const std::vector<std::size_t> jetMapBaseToJetIndex,
        const std::vector<T> & jetsTagPhi,
        const std::vector<T> & jetsTagEta,
        std::vector<T> jetsTagPhiForMatching,
        std::vector<T> jetsTagEtaForMatching,
        const std::vector<std::size_t> jetMapTagToJetIndex,
        double maxMatchingDistance
        )
{
    // Validation
    // If no jets in either collection, then return immediately.
    const std::size_t nJetsBase = jetsBaseEta.size();
    const std::size_t nJetsTag = jetsTagEta.size();
    if(!(nJetsBase && nJetsTag)) {
        return std::make_tuple(std::vector<int>(nJetsBase), std::vector<int>(nJetsTag));
    }
    // Require that the comparison vectors are greater than or equal to the standard collections.
    if (jetsBasePhiForMatching.size() < jetsBasePhi.size()) {
        throw std::invalid_argument("Base collection phi for matching is smaller than the input base collection.");
    }
    if (jetsBaseEtaForMatching.size() < jetsBaseEta.size()) {
        throw std::invalid_argument("Base collection eta for matching is smaller than the input base collection.");
    }
    if (jetsTagPhiForMatching.size() < jetsTagPhi.size()) {
        throw std::invalid_argument("Tag collection phi for matching is smaller than the input tag collection.");
    }
    if (jetsTagEtaForMatching.size() < jetsTagEta.size()) {
        throw std::invalid_argument("Tag collection eta for matching is smaller than the input tag collection.");
    }

    // Build the KD-trees using vectors
    // We build two trees:
    // treeBase, which contains the base collection.
    // treeTag, which contains the tag collection.
    // The trees are built to match in two dimensions (eta, phi)
    TKDTree<int, T> treeBase(jetsBaseEtaForMatching.size(), 2, 1), treeTag(jetsTagEtaForMatching.size(), 2, 1);
    // By utilizing SetData, we can avoid having to copy the data again.
    treeBase.SetData(0, jetsBaseEtaForMatching.data());
    treeBase.SetData(1, jetsBasePhiForMatching.data());
    treeBase.Build();
    treeTag.SetData(0, jetsTagEtaForMatching.data());
    treeTag.SetData(1, jetsTagPhiForMatching.data());
    treeTag.Build();

    // Storage for the jet matching indices.
    // matchIndexTag maps from the base index to the tag index.
    // matchBaseTag maps from the tag index to the base index.
    std::vector<int> matchIndexTag(nJetsBase, -1), matchIndexBase(nJetsTag, -1);

    // Find the tag jet closest to each base jet.
    for (std::size_t iBase = 0; iBase < nJetsBase; iBase++) {
        Double_t point[2] = {jetsBaseEta[iBase], jetsBasePhi[iBase]};
        Int_t index(-1); Double_t distance(-1);
        treeTag.FindNearestNeighbors(point, 1, &index, &distance);
        // test whether indices are matching:
        if (index >= 0 && distance < maxMatchingDistance) {
            //LOG(DEBUG) << "Found closest tag jet for " << iBase << " with match index " << index << " and distance " << distance << "\n";
            std::cout << "Found closest tag jet for " << iBase << " with match index " << index << " and distance " << distance << "\n";
            matchIndexTag[iBase] = index;
        } else {
            //LOG(DEBUG) << "Closest tag jet not found for" << iBase << ", distance to closest " << distance << "\n";
            std::cout << "Closest tag jet not found for " << iBase << ", distance to closest " << distance << "\n";
        }

#ifdef JET_MATCHING_VALIDATION
        if(index > -1){
            double distanceTest = std::sqrt(std::pow(jetsTagEta[index] - jetsBaseEta, 2) +  std::pow(jetsTagPhi[index] - jetsBasePhi, 2));
            if(std::abs(distanceTest - distance) > std::numeric_limits<double>::epsilon()){
                //LOG(DEBUG) << "Mismatch in distance from tag jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                std::cout << "Mismatch in distance from tag jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                //fIndexErrorRateBase->Fill(1);
            }
        }
#endif
    }

    // Find the base jet closest to each tag jet
    for (std::size_t iTag = 0; iTag < nJetsTag; iTag++) {
        Double_t point[2] = {jetsTagEta[iTag], jetsTagPhi[iTag]};
        Int_t index(-1); Double_t distance(-1);
        treeBase.FindNearestNeighbors(point, 1, &index, &distance);
        if(index >= 0 && distance < maxMatchingDistance) {
            //LOG(DEBUG) << "Found closest base jet for " << iTag << " with match index " << index << " and distance " << distance << std::endl;
            std::cout << "Found closest base jet for " << iTag << " with match index " << index << " and distance " << distance << std::endl;
            matchIndexBase[iTag] = index;
        } else {
            //LOG(DEBUG) << "Closest tag jet not found for " << iTag << ", distance to closest " << distance << "\n";
            std::cout << "Closest tag jet not found for " << iTag << ", distance to closest " << distance << "\n";
        }

#ifdef JET_MATCHING_VALIDATION
        if(index > -1){
            double distanceTest = std::sqrt(std::pow(jetsBaseEta[index] - jetsTagEta, 2) +  std::pow(jetsBasePhi[index] - jetsTagPhi, 2));
            if(std::abs(distanceTest - distance) > std::numeric_limits<double>::epsilon()){
                //LOG(DEBUG) << "Mismatch in distance from base jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                std::cout << "Mismatch in distance from base jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                //fIndexErrorRateTag->Fill(1);
            }
        }
#endif
    }

    // Convert indices in the duplicated jet vectors into the original jet indices.
    // First for the base -> tag map.
    for (auto & v : matchIndexTag) {
        // If it's -1, it means that it didn't find a matching jet.
        // We have to explicitly check for it here because it would be an invalid index.
        if (v != -1) {
            v = jetMapTagToJetIndex[v];
        }
    }
    // Then for the index -> base map.
    for (auto & v : matchIndexBase) {
        if (v != -1) {
            v = jetMapBaseToJetIndex[v];
        }
    }

    // Finally, we'll check for true matches, which are pairs where the base jet is the
    // closest to the tag jet and vice versa
    // As the lists are linear a loop over the outer base jet is sufficient.
    std::vector<int> baseToTagMap(nJetsBase, -1);
    std::vector<int> tagToBaseMap(nJetsTag, -1);
    //LOG(DEBUG) << "Starting true jet loop: nbase(" << nJetsBase << "), ntag(" << nJetsTag << ")\n";
    std::cout << "Starting true jet loop: nbase(" << nJetsBase << "), ntag(" << nJetsTag << ")\n";
    for (std::size_t iBase = 0; iBase < nJetsBase; iBase++) {
        //LOG(DEBUG) << "base jet " << iBase << ": match index in tag jet container " << matchIndexTag[iBase] << "\n";
        std::cout << "base jet " << iBase << ": match index in tag jet container " << matchIndexTag[iBase] << "\n";
        if (matchIndexTag[iBase] > -1){
            //LOG(DEBUG) << "tag jet " << matchIndexTag[iBase] << ": matched base jet " << matchIndexBase[matchIndexTag[iBase]] << "\n";
            std::cout << "tag jet " << matchIndexTag[iBase] << ": matched base jet " << matchIndexBase[matchIndexTag[iBase]] << "\n";
        }
        if (matchIndexTag[iBase] > -1 && matchIndexBase[matchIndexTag[iBase]] == iBase) {
            //LOG(DEBUG) << "True match! base index: " << iBase << ", tag index: " << matchIndexTag[iBase] << "\n";
            std::cout << "True match! base index: " << iBase << ", tag index: " << matchIndexTag[iBase] << ". Storing\n";
            baseToTagMap[iBase] = matchIndexTag[iBase];
            tagToBaseMap[matchIndexTag[iBase]] = iBase;
        }
    }

    return std::make_tuple(baseToTagMap, tagToBaseMap);
}


/**
  * Geometrical jet matching.
  *
  * Match jets in the "base" collection with those in the "tag" collection. Jets are matched within
  * the provided matching distance. Jets are required to match uniquely - namely: base <-> tag.
  * Only one direction of matching isn't enough.
  *
  * If no unique match was found for a jet, an index of -1 is stored.
  *
  * @param jetsBasePhi Base jet collection phi.
  * @param jetsBaseEta Base jet collection eta.
  * @param jetsTagPhi Tag jet collection phi.
  * @param jetsTagEta Tag jet collection eta.
  * @param maxMatchingDistance Maximum matching distance.
  *
  * @returns (Base to tag index map, tag to base index map) for uniquely matched jets.
  */
template<typename T>
std::tuple<std::vector<int>, std::vector<int>> MatchJetsGeometrically(
    std::vector<T> jetsBasePhi,
    std::vector<T> jetsBaseEta,
    std::vector<T> jetsTagPhi,
    std::vector<T> jetsTagEta,
    double maxMatchingDistance
)
{
    // Validation
    const std::size_t nJetsBase = jetsBaseEta.size();
    const std::size_t nJetsTag = jetsTagEta.size();
    if(!(nJetsBase && nJetsTag)) {
        // There are no jets, so nothing to be done.
        return std::make_tuple(std::vector<int>(nJetsBase), std::vector<int>(nJetsTag));
    }
    // Input sizes must match
    if (jetsBasePhi.size() != jetsBaseEta.size()) {
        throw std::invalid_argument("Base collection eta and phi sizes don't match. Check the inputs.");
    }
    if (jetsTagPhi.size() != jetsTagEta.size()) {
        throw std::invalid_argument("Tag collection eta and phi sizes don't match. Check the inputs.");
    }

    // To perform matching with periodic boundary conditions (ie. phi) with a KDTree, we need
    // to duplicate data up to maxMatchingDistance in phi because phi is periodic.
    // NOTE: vectors are modified in place to avoid copies.
    auto && [jetMapBaseToJetIndex, jetsBasePhiComparison, jetsBaseEtaComparison] = DuplicateJetsAroundPhiBoundary(jetsBasePhi, jetsBaseEta, maxMatchingDistance);
    auto && [jetMapTagToJetIndex, jetsTagPhiComparison, jetsTagEtaComparison] = DuplicateJetsAroundPhiBoundary(jetsTagPhi, jetsTagEta, maxMatchingDistance);

    // Finally, perform the actual matching.
    auto && [baseToTagMap, tagToBaseMap] = MatchJetsGeometricallyImpl(
        jetsBasePhi, jetsBaseEta, jetsBasePhiComparison, jetsBaseEtaComparison, jetMapBaseToJetIndex,
        jetsTagPhi, jetsTagEta, jetsTagPhiComparison, jetsTagEtaComparison, jetMapTagToJetIndex,
        maxMatchingDistance
    );

    return std::make_tuple(baseToTagMap, tagToBaseMap);
}

/**
  * Test standard geometrical matching.
  */
void test_jet_matching1() {
    std::cout << "***** Test ****\n";
    std::vector<double> jetsBaseEta = {0.5, -0.6};
    std::vector<double> jetsBasePhi = {1.2, 1.3};
    std::vector<double> jetsTagEta = {0.55, -0.65};
    std::vector<double> jetsTagPhi = {1.25, 1.4};
    double maxMatchingDistance = 1.0;

    auto && [baseToTagIndexMap, tagToBaseIndexMap] = MatchJetsGeometrically(
        jetsBasePhi, jetsBaseEta,
        jetsTagPhi, jetsTagEta,
        maxMatchingDistance
    );
    // assert doesn't work with root 6, so we do this hack with exceptions...
    if (baseToTagIndexMap.size() != jetsBasePhi.size()) throw std::runtime_error("baseToTagIndexMap is wrong size.");
    if (tagToBaseIndexMap.size() != jetsTagPhi.size()) throw std::runtime_error("tagToBaseIndexMap is wrong size.");
    if (baseToTagIndexMap[0] != 0) throw std::runtime_error("Match for first base jet failed.");
    if (baseToTagIndexMap[1] != 1) throw std::runtime_error("Match for second base jet failed.");
    if (tagToBaseIndexMap[0] != 0) throw std::runtime_error("Match for first tag jet failed.");
    if (tagToBaseIndexMap[1] != 1) throw std::runtime_error("Match for second tag jet failed.");
    std::cout << "\tSuccess for matching case 1!\n";
}

/**
  * Test phi boundary conditions.
  */
void test_jet_matching2() {
    std::cout << "***** Test ****\n";
    std::vector<double> jetsBaseEta = {0.5};
    std::vector<double> jetsBasePhi = {0.1};
    std::vector<double> jetsTagEta = {0.55};
    std::vector<double> jetsTagPhi = {6.1};
    double maxMatchingDistance = 0.5;

    auto && [baseToTagIndexMap, tagToBaseIndexMap] = MatchJetsGeometrically(
        jetsBasePhi, jetsBaseEta,
        jetsTagPhi, jetsTagEta,
        maxMatchingDistance
    );
    if (baseToTagIndexMap.size() != jetsBasePhi.size()) throw std::runtime_error("baseToTagIndexMap is wrong size.");
    if (tagToBaseIndexMap.size() != jetsTagPhi.size()) throw std::runtime_error("tagToBaseIndexMap is wrong size.");
    if (baseToTagIndexMap[0] != 0) throw std::runtime_error("Match for first base jet failed.");
    if (tagToBaseIndexMap[0] != 0) throw std::runtime_error("Match for first tag jet failed.");
    std::cout << "\tSuccess for matching case 2!\n";
}

/**
  * Test phi boundary conditions with an another jet within matching distance, but further away.
  */
void test_jet_matching3() {
    std::cout << "***** Test ****\n";
    std::vector<double> jetsBaseEta = {0.5};
    std::vector<double> jetsBasePhi = {0.1};
    std::vector<double> jetsTagEta = {0.45, 0.55};
    std::vector<double> jetsTagPhi = {0.7, 2 * M_PI};
    double maxMatchingDistance = 0.5;

    auto && [baseToTagIndexMap, tagToBaseIndexMap] = MatchJetsGeometrically(
        jetsBasePhi, jetsBaseEta,
        jetsTagPhi, jetsTagEta,
        maxMatchingDistance
    );
    if (baseToTagIndexMap.size() != jetsBasePhi.size()) throw std::runtime_error("baseToTagIndexMap is wrong size.");
    if (tagToBaseIndexMap.size() != jetsTagPhi.size()) throw std::runtime_error("tagToBaseIndexMap is wrong size.");
    if (baseToTagIndexMap[0] != 1) throw std::runtime_error("Match for first base jet failed.");
    if (tagToBaseIndexMap[0] != -1) throw std::runtime_error("Match for first tag jet failed.");
    if (tagToBaseIndexMap[1] != 0) throw std::runtime_error("Match for second tag jet failed.");
    std::cout << "\tSuccess for matching case 3!\n";
}

/**
  * Test phi boundary conditions with an another jet within matching distance, but closer than the periodic condition jet.
  */
void test_jet_matching4() {
    std::cout << "***** Test ****\n";
    std::vector<double> jetsBaseEta = {0.5};
    std::vector<double> jetsBasePhi = {0.4};
    std::vector<double> jetsTagEta = {0.45, 0.55};
    std::vector<double> jetsTagPhi = {0.7, 2 * M_PI};
    double maxMatchingDistance = 0.5;

    auto && [baseToTagIndexMap, tagToBaseIndexMap] = MatchJetsGeometrically(
        jetsBasePhi, jetsBaseEta,
        jetsTagPhi, jetsTagEta,
        maxMatchingDistance
    );
    if (baseToTagIndexMap.size() != jetsBasePhi.size()) throw std::runtime_error("baseToTagIndexMap is wrong size.");
    if (tagToBaseIndexMap.size() != jetsTagPhi.size()) throw std::runtime_error("tagToBaseIndexMap is wrong size.");
    if (baseToTagIndexMap[0] != 0) throw std::runtime_error("Match for first base jet failed.");
    if (tagToBaseIndexMap[0] != 0) throw std::runtime_error("Match for first tag jet failed.");
    if (tagToBaseIndexMap[1] != -1) throw std::runtime_error("Match for second tag jet failed.");
    std::cout << "\tSuccess for matching case 4!\n";
}

void jet_utils() {
    test_jet_matching1();
    test_jet_matching2();
    test_jet_matching3();
    test_jet_matching4();
}
