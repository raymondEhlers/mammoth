
//
// Based on code by Markus Fasel
//

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <TKDTree.h>


// TODO: Make naming uniform...

template<typename T>
std::tuple<std::vector<std::size_t>, std::vector<T>, std::vector<T>> DuplicateJetsAroundPhiEdges(
    std::vector<T> & jetsPhi,
    std::vector<T> & jetsEta,
    double maxMatchingDistance
)
{
    const std::size_t nJets = jetsPhi.size();
    std::vector<std::size_t> jetsMapToJetIndex(nJets);
    // First, fill them up with the indices of the existing jets.
    std::iota(jetsMapToJetIndex.begin(), jetsMapToJetIndex.end(), 0);

    // Copy base values
    std::vector<T> jetsPhiComparison(jetsPhi);
    std::vector<T> jetsEtaComparison(jetsEta);

    // NOTE: Assumes 0 <= phi < 2pi.
    // TODO: Validation phi range(?)
    // TODO: Remove additional margin afer some testing.
    double additionalMargin = 0.05;
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

template<typename T>
std::tuple<std::vector<int>, std::vector<int>> MatchJetsGeometricallyImpl(
        // NOTE: These could all be const, except that SetData() doesn't take a const. Sigh...
        const std::vector<T> & jetBasePhi,
        const std::vector<T> & jetBaseEta,
        std::vector<T> jetBasePhiComparison,
        std::vector<T> jetBaseEtaComparison,
        const std::vector<std::size_t> jetMapBaseToJetIndex,
        const std::vector<T> & jetTagPhi,
        const std::vector<T> & jetTagEta,
        std::vector<T> jetTagPhiComparison,
        std::vector<T> jetTagEtaComparison,
        const std::vector<std::size_t> jetMapTagToJetIndex,
        double maxMatchingDistance
        )
{
    // TODO: Further validation
    const std::size_t nJetsBase = jetBaseEta.size();
    const std::size_t nJetsTag = jetTagEta.size();
    if(!(nJetsBase && nJetsTag)) {
        return std::make_tuple(std::vector<int>(), std::vector<int>());
    }
    // TODO: Requier that the comparison is greater than or equal to the standard collections.

    // Build kd-trees
    //TArrayD etaBase(nJetsBase, jetBaseEta.data()), phiBase(nJetsBase, jetBasePhi.data()),
    //        etaTag(nJetsTag, jetTagEta.data()), phiTag(nJetsTag, jetTagPhi.data());
    //std::vector<AliEmcalJet *> jetsBase(nJetsBase), jetsTag(nJetsTag); // the storages are needed later for applying the tagging, in order to avoid multiple occurrence of jet selection

    /*for(auto jb : contBase.accepted()) {
        etaBase[countBase] = jb->Eta();
        phiBase[countBase] = jb->Phi();
        jetsBase[countBase] = jb;
        countBase++;
    }
    for(auto jt : contTag.accepted()) {
        etaTag[countTag] = jt->Eta();
        phiTag[countTag] = jt->Phi();
        jetsTag[countTag] = jt;
        countTag++;
    }*/

    // Build KDTrees using vector
    //TKDTreeID treeBase(jetBaseEta.size(), 2, 1), treeTag(jetTagEta.size(), 2, 1);
    TKDTree<int, T> treeBase(jetBaseEtaComparison.size(), 2, 1), treeTag(jetTagEtaComparison.size(), 2, 1);
    treeBase.SetData(0, jetBaseEtaComparison.data());
    treeBase.SetData(1, jetBasePhiComparison.data());
    treeBase.Build();
    treeTag.SetData(0, jetTagEtaComparison.data());
    treeTag.SetData(1, jetTagPhiComparison.data());
    treeTag.Build();

    std::vector<int> matchIndexTag(nJetsBase, -1), matchIndexBase(nJetsTag, -1);

    // Find the tag jet closest to each base jet.
    for (std::size_t iBase = 0; iBase < nJetsBase; iBase++) {
        Double_t point[2] = {jetBaseEta[iBase], jetBasePhi[iBase]};
        Int_t index(-1); Double_t distance(-1);
        treeTag.FindNearestNeighbors(point, 1, &index, &distance);
        // test whether indices are matching:
        if(index >= 0 && distance < maxMatchingDistance) {
            //LOG(DEBUG) << "Found closest tag jet for " << iBase << " with match index " << index << " and distance " << distance << "\n";
            std::cout << "Found closest tag jet for " << iBase << " with match index " << index << " and distance " << distance << "\n";
            matchIndexTag[iBase] = index;
        } else {
            //LOG(DEBUG) << "Closest tag jet not found for" << iBase << ", distance to closest " << distance << "\n";
            std::cout << "Closest tag jet not found for " << iBase << ", distance to closest " << distance << "\n";
        }

#ifdef JETTAGGERFAST_TEST
        if(index > -1){
            double distanceTest = std::sqrt(std::pow(jetTagEta[index] - jetBaseEta, 2) +  std::pow(jetTagPhi[index] - jetBasePhi, 2));
            if(std::abs(distanceTest - distance) > std::numeric_limits<double>::epsilon()){
                //LOG(DEBUG) << "Mismatch in distance from tag jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                std::cout << "Mismatch in distance from tag jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                //fIndexErrorRateBase->Fill(1);
            }
        }
#endif
    }

    // other way around
    for (std::size_t iTag = 0; iTag < nJetsTag; iTag++) {
        Double_t point[2] = {jetTagEta[iTag], jetTagPhi[iTag]};
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

#ifdef JETTAGGERFAST_TEST
        if(index > -1){
            double distanceTest = std::sqrt(std::pow(jetBaseEta[index] - jetTagEta, 2) +  std::pow(jetBasePhi[index] - jetTagPhi, 2));
            if(std::abs(distanceTest - distance) > std::numeric_limits<double>::epsilon()){
                //LOG(DEBUG) << "Mismatch in distance from base jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                std::cout << "Mismatch in distance from base jet with index from tree: " << distanceTest << ", distance from tree " << distance << "\n";
                //fIndexErrorRateTag->Fill(1);
            }
        }
#endif
    }

    // Convert...
    //std::cout << "before tag\n";
    for (auto & v : matchIndexTag) {
        //std::cout << "v: " << v << "\n";
        if (v != -1) {
            v = jetMapTagToJetIndex[v];
        }
    }
    //std::cout << "after tag\n";
    //for (const auto & v : matchIndexTag) {
    //    std::cout << "v: " << v << "\n";
    //}
    //std::cout << "before base\n";
    for (auto & v : matchIndexBase) {
        //std::cout << "v: " << v << "\n";
        if (v != -1) {
            v = jetMapBaseToJetIndex[v];
        }
    }
    //std::cout << "after base\n";
    //for (const auto & v : matchIndexBase) {
    //    std::cout << "v: " << v << "\n";
    //}

    std::vector<int> baseToTagMap(nJetsBase, -1);
    std::vector<int> tagToBaseMap(nJetsTag, -1);

    // check for "true" correlations
    // these are pairs where the base jet is the closest to the tag jet and vice versa
    // As the lists are linear a loop over the outer base jet is sufficient.
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
            std::cout << "True match! base index: " << iBase << ", tag index: " << matchIndexTag[iBase] << "\n";
            baseToTagMap[iBase] = matchIndexTag[iBase];
            tagToBaseMap[matchIndexTag[iBase]] = iBase;

            // NOTE: These tests below will only work if periodic phi is accounted for properly.
            /*AliEmcalJet *jetBase = jetsBase[iBase],
                                    *jetTag = jetsTag[matchIndexTag[iBase]];
            if(jetBase && jetTag) {
#ifdef JETTAGGERFAST_TEST
                if(std::abs(etaBase[iBase] - jetBase->Eta()) > std::numeric_limits<double>::epsilon() || std::abs(phiBase[iBase] - jetBase->Phi()) > std::numeric_limits<double>::epsilon()){
                    LOG(ERROR)() << "Selected incorrect base jet for tagging : eta test(" << jetBase->Eta() << ")/true(" << etaBase[iBase]
                                                      << "), phi test(" << jetBase->Phi() << ")/true(" << phiBase[iBase] << ")\n";
                    fContainerErrorRateBase->Fill(1);
                }
                if(std::abs(etaTag[matchIndexTag[iBase]] - jetTag->Eta()) > std::numeric_limits<double>::epsilon() || std::abs(phiTag[matchIndexTag[iBase]] - jetTag->Phi()) > std::numeric_limits<double>::epsilon()){
                    LOG(ERROR)() << "Selected incorrect tag jet for tagging : eta test(" << jetTag->Eta() << ")/true(" << etaTag[matchIndexTag[iBase]]
                                                      << "), phi test(" << jetTag->Phi() << ")/true(" << phiTag[matchIndexTag[iBase]] << ")\n";
                    fContainerErrorRateTag->Fill(1);
                }
#endif
                // Test if the position of the jets correp
                Double_t dR = jetBase->DeltaR(jetTag);
                switch(fJetTaggingType){
                case kTag:
                    jetBase->SetTaggedJet(jetTag);
                    jetBase->SetTagStatus(1);

                    jetTag->SetTaggedJet(jetBase);
                    jetTag->SetTagStatus(1);
                    break;
                case kClosest:
                    jetBase->SetClosestJet(jetTag,dR);
                    jetTag->SetClosestJet(jetBase,dR);
                    break;
                };
            }*/
        }
    }
    return std::make_tuple(baseToTagMap, tagToBaseMap);
}

template<typename T>
bool MatchJetsGeometrically(
    std::vector<T> jetBasePhi,
    std::vector<T> jetBaseEta,
    std::vector<T> jetTagPhi,
    std::vector<T> jetTagEta,
    double maxMatchingDistance
)
{
    // Validation
    // TODO: Additional Validation
    const std::size_t nJetsBase = jetBaseEta.size();
    const std::size_t nJetsTag = jetTagEta.size();
    if(!(nJetsBase && nJetsTag)) {
        return false;
    }

    // Need to duplicate data up to maxMatchingDistance in phi because phi is periodic.
    // NOTE: vectors are modified in place to avoid copies.
    auto && [jetMapBaseToJetIndex, jetBasePhiComparison, jetBaseEtaComparison] = DuplicateJetsAroundPhiEdges(jetBasePhi, jetBaseEta, maxMatchingDistance);
    auto && [jetMapTagToJetIndex, jetTagPhiComparison, jetTagEtaComparison] = DuplicateJetsAroundPhiEdges(jetTagPhi, jetTagEta, maxMatchingDistance);

    //std::cout << "jetMapBaseToJetIndex\n";
    //for (const auto v : jetMapBaseToJetIndex) {
    //    std::cout << v << "\n";
    //}
    //std::cout << "jetMapTagToJetIndex\n";
    //for (const auto v : jetMapTagToJetIndex) {
    //    std::cout << v << "\n";
    //}

    //auto res = MatchJetsGeometricallyImpl(
    auto && [baseToTagMap, tagToBaseMap] = MatchJetsGeometricallyImpl(
        jetBasePhi, jetBaseEta, jetBasePhiComparison, jetBaseEtaComparison, jetMapBaseToJetIndex,
        jetTagPhi, jetTagEta, jetTagPhiComparison, jetTagEtaComparison, jetMapTagToJetIndex,
        maxMatchingDistance
    );

    for (auto v : baseToTagMap) {
        std::cout << "v: " << v << "\n";
    }

    // Now, we remove the duplicated data, updating the mapping as needed.
    /*for (std::size_t i = 0; i < baseToTagMap.size(); ++i)
    {
        int tagIndex = baseToTagMap[i];
        // Update mapping if it's pointing to a duplicated tag.
        if (tagIndex != jetMapTagToJetIndex[tagIndex]) {
            baseToTagMap[i] = jetMapTagToJetIndex[tagIndex];
        }
    }
    std::map<std::size_t, int> finalBaseToTagMap;
    for (std::size_t i = 0; i < baseToTagMap.size(); ++i)
    {
        // Compare the values if we have duplicated jets.
        if (jetMapBaseToJetIndex[i] != i) {
            int tagMatchedToOriginalJet = baseToTagMap[jetMapBaseToJetIndex[i]];
            int tagMatchedToDuplicatedJet = baseToTagMap[i];
            if (tagMatchedToDuplicatedJet == tagMatchedToOriginalJet) {
                finalBaseToTagMap[jetMapBaseToJetIndex[i]] = tagMatchedToOriginalJet;
            }
            else {
                // Matching doesn't agree.
                // TODO: Take the closer jet.
                std::cout << "Matching doesn't agree...\n";
                finalBaseToTagMap[jetMapBaseToJetIndex[i]] = -1;
            }
        }
    }

    for (const auto v : finalBaseToTagMap) {
        std::cout << "v.first: " << v.first << ", v.second: " << v.second << "\n";
    }*/

    /*for (std::size_t i = 0; i < baseToTagMap.size(); ++i)
    {
        int baseJetIndex = jetMapBaseToJetIndex[iBase];
        if (baseJetIndex != jetMapBaseToJetIndex[iBase]) {

        }
        int tagJetIndex = jetMapTagToJetIndex[matchIndexTag[iBase]];
        //LOG(DEBUG) << "found a true match \n";
        std::cout << "found a true match \n";
        // TODO: Need to figure out storage...
        // NOTE: Need to deduplicate carefully...
        if (baseToTagMap[baseJetIndex] != -1) {
            // It's already been set - check it's still unique
            // This may not be the best way to go about this point.
            int existingValue = baseToTagMap[baseJetIndex];
            if (existingValue != ) {

            }
        }
        else {
            // Store the result.
            baseToTagMap[baseJetIndex] = tagJetIndex;
            tagToBaseMap[tagJetIndex] = baseJetIndex;
        }
    }*/

    return true;
}


void test_jet_matching() {
    std::vector<double> jetBaseEta = {0.5, -0.6};
    std::vector<double> jetBasePhi = {1.2, 1.3};
    std::vector<double> jetTagEta = {0.55, -0.65};
    std::vector<double> jetTagPhi = {1.25, 1.4};
    double maxMatchingDistance = 1.0;

    bool res = MatchJetsGeometrically(
        jetBasePhi, jetBaseEta,
        jetTagPhi, jetTagEta,
        maxMatchingDistance
    );
    std::cout << "Res " << res << "\n";
}

void test_jet_matching2() {
    std::vector<double> jetBaseEta = {0.5};
    std::vector<double> jetBasePhi = {0.1};
    std::vector<double> jetTagEta = {0.45, 0.55};
    std::vector<double> jetTagPhi = {0.7, 2 * M_PI};
    double maxMatchingDistance = 0.5;

    bool res = MatchJetsGeometrically(
        jetBasePhi, jetBaseEta,
        jetTagPhi, jetTagEta,
        maxMatchingDistance
    );
    std::cout << "Res " << res << "\n";
}

void test_jet_matching3() {
    std::vector<double> jetBaseEta = {0.5};
    std::vector<double> jetBasePhi = {0.1};
    std::vector<double> jetTagEta = {0.55};
    std::vector<double> jetTagPhi = {6.1};
    double maxMatchingDistance = 0.5;

    bool res = MatchJetsGeometrically(
        jetBasePhi, jetBaseEta,
        jetTagPhi, jetTagEta,
        maxMatchingDistance
    );
    std::cout << "Res " << res << "\n";
}
