
//
// Based on code by Markus Fasel
//

#include <cmath>
#include <limits>
#include <vector>

#include <TKDTree.h>


template<typename T>
std::vector<std::size_t> DuplicateJetsAroundPhiEdges(
    std::vector<T> & jetsPhi,
    std::vector<T> & jetsEta,
    double maxMatchingDistance
)
{
    const std::size_t nJets = jetsPhi.size();
    std::vector<std::size_t> jetsMapToOriginalIndex(nJets);
    // First, fill them up with the indices of the existing jets.
    std::iota(jetsMapToOriginalIndex.begin(), jetsMapToOriginalIndex.end(), 0);

    // NOTE: Assumes 0 <= phi < 2pi.
    // TODO: Validation phi range(?)
    // TODO: Remove additional margin afer some testing.
    double additionalMargin = 0.05;
    for (std::size_t i = 0; i < nJets; i++) {
        // Handle lower edge
        if (jetsPhi[i] <= (maxMatchingDistance + additionalMargin)) {
            jetsPhi.emplace_back(jetsPhi[i] + 2 * M_PI);
            jetsEta.emplace_back(jetsEta[i]);
            jetsMapToOriginalIndex.emplace_back(jetsMapToOriginalIndex[i]);
        }
        // Handle upper edge
        if (jetsPhi[i] >= (2 * M_PI - (maxMatchingDistance + additionalMargin))) {
            jetsPhi.emplace_back(jetsPhi[i] - 2 * M_PI);
            jetsEta.emplace_back(jetsEta[i]);
            jetsMapToOriginalIndex.emplace_back(jetsMapToOriginalIndex[i]);
        }
    }

    return jetsMapToOriginalIndex;
}

//bool AliEmcalJetTaggerTaskFast::MatchJetsGeo(AliJetContainer &contBase, AliJetContainer &contTag, Float_t maxDist) const {
template<typename T>
bool MatchJetsGeometricallyImpl(
        // NOTE: These could all be const, except that SetData() doesn't take a const. Thanks guys...
        std::vector<T> & jetBasePhi,
        std::vector<T> & jetBaseEta,
        std::vector<std::size_t> & jetMapBaseToOriginalIndex,
        //const std::vector<int> jetBaseIndex,
        std::vector<T> & jetTagPhi,
        std::vector<T> & jetTagEta,
        std::vector<std::size_t> & jetMapTagToOriginalIndex,
        //const std::vector<int> jetTagIndex,
        double maxMatchingDistance
        )
{
    // TODO: Further validation
    const std::size_t nJetsBase = jetBaseEta.size();
    const std::size_t nJetsTag = jetTagEta.size();
    if(!(nJetsBase && nJetsTag)) return false;

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
    TKDTreeID treeBase(jetBaseEta.size(), 2, 1), treeTag(jetTagEta.size(), 2, 1);
    treeBase.SetData(0, jetBaseEta.data());
    treeBase.SetData(1, jetBasePhi.data());
    treeBase.Build();
    treeTag.SetData(0, jetTagEta.data());
    treeTag.SetData(1, jetTagPhi.data());
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

    // check for "true" correlations
    // these are pairs where the base jet is the closest to the tag jet and vice versa
    // As the lists are linear a loop over the outer base jet is sufficient.
    //LOG(DEBUG) << "Starting true jet loop: nbase(" << nJetsBase << "), ntag(" << nJetsTag << ")\n";
    std::cout << "Starting true jet loop: nbase(" << nJetsBase << "), ntag(" << nJetsTag << ")\n";
    for (std::size_t iBase = 0; iBase < nJetsBase; iBase++) {
        //LOG(DEBUG) << "base jet " << iBase << ": match index in tag jet container " << matchIndexTag[iBase] << "\n";
        std::cout << "base jet " << iBase << ": match index in tag jet container " << matchIndexTag[iBase] << "\n";
        if(matchIndexTag[iBase] > -1){
          //LOG(DEBUG) << "tag jet " << matchIndexTag[iBase] << ": matched base jet " << matchIndexBase[matchIndexTag[iBase]] << "\n";
            std::cout << "tag jet " << matchIndexTag[iBase] << ": matched base jet " << matchIndexBase[matchIndexTag[iBase]] << "\n";
        }
        if(matchIndexTag[iBase] > -1 && matchIndexBase[matchIndexTag[iBase]] == iBase) {
            //LOG(DEBUG) << "found a true match \n";
            std::cout << "found a true match \n";
            // TODO: Need to figure out storage...
            // NOTE: Need to deduplicate carefully...
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
    return true;
}

template<typename T>
bool MatchJetsGeometrically(
    std::vector<T> jetBasePhi,
    std::vector<T> jetBaseEta,
    //const std::vector<int> jetBaseIndex,
    std::vector<T> jetTagPhi,
    std::vector<T> jetTagEta,
    //const std::vector<int> jetTagIndex,
    double maxMatchingDistance
)
{
    // TODO: Validation
    const std::size_t nJetsBase = jetBaseEta.size();
    const std::size_t nJetsTag = jetTagEta.size();
    if(!(nJetsBase && nJetsTag)) return false;

    // Need to duplicate data up to maxMatchingDistance in phi because phi is cyclical.
    // NOTE: vectors are modified in place to avoid copies.
    std::vector<std::size_t> jetMapBaseToOriginalIndex = DuplicateJetsAroundPhiEdges(jetBasePhi, jetBaseEta, maxMatchingDistance);
    std::vector<std::size_t> jetMapTagToOriginalIndex = DuplicateJetsAroundPhiEdges(jetTagPhi, jetTagEta, maxMatchingDistance);

    bool res = MatchJetsGeometricallyImpl(
        jetBasePhi, jetBaseEta, jetMapBaseToOriginalIndex,
        jetTagPhi, jetTagEta, jetMapTagToOriginalIndex,
        maxMatchingDistance
    );
    return res;
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
    std::vector<double> jetTagEta = {0.55, 0.45};
    std::vector<double> jetTagPhi = {2 * M_PI, 0.7};
    double maxMatchingDistance = 0.5;

    bool res = MatchJetsGeometrically(
        jetBasePhi, jetBaseEta,
        jetTagPhi, jetTagEta,
        maxMatchingDistance
    );
    std::cout << "Res " << res << "\n";
}
