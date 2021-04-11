
//
// Based on code by Markus Fasel
//

#include <vector>

#include <TKDTree.h>

//bool AliEmcalJetTaggerTaskFast::MatchJetsGeo(AliJetContainer &contBase, AliJetContainer &contTag, Float_t maxDist) const {
template<typename T>
bool MatchJetsGeometrically(
        const std::vector<T> jetBasePhi,
        const std::vector<T> jetBaseEta,
        //const std::vector<int> jetBaseIndex,
        const std::vector<T> jetTagPhi,
        const std::vector<T> jetTagEta,
        //const std::vector<int> jetTagIndex,
        double maxMatchingDistance,
        )
{
    // TODO: Validation
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

    // Using TArray
    TKDTreeID treeBase(etaBase.GetSize(), 2, 1), treeTag(etaTag.GetSize(), 2, 1);
    treeBase.SetData(0, etaBase.GetArray());
    treeBase.SetData(1, phiBase.GetArray());
    treeBase.Build();
    treeTag.SetData(0, etaTag.GetArray());
    treeTag.SetData(1, phiTag.GetArray());
    treeTag.Build();
    // Using vector

    std::vector<int> matchIndexTag(nJetsBase, -1), matchIndexBase(nJetsTag, -1);

    // find the closest distance to the full jet
    int countBase(0), countTag(0);
    countBase = 0;
    for (std::size_t i = 0; i < nJetsBase; i++) {
    for(auto j : contBase.accepted()) {
        Double_t point[2] = {j->Eta(), j->Phi()};
        Int_t index(-1); Double_t distance(-1);
        treeTag.FindNearestNeighbors(point, 1, &index, &distance);
        // test whether indices are matching:
        if(index >= 0 && distance < maxDist){
            AliDebugStream(1) << "Found closest tag jet for " << countBase << " with match index " << index << " and distance " << distance << std::endl;
            matchIndexTag[countBase]=index;
        } else {
            AliDebugStream(1) << "Not found closest tag jet for " << countBase << ", distance to closest " << distance << std::endl;
        }

#ifdef JETTAGGERFAST_TEST
        if(index>-1){
            Double_t distanceTest(-1);
            distanceTest = TMath::Sqrt(TMath::Power(etaTag[index] - j->Eta(), 2) +  TMath::Power(phiTag[index] - j->Phi(), 2));
            if(TMath::Abs(distanceTest - distance) > DBL_EPSILON){
                AliDebugStream(1) << "Mismatch in distance from tag jet with index from tree: " << distanceTest << ", distance from tree " << distance << std::endl;
                fIndexErrorRateBase->Fill(1);
            }
        }
#endif

        countBase++;
    }

    // other way around
    countTag = 0;
    for(auto j : contTag.accepted()){
        Double_t point[2] = {j->Eta(), j->Phi()};
        Int_t index(-1); Double_t distance(-1);
        treeBase.FindNearestNeighbors(point, 1, &index, &distance);
        if(index >= 0 && distance < maxDist){
            AliDebugStream(1) << "Found closest base jet for " << countBase << " with match index " << index << " and distance " << distance << std::endl;
            matchIndexBase[countTag]=index;
        } else {
            AliDebugStream(1) << "Not found closest tag jet for " << countBase << ", distance to closest " << distance << std::endl;
        }

#ifdef JETTAGGERFAST_TEST
        if(index>-1){
            Double_t distanceTest(-1);
            distanceTest = TMath::Sqrt(TMath::Power(etaBase[index] - j->Eta(), 2) +  TMath::Power(phiBase[index] - j->Phi(), 2));
            if(TMath::Abs(distanceTest - distance) > DBL_EPSILON){
                AliDebugStream(1) << "Mismatch in distance from base jet with index from tree: " << distanceTest << ", distance from tree " << distance << std::endl;
                fIndexErrorRateTag->Fill(1);
            }
        }
#endif

        countTag++;
    }

    // check for "true" correlations
    // these are pairs where the base jet is the closest to the tag jet and vice versa
    // As the lists are linear a loop over the outer base jet is sufficient.
    AliDebugStream(1) << "Starting true jet loop: nbase(" << nJetsBase << "), ntag(" << nJetsTag << ")\n";
    for(int ibase = 0; ibase < nJetsBase; ibase++) {
        AliDebugStream(2) << "base jet " << ibase << ": match index in tag jet container " << matchIndexTag[ibase] << "\n";
        if(matchIndexTag[ibase] > -1){
          AliDebugStream(2) << "tag jet " << matchIndexTag[ibase] << ": matched base jet " << matchIndexBase[matchIndexTag[ibase]] << "\n";
        }
        if(matchIndexTag[ibase] > -1 && matchIndexBase[matchIndexTag[ibase]] == ibase) {
            AliDebugStream(2) << "found a true match \n";
            AliEmcalJet *jetBase = jetsBase[ibase],
                                    *jetTag = jetsTag[matchIndexTag[ibase]];
            if(jetBase && jetTag) {
#ifdef JETTAGGERFAST_TEST
                if(TMath::Abs(etaBase[ibase] - jetBase->Eta()) > DBL_EPSILON || TMath::Abs(phiBase[ibase] - jetBase->Phi()) > DBL_EPSILON){
                    AliErrorStream() << "Selected incorrect base jet for tagging : eta test(" << jetBase->Eta() << ")/true(" << etaBase[ibase]
                                                      << "), phi test(" << jetBase->Phi() << ")/true(" << phiBase[ibase] << ")\n";
                    fContainerErrorRateBase->Fill(1);
                }
                if(TMath::Abs(etaTag[matchIndexTag[ibase]] - jetTag->Eta()) > DBL_EPSILON || TMath::Abs(phiTag[matchIndexTag[ibase]] - jetTag->Phi()) > DBL_EPSILON){
                    AliErrorStream() << "Selected incorrect tag jet for tagging : eta test(" << jetTag->Eta() << ")/true(" << etaTag[matchIndexTag[ibase]]
                                                      << "), phi test(" << jetTag->Phi() << ")/true(" << phiTag[matchIndexTag[ibase]] << ")\n";
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
            }
        }
    }
    return kTRUE;
}
