#include "mammoth/jetFinding.hpp"

namespace mammoth {

std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
)
{
  std::vector<std::vector<unsigned int>> indices;
  for (auto jet : jets) {
    std::vector<unsigned int> constituentIndicesInJet;
    for (auto constituent : jet.constituents()) {
      constituentIndicesInJet.push_back(constituent.user_index());
    }
    indices.emplace_back(constituentIndicesInJet);
  }
  return std::move(indices);
}

std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
)
{
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  for (unsigned int i = 0; i < pseudoJets.size(); ++i) {
    subtractedToUnsubtractedIndices.push_back(pseudoJets[i].user_index());
    // The indexing may be different due to the subtraction. So we reset it be certain.
    pseudoJets[i].set_user_index(i);
  }

  return std::move(subtractedToUnsubtractedIndices);
}

} // namesapce mammoth
