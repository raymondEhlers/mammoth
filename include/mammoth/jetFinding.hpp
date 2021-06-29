#include <fastjet/PseudoJet.hh>

namespace mammoth {

// Convenience
template<typename T>
using FourVectorTuple = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>;

template<typename T>
std::vector<fastjet::PseudoJet> & vectorsToPseudoJets(
    const FourVectorTuple<T> & fourVectors
);

template<typename T>
FourVectorTuple<T> pseudoJetsToVectors(
    const std::vector<fastjet::PseudoJet> & jets
);

std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
);

std::vector<unsigned int> updateSubtractedConstituentIndices(
  const std::vector<fastjet::PseudoJet> & pseudoJets
);

template<typename T>
std::tuple<FourVectorTuple<T>, std::vector<std::vector<unsigned int>>, std::optional<std::tuple<FourVectorTuple<T>, std::vector<unsigned int>>>> findJets(
  FourVectorTuple<T> & columnFourVectors,
  double jetR,
  std::tuple<double, double> etaRange = std::make_tuple(-0.9, 0.9)
);

}
