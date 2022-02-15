#include "mammoth/jetFinding.hpp"

namespace mammoth {

std::vector<float> extractJetsArea(
  const std::vector<fastjet::PseudoJet> & jets
)
{
  std::size_t nJets = jets.size();
  std::vector<float> jetsArea(nJets);
  for (std::size_t i = 0; i < nJets; ++i) {
    jetsArea.at(i) = jets.at(i).area();
  }
  return jetsArea;
}

std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
)
{
  std::vector<std::vector<unsigned int>> indices;
  for (auto jet : jets) {
    std::vector<unsigned int> constituentIndicesInJet;
    for (auto constituent : jet.constituents()) {
      // We want to avoid ghosts, which have index -1
      if (constituent.user_index() != -1) {
        constituentIndicesInJet.push_back(constituent.user_index());
      }
    }
    indices.emplace_back(constituentIndicesInJet);
  }
  return indices;
}

std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
)
{
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  // NOTE: For event-wise CS, we don't need to account for ghost particles here (which have user index -1).
  //       However, if we use jet-wise CS, we should maintain the -1 label for when we use active ghosts
  for (unsigned int i = 0; i < pseudoJets.size(); ++i) {
    subtractedToUnsubtractedIndices.push_back(pseudoJets[i].user_index());
    // The indexing may be different due to the subtraction (for example, if a particle is entirely subtracted
    // away). Since we want the index to be continuous (up to ghost particle), we reassign it here to be certain.
    // If it's a ghost particle (with user_index == -1), we keep that assignment so we don't lose that identification.
    pseudoJets[i].set_user_index((pseudoJets[i].user_index() != -1) ? i : -1);
  }

  return subtractedToUnsubtractedIndices;
}


namespace JetSubstructure
{

/**
 * Subjets
 */

/**
 * Default constructor
 */
Subjets::Subjets():
  fSplittingNodeIndex{},
  fPartOfIterativeSplitting{},
  fConstituentIndices{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
Subjets::Subjets(const Subjets& other)
 : fSplittingNodeIndex{other.fSplittingNodeIndex},
  fPartOfIterativeSplitting{other.fPartOfIterativeSplitting},
  fConstituentIndices{other.fConstituentIndices}
{
  // Nothing more to be done.
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
Subjets& Subjets::operator=(Subjets other)
{
  swap(*this, other);
  return *this;
}

bool Subjets::Clear()
{
  fSplittingNodeIndex.clear();
  fPartOfIterativeSplitting.clear();
  fConstituentIndices.clear();
  return true;
}

std::tuple<unsigned short, bool, const std::vector<unsigned short>> Subjets::GetSubjet(int i) const
{
  return std::make_tuple(fSplittingNodeIndex.at(i), fPartOfIterativeSplitting.at(i), fConstituentIndices.at(i));
}

void Subjets::AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting, const std::vector<unsigned short> & constituentIndices)
{
  fSplittingNodeIndex.emplace_back(splittingNodeIndex);
  // NOTE: emplace_back isn't supported for std::vector<bool> until c++14.
  fPartOfIterativeSplitting.push_back(partOfIterativeSplitting);
  // Originally, we stored the constituent indices and their jagged indices separately to try to coax ROOT
  // into storing the nested vectors in a columnar format. However, even with that design, uproot can't
  // recreate the nested jagged array without a slow python loop. So we just store the indices directly
  // and wait for uproot 4. See: https://stackoverflow.com/q/60250877/12907985
  fConstituentIndices.emplace_back(constituentIndices);
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string Subjets::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Subjets:\n";
  for (std::size_t i = 0; i < fSplittingNodeIndex.size(); i++)
  {
    tempSS << "#" << (i + 1) << ": Splitting Node: " << fSplittingNodeIndex.at(i)
        << ", part of iterative splitting = " << fPartOfIterativeSplitting.at(i)
        << ", number of jet constituents = " << fConstituentIndices.at(i).size() << "\n";
  }
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * Subjets::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& Subjets::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

/**
 * Jet splittings
 */

/**
 * Default constructor.
 */
JetSplittings::JetSplittings():
  fKt{},
  fDeltaR{},
  fZ{},
  fParentIndex{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
JetSplittings::JetSplittings(const JetSplittings& other)
 : fKt{other.fKt},
  fDeltaR{other.fDeltaR},
  fZ{other.fZ},
  fParentIndex{other.fParentIndex}
{
  // Nothing more to be done.
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
JetSplittings& JetSplittings::operator=(JetSplittings other)
{
  swap(*this, other);
  return *this;
}

bool JetSplittings::Clear()
{
  fKt.clear();
  fDeltaR.clear();
  fZ.clear();
  fParentIndex.clear();
  return true;
}

void JetSplittings::AddSplitting(float kt, float deltaR, float z, short i)
{
  fKt.emplace_back(kt);
  fDeltaR.emplace_back(deltaR);
  fZ.emplace_back(z);
  fParentIndex.emplace_back(i);
}

std::tuple<float, float, float, short> JetSplittings::GetSplitting(int i) const
{
  return std::make_tuple(fKt.at(i), fDeltaR.at(i), fZ.at(i), fParentIndex.at(i));
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string JetSplittings::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Jet splittings:\n";
  for (std::size_t i = 0; i < fKt.size(); i++)
  {
    tempSS << "#" << (i + 1) << ": kT = " << fKt.at(i)
        << ", deltaR = " << fDeltaR.at(i) << ", z = " << fZ.at(i)
        << ", parent = " << fParentIndex.at(i) << "\n";
  }
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * JetSplittings::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& JetSplittings::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

/**
 * Jet substructure splittings container.
 */

/**
 * Default constructor.
 */
JetSubstructureSplittings::JetSubstructureSplittings():
  fJetSplittings{},
  fSubjets{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
JetSubstructureSplittings::JetSubstructureSplittings(
 const JetSubstructureSplittings& other)
 : fJetSplittings{other.fJetSplittings},
  fSubjets{other.fSubjets}
{
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
JetSubstructureSplittings& JetSubstructureSplittings::operator=(
 JetSubstructureSplittings other)
{
  swap(*this, other);
  return *this;
}

bool JetSubstructureSplittings::Clear()
{
  fJetSplittings.Clear();
  fSubjets.Clear();
  return true;
}

/**
 * Add a jet splitting to the object.
 *
 * @param[in] kt Kt of the splitting.
 * @param[in] deltaR Delta R between the subjets.
 * @param[in] z Momentum sharing between the subjets.
 */
void JetSubstructureSplittings::AddSplitting(float kt, float deltaR, float z, short parentIndex)
{
  fJetSplittings.AddSplitting(kt, deltaR, z, parentIndex);
}

/**
 * Add a subjet to the object.
 *
 * @param[in] part Constituent to be added.
 */
void JetSubstructureSplittings::AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
                     const std::vector<unsigned short>& constituentIndices)
{
  return fSubjets.AddSubjet(splittingNodeIndex, partOfIterativeSplitting, constituentIndices);
}

std::tuple<float, float, float, short> JetSubstructureSplittings::GetSplitting(int i) const
{
  return fJetSplittings.GetSplitting(i);
}

std::tuple<unsigned short, bool, const std::vector<unsigned short>> JetSubstructureSplittings::GetSubjet(int i) const
{
  return fSubjets.GetSubjet(i);
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string JetSubstructureSplittings::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Splitting information: ";
  tempSS << fSubjets;
  tempSS << fJetSplittings;
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * JetSubstructureSplittings::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& JetSubstructureSplittings::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

} /* namespace JetSubstructure */

void ExtractJetSplittings(
  JetSubstructure::JetSubstructureSplittings & jetSplittings,
  fastjet::PseudoJet & inputJet,
  int splittingNodeIndex,
  bool followingIterativeSplitting,
  const bool storeRecursiveSplittings
)
{
    fastjet::PseudoJet j1;
    fastjet::PseudoJet j2;
    if (inputJet.has_parents(j1, j2) == false) {
        // No parents, so we're done - just return.
        return;
    }

    // j1 should always be the harder of the two subjets.
    if (j1.perp() < j2.perp()) {
        std::swap(j1, j2);
    }

    // We have a splitting. Record the properties.
    double z = j2.perp() / (j2.perp() + j1.perp());
    double delta_R = j1.delta_R(j2);
    double xkt = j2.perp() * sin(delta_R);
    // Add the splitting node.
    jetSplittings.AddSplitting(xkt, delta_R, z, splittingNodeIndex);
    // Determine which splitting parent the subjets will point to (ie. the one that
    // we just stored). It's stored at the end of the splittings array. (which we offset
    // by -1 to stay within the array).
    splittingNodeIndex = jetSplittings.GetNumberOfSplittings() - 1;
    // Store the subjets
    std::vector<unsigned short> j1ConstituentIndices, j2ConstituentIndices;
    for (auto constituent: j1.constituents()) {
        j1ConstituentIndices.emplace_back(constituent.user_index());
    }
    for (auto constituent: j2.constituents()) {
        j2ConstituentIndices.emplace_back(constituent.user_index());
    }
    jetSplittings.AddSubjet(splittingNodeIndex, followingIterativeSplitting, j1ConstituentIndices);
    jetSplittings.AddSubjet(splittingNodeIndex, false, j2ConstituentIndices);

    // Recurse as necessary to get the rest of the splittings.
    ExtractJetSplittings(jetSplittings, j1, splittingNodeIndex, followingIterativeSplitting, storeRecursiveSplittings);
    if (storeRecursiveSplittings == true) {
        ExtractJetSplittings(jetSplittings, j2, splittingNodeIndex, false, storeRecursiveSplittings);
    }
}

} // namespace mammoth

/**
 * Subjets
 */

/**
 * Implementation of the output stream operator for JetSubstructure::Subjets. Printing
 * basic task information provided by function toString
 * @param in output stream
 * @param myTask Task which will be printed
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::Subjets& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}

/**
 * Swap function. Created using guide described here: https://stackoverflow.com/a/3279550.
 */
void swap(mammoth::JetSubstructure::Subjets& first,
     mammoth::JetSubstructure::Subjets& second)
{
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fSplittingNodeIndex, second.fSplittingNodeIndex);
  swap(first.fPartOfIterativeSplitting, second.fPartOfIterativeSplitting);
  swap(first.fConstituentIndices, second.fConstituentIndices);
}

/**
 * JetSplittings
 */

/**
 * Implementation of the output stream operator for JetSubstructure::JetSplittings. Printing
 * basic task information provided by function toString
 * @param in output stream
 * @param myTask Task which will be printed
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSplittings& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}

/**
 * Swap function. Created using guide described here: https://stackoverflow.com/a/3279550.
 */
void swap(mammoth::JetSubstructure::JetSplittings& first,
     mammoth::JetSubstructure::JetSplittings& second)
{
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fKt, second.fKt);
  swap(first.fDeltaR, second.fDeltaR);
  swap(first.fZ, second.fZ);
  swap(first.fParentIndex, second.fParentIndex);
}

/**
 * Jet substructure splittings
 */

/**
 * Implementation of the output stream operator for JetSubstructureSplittings. Printing
 * basic task information provided by function toString
 * @param in output stream
 * @param myTask Task which will be printed
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSubstructureSplittings& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}

/**
 * Swap function. Created using guide described here: https://stackoverflow.com/a/3279550.
 */
void swap(mammoth::JetSubstructure::JetSubstructureSplittings& first,
     mammoth::JetSubstructure::JetSubstructureSplittings& second)
{
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fJetSplittings, second.fJetSplittings);
  swap(first.fSubjets, second.fSubjets);
}
