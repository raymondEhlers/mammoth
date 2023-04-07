#pragma once

/**
 * Tools for jet finding
 *
 * @author: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
 */

#include <fastjet/PseudoJet.hh>
#include <fastjet/JetDefinition.hh>

#include "jetFinding.hpp"

namespace mammoth {

/**
  * Provide a streamer for printing out PseudoJet properties. Just for convenience.
  */
std::ostream& operator<<(std::ostream& in, const fastjet::PseudoJet & p);

/**
 * @brief Combines particles together, accounting for negative energy holes during the recombination.
 *
 * For use with substructure, etc when working with theory calculations. It propagates the user index when possible.
 * The convention is as follows:
 *   - If particle has positive user_index: +1 (i.e. add its four-vector)
 *   - If particle has negative user_index: -1 (i.e. subtract its four-vector)
 *
 * Original idea + code from Yasuki Tachibana, with some further changes from James Mulligan
 */
class FJNegativeEnergyRecombiner : public fastjet::JetDefinition::Recombiner {
public:
  FJNegativeEnergyRecombiner(const int ui) : _ui(ui) {}

  virtual std::string description() const {
    return "E-scheme Recombiner that checks a flag for a 'negative momentum' particle, and subtracts the 4-momentum in recombinations.";
  }

  virtual void recombine(const fastjet::PseudoJet & particle1,
                         const fastjet::PseudoJet & particle2,
                         fastjet::PseudoJet & combined_particle) const;

private:
  const int _ui;
};


/**
 * @brief Negative energy recombiner
 *
 * Simple wrapper for negative energy recombiner.
 */
struct NegativeEnergyRecombiner : public mammoth::Recombiner {
  const int identifierIndex;

  /**
   * @brief Construct a new Negative Recombiner object
   *
   * This is equivalent to the brace initialization that I usually use, but that doesn't work with pybind11
   * (even though the base class is abstract), so we have to write it by hand.
   *
   * @param identifierIndex Index to identify particles with overall negative energy
   */
  NegativeEnergyRecombiner(const int _identifierIndex):
    identifierIndex(_identifierIndex) {}

  /**
   * @brief Create the recombiner on the stored settings.
   *
   * @return std::unique_ptr<fastjet::Recombiner> The jet recombiner.
   */
  fastjet::JetDefinition::Recombiner* create() const override;

  /**
   * Prints information about the recombiner.
   *
   * @return std::string containing information about the recombiner.
   */
  std::string to_string() const override;
};

std::ostream& operator<<(std::ostream& in, const NegativeEnergyRecombiner & c);

} /* namespace mammoth */
