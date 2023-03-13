#include "mammoth/jetFindingTools.hpp"

namespace mammoth {

void FJNegativeEnergyRecombiner::recombine(const fastjet::PseudoJet &particle1,
                                            const fastjet::PseudoJet &particle2,
                                            fastjet::PseudoJet &combined_particle) const
{
    // Define coefficients with which to combine particles
    // If particle has positive user_index: +1 (i.e. add its four-vector)
    // If particle has negative user_index: -1 (i.e. subtract its four-vector)
    int c1 = 1, c2 = 1;
    if (particle1.user_index() < 0)
    {
        c1 = -1;
    }
    if (particle2.user_index() < 0)
    {
        c2 = -1;
    }

    // Recombine particles
    combined_particle = c1 * particle1 + c2 * particle2;

    // If the combined particle has negative energy, flip the four-vector
    // and assign it a new user index
    if (combined_particle.E() < 0)
    {
        std::cout << "NER: Negative E particle. Previous: 1: " << particle1.user_index() << ", 2: " << particle2.user_index() << "\n";
        combined_particle.set_user_index(_ui);
        combined_particle.reset_momentum(
            -combined_particle.px(), -combined_particle.py(),
            -combined_particle.pz(), -combined_particle.E());
    }
    else
    {
        std::cout << "NER: setting index of combined particle. Previous: 1: " << particle1.user_index() << ", 2: " << particle2.user_index() << "\n";
        combined_particle.set_user_index(0);
    }
}

fastjet::JetDefinition::Recombiner* NegativeEnergyRecombiner::create() const {
    return new FJNegativeEnergyRecombiner(this->identifierIndex);
}

std::string NegativeEnergyRecombiner::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "NegativeEnergyRecombiner(identifierIndex=" << this->identifierIndex
     << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& in, const NegativeEnergyRecombiner & c) {
  in << c.to_string();
  return in;
}

} /* namespace mammoth */