"""Generate a chunk of pythia events for a given configuration.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import pythia8mc  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


def setup_pythia() -> pythia8mc.Pythia:
    """Setup pythia to generate events.

    Note:
        This represents the minimal configuration. Later, we may customize this further.

    Returns:
        pythia: The pythia object which can be used to generate events.
    """
    pythia = pythia8mc.Pythia()

    # Parameters (you could make these function arguments)
    # Beam particles
    # **Q**: What does 2212 correspond to? (Hint: Check the Particle Data Group)
    beam_A = 2212
    beam_B = 2212
    # Center of mass energy
    sqrt_s = 14000
    # Momentum transfer range (ie. pt hat range)
    pt_hat_min, pt_hat_max = 80, 120

    # Initial the pythia settings
    # General
    pythia.readString(f"Beams:idA = {beam_A}")
    pythia.readString(f"Beams:idB = {beam_B}")
    pythia.readString(f"Beams:eCM = {sqrt_s}.")
    # Provide a unique seed for each run to ensure the outputs for multiple runs or processes are not the same.
    # **Q**: How can you do this properly?
    # Check the documentation for what is valid!
    random_seed = 0
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {random_seed}")

    # Setup the physics processes
    # Enable all hard momentum transfer QCD processes.
    # **Q**: What does this mean? You don't need to have a precise answer, but it's good to have a general sense.
    pythia.readString("HardQCD:all = on")
    # Set the momentum transfer range
    # **Q**: What would we see if we didn't set this range? (Hint: Think about the cross section at increasing energy)
    pythia.readString(f"PhaseSpace:pTHatMin = {pt_hat_min:.1f}")
    pythia.readString(f"PhaseSpace:pTHatMax = {pt_hat_max:.1f}")

    return pythia


def generate_chunk(
    n_events: int,
    max_rapidity: float,
    jet_R: float,
) -> None:
    """Generate a chunk of pythia events for a given configuration."""

    # First, we need to setup pythia
    pythia = setup_pythia()
    # Complete the pythia setup
    # (Since this effectively starts pythia, it's often good to have more control
    # over when it truly starts.)
    pythia.init()

    # NOTE: It's nicer if I can return a chunk in an awkward array to be written automatically by the framework, but I
    #       also need to keep the pythia object alive to preserve the xsec precision. I'm not quite sure how that works yet.
    #       So for now, I need to think about it. And also how it interacts with the generator objects - are those sources
    #       more appropriate for this? Probably?

    # We'll run our analysis code inside of the event loop.
    for i_event in range(n_events):
        pythia.next()
        logger.debug(f"Generating event {i_event}")

        fj_particles = []
        for pythia_particle in pythia.event:
            # Add some selections to the pythia particles:
            # - Only keep stable ("final state") particles.
            # - Require that that particle "is visible" (ie. interacts via the EM or strong force)
            # - Only select mid-rapidity particles, which we'll define as |eta| < 2
            # - Accept all particles in phi
            # **Q**: How would you implement these selections? Hint: See the PYTHIA documentation
            if pythia_particle.isFinal() and pythia_particle.isVisible() and abs(pythia_particle.eta()) < max_rapidity:
                pj = fj.PseudoJet(pythia_particle.px(), pythia_particle.py(), pythia_particle.pz(), pythia_particle.e())
                fj_particles.append(pj)

            # For all particles you keep, you should convert them into a suitable type for fastjet.
            # This means you need to convert them into a PseudoJet. You can store them in the `fj_particles` list.

        # Create the jet finder, known as the `Clustering Sequence`
        # **Q**: What do all of the arguments mean here? You don't need to understand in great detail,
        #           but it's good to have a general idea.
        cluster_sequence = fj.ClusterSequenceArea(fj_particles, jet_definition, area_definition)

        # Apply some selections to the jets
        jets = selector(cluster_sequence.inclusive_jets())

        # Once you have the jets, you can fill them into a histogram.
        # As a first example, we can fill the jet pt (which you defined above)
        # NOTE: This is not especially efficient in python, but it's a good starting point.
        #       The most efficiency approach would be via array-wise operations as done in
        #       eg. numpy, but that's a more advanced topic for another time.
        for j in jets:
            hist_jet_pt.fill(j.pt())
