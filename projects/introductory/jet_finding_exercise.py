""" Template for running pythia with jet finding.

Intentionally left as an exercise for the reader to fill in the details. To do this,
you'll need to install the pythia8mc and fastjet packages. You can do this via pip:

```bash
$ pip install pythia8mc fastjet
```

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: ... # Fill in your name here!
"""

from __future__ import annotations

import fastjet as fj  # pyright: ignore[reportMissingImports]

# NOTE: This would be just pythia8 if you compile it yourself. However, it's pythia8mc if you install via PyPI (ie. pip)
import pythia8mc  # pyright: ignore[reportMissingImports]


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
    random_seed = ...
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


def setup_jet_finding_settings() -> tuple[fj.JetDefinition, fj.AreaDefinition, fj.Selector]:
    """Setup the jet finder.

    Use the following settings to get started:
    - R = 0.4
    - anti-kt jet clustering algorithm
    - Recombination scheme: E recombination scheme
    - Strategy: the fastjet::Best strategy (or just the default)
    - Area type: active_area

    Returns:
        jet_definition, area_definition: The jet definition and the area definition.
    """
    # Define base parameters. Could make these arguments
    jet_R = 0.4
    clustering_algorithm = ...
    # ...

    # Create the derived fastjet settings
    jet_definition = ...
    area_definition = ...

    return jet_definition, area_definition


def run(n_events: int) -> None:
    """Run pythia and find jets.

    Args:
        n_events: Number of pythia events to generate.
    """
    # First, we need to setup pythia
    pythia = setup_pythia()

    # Setup your jet finder.
    # NOTE: If you can, you may want to move this out of the event loop. It depends on
    #       exactly how the package was coded up.
    jet_definition, area_definition = setup_jet_finding_settings()

    # Define some objects to store the results. Usually, we do this via histograms
    # This can be via the ROOT package (you'll have to set it up separately)
    # or via the `hist` package (scikit-hep/hist on GitHub, installable via pip).
    hist_jet_pt = ...

    # We'll run our analysis code inside of the event loop.
    for i_event in range(n_events):
        pythia.next()
        print(f"Generating event {i_event}")

        fj_particles = []
        for pythia_particle in pythia.event:  # noqa: B007
            # Add some selections to the pythia particles:
            # - Only keep stable ("final state") particles.
            # - Require that that particle "is visible" (ie. interacts via the EM or strong force)
            # - Only select mid-rapidity particles, which we'll define as |eta| < 2
            # - Accept all particles in phi
            # **Q**: How would you implement these selections? Hint: See the PYTHIA documentation
            ...

            # For all particles you keep, you should convert them into a suitable type for fastjet.
            # This means you need to convert them into a PseudoJet. You can store them in the `fj_particles` list.

        # Create the jet finder, known as the `Clustering Sequence`
        # **Q**: What do all of the arguments mean here? You don't need to understand in great detail,
        #           but it's good to have a general idea.
        cluster_sequence = ...

        # Apply some selections to the jets
        # - Remove jets with pt < 10 GeV
        # - Ensure the jets are fully contained within the "fiducial acceptance" that we've defined,
        #   which means that the jet must be within |eta| < (2.0 - R).
        # **Q**: How would you implement these selections? Hint: See the fastjet documentation
        # Advanced **Q**: Why do we want the jet to be fully contained within our acceptance?
        jets = ...

        # Once you have the jets, you can fill them into a histogram.
        # As a first example, we can fill the jet pt (which you defined above)
        # NOTE: This is not especially efficient in python, but it's a good starting point.
        #       The most efficiency approach would be via array-wise operations as done in
        #       eg. numpy, but that's a more advanced topic for another time.
        for j in jets:
            hist_jet_pt.fill(j.pt())

        # Other analysis code on the jets could go here.

    # Since we generated in pt hat bins, our jets occur too often (eg. they're much more likely than
    # the physical cross-section). Fortunately, we can correct for this by using the pythia scaling information:
    # We need to rescale the histograms by the generated cross-section (sigmaGen) over the sum of the weights
    # used in generating the events (weightSum). ie. scale_factor = sigmaGen / weightSum
    # Hint: See the Pythia::Info object
    sigma_gen = ...
    weight_sum = ...
    scale_factor = sigma_gen / weight_sum
    # Apply this scale factor to your histogram.
    # You might also consider saving the unscaled and scaled versions to visually see how different they are.
    # **Q**: What does the size of this number suggest about how rare the process is?

    # Once the event loop is done, you can save the histograms to a file.
    # This can be done via the `uproot` package (scikit-hep/uproot on GitHub, installable via pip).
    # This will write the histograms themselves to a `.root` file, which can be read via uproot or via ROOT itself.
    ...  # noqa: PIE790

    # And now we're all done! Often, we'll print the statistics from the generation
    pythia.stat()
    print("All done!")

    # Since you've completed the generation, you can now go on to plotting the histograms.
    # Do this as your next step, in a separate script.
    # There are many ways to do that. One simple way is via `scikit-hep/hist` or `scikit-hep/mplhep`.
    # Another option is via ROOT, if you're familiar with it.
    # Always remember to label your figures!
    # **Q**: What is the most appropriate way to represent the data in the histogram?
    #   Hint: is it exponential? power law? Could scaling or transforming the axis make it easier to interpret?
    # **Q**: What is the relevant label for the y-axis?


if __name__ == "__main__":
    run(n_events=100)
