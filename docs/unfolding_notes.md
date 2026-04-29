# Notes and pointers on unfolding

I originally prepared these brief notes on unfolding for A. Takacs. It's a bit oriented towards theorists, but should generally be useful for eg. ALICE folks, newer students, etc. For fully new students, more detail is needed, but this is enough to give some pointers on unfolding techniques.

- For general unfolding, I think [these slides are quite good](https://www.desy.de/~sschmitt/talks/UnfoldStatSchool2014.pdf). It's focused on HEP rather then heavy-ions, so I don't think the background advice is necessarily so useful, but overall, I think it is quite good.
  - Especially useful for comparing bin-by-bin vs full unfolding
- As a very first introduction, the RooUnfold tutorials aren't terrible. Not especially well documented, but they provide some nice working code to play with.
- For dealing with background, I don't have any particularly good written up resources, but here are some details:
  - Baseline ALICE paper on [background fluctuations and subtraction](https://arxiv.org/abs/1201.2423). This forms the basis of the STAR and ALICE "standard" area-based subtraction - eg. pt_measured_including_bg - \rho \* A . The `fastjet::JetMediuanBackgroundEstimator` works fine for this. We generally exclude the two leading jets when calculating $\rho$.
  - We usually subtract for substructure using [Constituent Subtraction](https://arxiv.org/abs/1403.3108). You can just use the fjcontrib implementation. Can either be event-wise (subtract before jet finding) or jet-wise (subtract after jet finding). We usually use event-wise because it tends to be somewhat less biased, which means our unfolding is easier. Common parameter are R_max and alpha. Usually we take alpha=0 and R_max depends on the jet R. Often 0.25 for R = 0.4 and 0.1 for R = 0.2 . Note that if you apply it event-wise, you don't need to do any additional \rho \* A subtraction - the pt is already subtracted in this case.
  - ATLAS and CMS, and now sPHENIX do some subtraction at their calorimeter level. In my understand, it's not so dissimilar for the rho \* A subtraction above, but it's applied at the calorimeter level instead, and they use some sort of $(\eta, \varphi)$ phasespace dependent $\rho$. ALICE can get away without doing that because our acceptance is smaller. I'm not an expert on this though
- For dealing with the actual unfolding, you just need to construct your response between particle level and measured level (what we in ALICE often call "hybrid level". For your purposes, most likely the particle level combined with some background that you generate, since I'm assuming you're not running GEANT or anything). So explicitly for an example of Rg, you need (pt_part, Rg_part, pt_measured_bg_sub, Rg_measured)
  - Usual binning caveats apply (eg. you can unfold more finely binned at particle level than you are at measured level)
  - You'll want to construct your particle level binning range to be larger than the our measured level. Eg: if measured is [1, 2, 3, 4], then part should be: [0, 1, 2, 3, 4, 6]. You want to cover regions that could reasonably migrate into your measured so the unfolding can do that migration.
  - Bins with 0 counts will wreck your unfolding. Avoid them or adjust your binning.
  - If there's a ton of background dominated contributions in the response, you may need to eg. cut that region out. Off diagonal components will also often wreck your unfolding.
- You can try filling the `Miss` on your response object when it's outside of the phase space that you want to accept. However, I know there have historically been some issues with it tracking entries correctly. It may work perfectly fine, but I haven't tried it. Instead, we usually just keep track of our efficiency manually by keeping two hists: the full efficiency hist, which is filled before any kinematic selections, and then the phase space selected hist. The ratio of those in the phase space of interest corrects for these misses. (at least this is how I understand the `Miss` functionality).

  - Just to be explicit, I mean:

  ```python
  for t in trees:
      # Define hists
      define_hists(f)
      # ...

      # Cuts at particle  level (ie. defining the possible region of your measurements. eg. No jets above 300 GeV at particle level will contribute to a 60 GeV measured jet.)
      # ...

      # Fill full efficiency hists
      # ...

      # Cuts at measured level (ie. you want to consider 40-120 GeV measured jets)
      # ...

      # Fill hists + response object
      # ....
  ```

  - If your efficiency correction is huge, it will make experimentalists nervous since the more you correct, the more model dependence you include. You can get out of this if you can show that the trend is model independent, but that can be trickier.

- Convergence is usually determined by looking at the number of iterations and closure checks (see next). Generally, you want the movement of the next iteration to be small compared to the previous. Pick too high, and you'll explode the stat uncertainty (it inflates due to the regularization of the matrix inversion, and will tend to oscillate).
- The last bit is the closure tests. Usually the most trivial tests we do are just dividing the dataset into two halves and using one to construct the response and the other as the input spectrum. After unfolding, we should recover the particle level spectra within uncertainties. You can also refold to double check that you get what you put in. If not, you may have a technical error. If in general you can't get convergence, it can indicate that you just can't unfold that observable.
  - Further steps would be eg. check that changing the shape of the input spectra while keeping the same response still converges.

## Some additional resources

- [A multidimensional unfolding method based on Bayes' theorem](https://cds.cern.ch/record/265717?ln=en)
- [A theorists on unfolding in heavy-ion: High-dimensional unfolding in Large Backgrounds](http://arxiv.org/abs/2507.06291) (n.b. related to why I originally prepared this note)
