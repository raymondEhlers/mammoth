# Entanglement entropy analysis

Also referred to as the Entropy-Based Correlators (EBC) analysis.

# Analysis code

The general structure of the analysis is:

- Analyze chunks into a dijet skim. This is covered in `analyze_chunk`
  - As of 2026 April, RJE only implemented the one level analysis for e.g. pp data or a single level (part or det) of pythia. Proper support for two levels (e.g. for det-part level pp_MC support) needs to be implemented.
- Further calculate the entropy and mutual information from the dijet skim. This is covered by `skim_to_hist` and `analysis_tools`
  - This part of the analysis is run through the notebooks in the [`projects/entanglement_entropy` directory](https://github.com/raymondEhlers/mammoth/tree/main/projects/entanglement_entropy)

# Running the analysis

For general information on running analyses, see [docs/running_an_analysis.md](https://github.com/raymondEhlers/mammoth/tree/main/docs/running_an_analysis.md).

This chunk analysis is run via `run_analysis` module in this directory - i.e. `python3 -m mammoth.entanglement_entropy.run_analysis`.
Configure the analysis via:

- Run / execution options in the `main(...)` function
- Run with parsl or dask in the call to `main(...)`
- Configure the analysis parameters in the 'track skim config' yaml in the `src/mammoth/alice/config` directory

The calculation of the entropy and mutual information is run through the notebooks in the [`projects/entanglement_entropy` directory](https://github.com/raymondEhlers/mammoth/tree/main/projects/entanglement_entropy).

## Analysis runs:

As usual, all analyses are listed in the track skim. The particular ones of note as of 2026 April are (including a storage location on the LBL hiccup cluster):

- pp
  - 0004: Run 3, 13.6 TeV, LHC22o test production. Histograms only (no dijet skim), partial statistics.
    - Stored at: `/rstorage/rehlers/trains/pp/0004/skim`
  - 0008: Run 3, 5.36 TeV, LHC24 pp ref test production. Dijet skim. Lost a large number of jobs due to running out of memory, but still dramatically more statistics than in Run 2. Used for studies presented in April 2026.
    - This should be rerun as a proper production so we get the full statistics. IMPORTANT: Use a smaller chunk size so we don't lose jobs.
    - Stored at: `/rstorage/rehlers/trains/pp/0008/skim`
  - 0063: Run 2, 5.02 TeV, LHC17pq pp ref production. Dijet skim. Statistics are limited because it's run 2 pp, but gave the opportunity to assess the impact of statistics, as well as investigate sqrt_s dependence.
    - Stored at: `/rstorage/rehlers/trains/pp/0063/skim`
- pp_MC:
  - 86: Run 2, 5.02 TeV, LHC18b8 ALICE pythia production. Dijet skim (particle level only). The goal was to investigate what it looks like at particle level. Was never really checked closely as of April 2026 (since studies could still be done on data).
    - Stored at: `/rstorage/rehlers/trains/pp_MC/0086/skim`

Details on how the ALICE track skims are prepared is [available here in the guide on running analyses](https://github.com/raymondEhlers/mammoth/tree/main/docs/running_an_analysis.md). To get access to the stored files on hiccup, please contact Peter.
