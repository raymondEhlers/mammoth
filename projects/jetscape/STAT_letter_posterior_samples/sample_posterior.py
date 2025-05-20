# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv-3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sample STAT letter posterior distribution
#
# Luna provided samples of the posterior distribution. I want to sample that to draw the e.g. most probable values, or a bound around the MAP. (The latter is for the HetGP paper)

# %%
from __future__ import annotations

from pathlib import Path

import polars as pl

# %%
lf = pl.read_csv(
    Path(
        "projects/jetscape/STAT_letter_posterior_samples/STAT20230320ExponentialRBF_N4_MCMCSamples_NominalJetRAAPaper.txt"
    ),
    separator=" ",
)

# %%
lf.describe()

# %%
pl.Config.set_tbl_rows(20)
(
    lf.describe(percentiles=(0.1, 0.25, 0.3333, 0.5, 0.6667, 0.75, 0.9))
    .filter(pl.col("statistic").is_in(["33.33%", "50%", "66.67%"]))
    .write_csv(Path("projects/jetscape/STAT_letter_posterior_samples/samples_thirds.csv"))
)

# %%
# ! head -n 5 projects/jetscape/STAT_letter_posterior_samples/samples_thirds.csv

# %%
