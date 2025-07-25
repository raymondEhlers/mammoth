# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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

# %% [markdown]
# # 33-66%

# %%
pl.Config.set_tbl_rows(20)
(
    lf.describe(percentiles=(0.1, 0.25, 0.3333, 0.5, 0.6667, 0.75, 0.9))
    .filter(pl.col("statistic").is_in(["33.33%", "50%", "66.67%"]))
    .write_csv(Path("projects/jetscape/STAT_letter_posterior_samples/sample_posterior_thirds.csv"))
)


# %%
# ! head -n 5 projects/jetscape/STAT_letter_posterior_samples/sample_posterior_thirds.csv

# %% [markdown]
# # 90%

# %%
pl.Config.set_tbl_rows(20)
(
    lf.describe(percentiles=(0.05, 0.1, 0.5, 0.9, 0.95))
    .filter(pl.col("statistic").is_in(["5%", "10%", "50%", "90%", "95%"]))
    .write_csv(Path("projects/jetscape/STAT_letter_posterior_samples/sample_posterior_90_percent.csv"))
)

# %%
# ! head -n 6 projects/jetscape/STAT_letter_posterior_samples/sample_posterior_90_percent.csv

# %% [markdown]
# # More ranges for discussion

# %%
pl.Config.set_tbl_rows(20)
(
    lf.describe(percentiles=(0.001, 0.01, 0.05, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.9, 0.95, 0.99, 0.999))
    # .filter(pl.col("statistic"))
    .write_csv(Path("projects/jetscape/STAT_letter_posterior_samples/sample_posterior_explore_range.csv"))
)

# %%
# ! head -n 20 projects/jetscape/STAT_letter_posterior_samples/sample_posterior_explore_range.csv

# %%
