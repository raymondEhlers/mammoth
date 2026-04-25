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
#     display_name: .venv-3.13
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Toy model to test statistics dependence of entanglement entropy
#
# The experiment:
#
# - Underlying distribution will be a Gaussian with central value of 1, width of 0.2
# - Sample n times, where n will vary:
#     - e.g. 20, 100, 1000, 10k, 1000k, 1m, 2m, 3m
# - Calculate unsummed entropy + mutual information
# - Where does the impacts come in?
#
# - Variations:
#     - [ ] Start continuously, and then add binning

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pachyderm.plot as pb
import polars as pl  # noqa: F401
import uproot  # noqa: F401

from mammoth.framework.io import output_utils  # noqa: F401

pb.configure()

# %load_ext autoreload

# %autoreload 2

base_path = Path("projects/entanglement_entropy/statistics_toy")

# %% [markdown]
# ## Sample of the distribution

# %%
rng = np.random.default_rng()

# %%
mean = 1
width = 0.2

# n.b. Need to check whether these sizes are really appropriate...
# Adapt based on how the toy looks
target_size = [20, 100, 500, 1000, 10_000, 100_000, 1_000_000, 2_000_000, 3_000_000]

distribution_samples = {s: rng.normal(mean, width, size=s) for s in target_size}

# %%
distribution_samples[20]

# %%

# %% [markdown]
# ## Note on implementation, 2026-04-25, RJE
#
# I never follow through with the implementation of this because it became clear based on the literature that this was a known issue with measuring the mutual information. So there was no real benefit to going back and implementing a toy to see what we could already see in the data.
#
# As work is done on the correction, it may become more useful to have a simple toy. In that case, this should be straightforward to pick up from here. The biggest difficult is trying to implement a continuous version - the generalization of the entropy to a continuous distribution isn't entirely trivial. Some links that I collected on the topic:
#
# - On entropy of a Gaussian (to compare as an expected value): https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
# - Differential entropy, relating to the contuious vs discreete distributions: https://www.sciencedirect.com/topics/engineering/differential-entropy

# %% [markdown]
#
