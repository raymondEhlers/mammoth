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

# %%
"""Simple toy for exploring this measurement.

Generated some examples with Claude for gaining intuition for how this works

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# %% [markdown]
# ## Starting with a super simple example

# %%
# Parameters
n_events = 1_000_000
mean_A, mean_B = 6, 6
sigma_A, sigma_B = 2.5, 2.5
rho = 0.4

# Covariance matrix for a Gaussian model (we'll discretize to integers)
cov = [[sigma_A**2, rho * sigma_A * sigma_B], [rho * sigma_A * sigma_B, sigma_B**2]]

# Generate correlated Gaussian multiplicities and round to integers ≥ 0
rng = np.random.default_rng()
data = rng.multivariate_normal([mean_A, mean_B], cov, size=n_events)
n_A = np.clip(np.round(data[:, 0]), 0, None).astype(int)
n_B = np.clip(np.round(data[:, 1]), 0, None).astype(int)

# Build joint histogram
nA_max, nB_max = n_A.max(), n_B.max()
joint_hist = np.zeros((nA_max + 1, nB_max + 1))
for a, b in zip(n_A, n_B, strict=True):
    joint_hist[a, b] += 1

# Normalize to get probability distribution
P_AB = joint_hist / n_events

# %%
# Compute marginals
P_A = P_AB.sum(axis=1)  # sum over n_B
P_B = P_AB.sum(axis=0)  # sum over n_A
P_A, P_B

# %%
# Compute entropy


def shannon_entropy(P):
    return -np.sum(P[P > 0] * np.log(P[P > 0]))


S_A = shannon_entropy(P_A)
S_B = shannon_entropy(P_B)
S_AB = shannon_entropy(P_AB)
S_A, S_B, S_AB

# %%
# Mutual information
I_AB = S_A + S_B - S_AB
print(f"S_A = {S_A:.3f} nats")
print(f"S_B = {S_B:.3f} nats")
print(f"S_AB = {S_AB:.3f} nats")
print(f"I(A:B) = {I_AB:.3f} nats")

# %%

# %% [markdown]
# ## Something more realistic
#
# Using a Negative Binomial Distribution to model multiplicity with parameters:
#
# pp:
# - n_bar = 6 (average mult)
# - k = 1.8
#
# PbPb:
# - n_bar = 1600 (average mult)
# - k = 400 (e.g. something large)

# %% [markdown]
#

# %%
# Setup distributions
import scipy.special as sp

rng = np.random.default_rng()

# Has issues with large values
# def nbd_pmf(n, mean, k):
#    return sp.gamma(n+k)/(sp.gamma(k)*sp.gamma(n+1)) * (mean/(mean+k))**n * (k/(mean+k))**k


def nbd_pmf(n, mean, k):
    # Use log gamma to avoid overflow
    log_coefficient = sp.gammaln(n + k) - sp.gammaln(k) - sp.gammaln(n + 1)
    log_p = log_coefficient + n * np.log(mean / (mean + k)) + k * np.log(k / (mean + k))
    return np.exp(log_p)


def sample_nbd(mean, k, size):
    # Inverse transform sampling
    n_vals = np.arange(0, int(mean * 5))  # range
    pmf = np.array([nbd_pmf(n, mean, k) for n in n_vals])
    cdf = np.cumsum(pmf)
    r = rng.rand(size)
    return np.searchsorted(cdf, r)


def generate_correlated_nbd(meanA, kA, meanB, kB, rho, n_events):
    cov = [[1.0, rho], [rho, 1.0]]
    gauss = rng.multivariate_normal([0, 0], cov, size=n_events)
    # Map Gaussian quantiles to NBD samples
    ranksA = np.argsort(np.argsort(gauss[:, 0])) / n_events
    ranksB = np.argsort(np.argsort(gauss[:, 1])) / n_events

    # Prepare NBD CDFs
    n_vals_A = np.arange(0, int(meanA * 5))
    pmfA = np.array([nbd_pmf(n, meanA, kA) for n in n_vals_A])
    cdfA = np.cumsum(pmfA)

    n_vals_B = np.arange(0, int(meanB * 5))
    pmfB = np.array([nbd_pmf(n, meanB, kB) for n in n_vals_B])
    cdfB = np.cumsum(pmfB)

    # Map quantiles to multiplicities
    n_A = np.searchsorted(cdfA, ranksA)
    n_B = np.searchsorted(cdfB, ranksB)
    return n_A, n_B


# %%
# Computing mutual information
def shannon_entropy(P):
    return -np.sum(P[P > 0] * np.log(P[P > 0]))


def mutual_information(n_A, n_B):
    nA_max, nB_max = n_A.max(), n_B.max()
    joint_hist = np.zeros((nA_max + 1, nB_max + 1))
    for a, b in zip(n_A, n_B, strict=True):
        joint_hist[a, b] += 1
    P_AB = joint_hist / len(n_A)
    P_A = P_AB.sum(axis=1)
    P_B = P_AB.sum(axis=0)
    S_A = shannon_entropy(P_A)
    S_B = shannon_entropy(P_B)
    S_AB = shannon_entropy(P_AB)
    return S_A, S_B, S_AB, S_A + S_B - S_AB


# %%
def calculate_mutual_info(n_A, n_B):
    S_A, S_B, S_AB, I_AB = mutual_information(n_A, n_B)

    print(f"S_A  = {S_A:.3f} nats")
    print(f"S_B  = {S_B:.3f} nats")
    print(f"S_AB = {S_AB:.3f} nats")
    print(f"I(A:B) = {I_AB:.3f} nats (~{I_AB / np.log(2):.3f} bits)")


# %%
# pp example:
meanA = 6.0  # mean multiplicity in region A
kA = 1.8  # NBD shape parameter for A
meanB = 6.0  # mean multiplicity in region B
kB = 1.8  # NBD shape parameter for B
rho = 0.3  # modest correlation between A and B
n_events = 1_000_000  # number of events to generate

# Generate synthetic multiplicity data
n_A, n_B = generate_correlated_nbd(meanA, kA, meanB, kB, rho, n_events)
print("pp")
print(f"n_bar = {meanA}, k={kA}")
calculate_mutual_info(n_A, n_B)

# %%
# PbPb example:
meanA = 1600.0  # mean multiplicity in region A
kA = 400.0  # NBD shape parameter for A
meanB = 1600.0  # mean multiplicity in region B
kB = 400.0  # NBD shape parameter for B
rho = 0.3  # same correlation coefficient
n_events = 500_000  # fewer events due to large multiplicities

# Generate synthetic multiplicity data
n_A, n_B = generate_correlated_nbd(meanA, kA, meanB, kB, rho, n_events)
print("PbPb")
print(f"n_bar = {meanA}, k={kA}")
calculate_mutual_info(n_A, n_B)

# %%
