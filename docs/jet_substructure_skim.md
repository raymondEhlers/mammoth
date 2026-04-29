# Jet substructure skim format

To facilitate reclustered jet substructure analyses, I developed a flat, array-based representation of the reclusive jet splitting. This format is documented below in case it needs to reproduced or understood separately. Of particular interest is how the index references between those arrays work — is essential for any analysis that reads this format.

This jet splitting tree is then further analyzed in the `analyze_chunk_to_groomed_flat_tree` module to produce a fully flat tree (e.g. an n-tuple), which is trivial to then extract measured quantities (all substructure properties are already calculated).

> [!note]
> Even though we've moved to the track skims, I still use the format internally to do the reclustering. So it's important even if the original source of the output files (generated via `AliAnalysisTaskJetDynamicalGrooming` in AliPhysics) is not used anymore.

> [!tip]
> I used Claude to generate the first draft of the documentation. I've read through it myself, so it should be correct, but read this with a critical eye.

# Overview

Jet reclustering (such as Cambridge/Aachen) produces a binary splitting tree. At each node (splitting), a jet or subjet is decomposed into two subjets: the harder one (higher $p_T$, sometimes called `j1`) and the softer one (sometimes called `j2`). This tree is recorded by flattening it into three parallel arrays stored per jet:

- `jet_constituents`: All particles in the jet
- `jet_splittings`: All splittings (nodes in the binary tree), in depth-first order following the harder branch
- `subjets`: The two subjets (harder and softer) produced at each splitting

The arrays within one jet event are tied together by integer index references rather than by nesting, which is what makes the format "flat." Reconstructing the tree means following those indices.

# The three arrays

## jet_constituents

A flat list of all particles (tracks) in the jet. Each constituent carries kinematic properties:

| Field | C++ branch                                 | Description                                                                                  |
| ----- | ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `pt`  | `fJetConstituents.fPt`                     | Transverse momentum                                                                          |
| `eta` | `fJetConstituents.fEta`                    | Pseudorapidity                                                                               |
| `phi` | `fJetConstituents.fPhi`                    | Azimuthal angle                                                                              |
| `id`  | `fJetConstituents.fID` (or `fGlobalIndex`) | MC label (`GetLabel()`) for MC, or global track index (with an offset of 2,000,000) for data |

The position of a constituent in this array is its **constituent index**. Subjets reference their particles via these indices.

## jet_splittings

A flat list of all observed splittings, ordered by a depth-first traversal of the splitting tree with the harder branch visited first. Each splitting records the kinematic properties of the declustering at that node:

| Field          | C++ branch                    | Description                                                                                        |
| -------------- | ----------------------------- | -------------------------------------------------------------------------------------------------- |
| `kt`           | `fJetSplittings.fKt`          | Relative transverse momentum: $k_T = p_{T,\text{softer}} \sin(\Delta R)$                           |
| `delta_R`      | `fJetSplittings.fDeltaR`      | Angular separation between the two subjets                                                         |
| `z`            | `fJetSplittings.fZ`           | Momentum sharing fraction: $z = p_{T,\text{softer}} / (p_{T,\text{softer}} + p_{T,\text{harder}})$ |
| `parent_index` | `fJetSplittings.fParentIndex` | Index (into `jet_splittings`) of the parent splitting — see [Index mapping](#index-mapping)        |

The position of a splitting in this array is its **splitting index**.

> [!note]
> The `parent_index` of the very first splitting (the root declustering of the full jet) is `-1`, which serves as a sentinel indicating there is no parent.

## subjets

A flat list of subjets, with exactly two subjets recorded per splitting: the harder subjet first, then the softer subjet. Each subjet carries:

| Field                         | C++ branch                           | Description                                                                              |
| ----------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `parent_splitting_index`      | `fSubjets.fSplittingNodeIndex`       | Index (into `jet_splittings`) of the splitting that produced this subjet                 |
| `part_of_iterative_splitting` | `fSubjets.fPartOfIterativeSplitting` | `True` if this subjet is the harder product and its branch is being followed recursively |
| `constituent_indices`         | `fSubjets.fConstituentIndices`       | Indices (into `jet_constituents`) of the particles belonging to this subjet              |

# Index mapping

This is the core of the format. The three arrays are linked to each other through integer indices.

## parent_index in jet_splittings

`JetSplitting.parent_index` answers the question: _which earlier splitting produced the subjet that was declustered here?_

The declustering is recursive. When a splitting at index `i` produces harder subjet `j1`, that subjet may itself be declustered. The resulting new splitting is assigned `parent_index = i`. This means you can reconstruct the lineage of any splitting by following `parent_index` references up toward `-1` (the root).

> [!tip]
> Because the array is stored in depth-first order (harder branch first), if the softer branch is also recursed into (controlled by `fStoreRecursiveSplittings` in the C++ task), its splittings will appear later in the array, also with `parent_index` pointing back to the splitting that produced them. I nearly always stored the recursive splittings.

## parent_splitting_index in subjets

`Subjet.parent_splitting_index` answers the question: _which splitting produced this subjet?_

Both the harder and softer subjet at a given declustering share the same `parent_splitting_index`, pointing to the same entry in `jet_splittings`. This is the reverse of the relationship described above: given a splitting index `i`, you can find both of its product subjets by looking for all entries in `subjets` where `parent_splitting_index == i`.

## constituent_indices in subjets

`Subjet.constituent_indices` is a list of integer indices into `jet_constituents`. Indexing `jet_constituents` with this list gives the particles belonging to that subjet.

```python
# Retrieve the particles in the first subjet of a jet
particles = jet["jet_constituents"][jet["subjets"][0].constituent_indices]
```

Note that the constituents of a subjet are always a subset of the constituents of its parent subjet (and ultimately of the full jet). The C++ code assigns `user_index()` values from FastJet to constituents before clustering, and those become the indices stored here.

## part_of_iterative_splitting

`Subjet.part_of_iterative_splitting` marks whether a subjet is on the **iterative splitting chain**: the sequence of harder subjets obtained by always following the harder branch at each declustering. This flag is `True` for each harder subjet (`j1`) as long as the recursion is following the harder branch, and `False` for every softer subjet (`j2`).

> [!definition] Iterative splitting
> The iterative splitting is the sequence of splittings obtained by repeatedly taking the harder subjet at each declustering. It defines the "primary" path through the splitting tree from the full jet down to a single particle.

The `iterative_splitting_index` property on `SubjetArray` returns the splitting indices that form the iterative chain:

```python
# Splitting indices on the harder-branch chain
iterative_indices = jet["subjets"].iterative_splitting_index
# → e.g. [0, 1, 2] for a jet with three successive harder-branch splittings

# The actual splittings on that chain
iterative_splittings = jet["jet_splittings"].iterative_splittings(jet["subjets"])
```

# Concrete example

Consider a jet with five constituents (indexed 0–4) that undergoes three splittings.

```
Full jet
    │
    └─ Splitting 0 (root, parent_index = -1)
           j1 (harder): constituents [0,1,2,3]   part_of_iterative_splitting = True
           j2 (softer): constituents [4]          part_of_iterative_splitting = False

        └─ j1 is declustered further →
               Splitting 1 (parent_index = 0)
                   j1 (harder): constituents [0,1]    part_of_iterative_splitting = True
                   j2 (softer): constituents [2,3]    part_of_iterative_splitting = False

               └─ j1 is declustered further →
                      Splitting 2 (parent_index = 1)
                          j1 (harder): constituents [0]  part_of_iterative_splitting = True
                          j2 (softer): constituents [1]  part_of_iterative_splitting = False
```

The resulting arrays look like this:

**jet_splittings** (one row per splitting, in DFS order):

| index | kt  | delta_R | z   | parent_index |
| ----- | --- | ------- | --- | ------------ |
| 0     | …   | …       | …   | -1           |
| 1     | …   | …       | …   | 0            |
| 2     | …   | …       | …   | 1            |

**subjets** (two rows per splitting, harder then softer):

| index | parent_splitting_index | part_of_iterative_splitting | constituent_indices |
| ----- | ---------------------- | --------------------------- | ------------------- |
| 0     | 0                      | True                        | [0, 1, 2, 3]        |
| 1     | 0                      | False                       | [4]                 |
| 2     | 1                      | True                        | [0, 1]              |
| 3     | 1                      | False                       | [2, 3]              |
| 4     | 2                      | True                        | [0]                 |
| 5     | 2                      | False                       | [1]                 |

The iterative splitting chain consists of splittings 0, 1, and 2 (all of them in this example since the softer branch is never stored unless `fStoreRecursiveSplittings` is enabled). This is recovered by:

```python
subjets.iterative_splitting_index = parent_splitting_index[part_of_iterative_splitting]
# = [0, 1, 2] (the parent_splitting_index values for the three True entries)
```

# Python API

The behaviors defined in [jet_substructure.py](https://github.com/raymondEhlers/mammoth/blob/main/src/mammoth/framework/analysis/jet_substructure.py) expose methods directly on the awkward arrays.

## Loading the data

> [!warning]
> This describes the process of loading the output from `AliAnalysisTaskJetDynamicalGrooming`. This part is not used directly in the track skim. Instead, you should reference the reclustered substructure module.

Data is loaded from a parquet file (converted from the original ROOT TTree) via `parquet_to_substructure_analysis`. The result is a dict keyed by prefix (e.g. `"hybrid_level"`, `"det_level"`, `"true"`), where each value is an array of jets with the three sub-arrays attached:

```python
from mammoth.framework.analysis import jet_substructure as subs
from pathlib import Path

arrays = subs.parquet_to_substructure_analysis(
    filename=Path("AnalysisResults.parquet"),
    prefixes={"hybrid_level": "data", "det_level": "detLevel", "true": "matched"},
)
jets = arrays["hybrid_level"]

# Access sub-arrays for a single jet
jet = jets[0]
jet["jet_pt"]  # scalar: jet pT
jet["jet_constituents"]  # JetConstituentArray
jet["jet_splittings"]  # JetSplittingArray
jet["subjets"]  # SubjetArray
```

## Common operations

```python
splittings = jet["jet_splittings"]
subjets = jet["subjets"]

# Derived kinematic properties of a splitting
splittings.parent_pt  # pt of the parent subjet: kt / sin(delta_R) / z
splittings.theta(R=0.4)  # delta_R / R

# Retrieve only the iterative (harder-branch) splittings
iterative = splittings.iterative_splittings(subjets)

# Soft drop: first splitting with z > z_cut
z_g, index, all_passing = splittings.soft_drop(z_cutoff=0.2)

# Dynamical grooming measures
dcore, idx, _ = splittings.dynamical_core(R=0.4)
dkt, idx, _ = splittings.dynamical_kt(R=0.4)

# Access constituents of a particular subjet
subjet = subjets[0]
particles = jet["jet_constituents"][subjet.constituent_indices]

# Navigate to the splitting that produced a given subjet
parent_splitting = subjet.parent_splitting(splittings)
```

# Branch names in the ROOT / parquet files

The C++ task writes the data under prefixes (`data`, `detLevel`, `matched`). The full branch names follow this pattern:

```
{prefix}.fJetPt
{prefix}.fJetConstituents.fPt
{prefix}.fJetConstituents.fEta
{prefix}.fJetConstituents.fPhi
{prefix}.fJetConstituents.fID         # or fGlobalIndex for older files
{prefix}.fJetSplittings.fKt
{prefix}.fJetSplittings.fDeltaR
{prefix}.fJetSplittings.fZ
{prefix}.fJetSplittings.fParentIndex
{prefix}.fSubjets.fPartOfIterativeSplitting
{prefix}.fSubjets.fSplittingNodeIndex
{prefix}.fSubjets.fConstituentIndices
```

Additional event-level branches are also stored:

```
pt_hard_bin
pt_hard
```

> [!note]
> For data, a `data_leading_track_pt` branch may also be present. It is handled automatically by `parquet_to_substructure_analysis`.
