# Mammoth

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/raymondEhlers/mammoth/workflows/CI/badge.svg
[actions-link]:             https://github.com/raymondEhlers/mammoth/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/mammoth
[conda-link]:               https://github.com/conda-forge/mammoth-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/raymondEhlers/mammoth/discussions
[pypi-link]:                https://pypi.org/project/mammoth/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/mammoth
[pypi-version]:             https://img.shields.io/pypi/v/mammoth
[rtd-badge]:                https://readthedocs.org/projects/mammoth/badge/?version=latest
[rtd-link]:                 https://mammoth.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

Mammoth is a package for analysis and interpretation of heavy-ion collision data.
Analyses are performed using a (predominately) columnar based paradigm.
The main analysis framework contains a collection of python based functionality, as well as c++ code bound to python (e.g. for jet finding via fastjet).
I/O adapters for a variety of formats are implemented, enabling analysis of real measure data, as well as outputs from Monte Carlo generators.

This package has been used for a variety of analyses, and the framework is fully validated to reproduce e.g. the ALICE analysis framework.
For example, mammoth was used for the measurement recently released from the ALICE collaboration to [search for quasi-particle (i.e. Moliere) scattering using jet substructure](https://arxiv.org/abs/2409.12837).

<p align="center"><img alt="Figure from ALICE quasi-particle scattering paper" src="docs/images/unfolded_kt_pp_PbPb_ratios_only_R02_dynamical_kt_soft_drop_z_cut_02.png" width="70%" /></p>

> [!IMPORTANT]
> This package is developed for my own use. Framework and code is subject to change, and documentation is quite limited!

## Installation

Easiest is to use `pdm`, which will handle dependencies correctly (the lock file is committed in the repo). Just install with `pdm install -d` to include the dev dependencies.

<!-- insert detail collapsible block -->

<details>
 <summary>Old `pip` method: Now used to install mammoth for other projects</summary>

> In practice, getting the right compiled bindings was initially tricky.
> I think this is less critical in 2024 after I moved to the cookie-cutter repo, but I haven't tested it again recently, so I maintain this information for now
>
> First, remove `pachyderm` from the dependencies, because it's usually installed locally and it most likely won't be picked up correctly by pip.
> Next, run
>
> ```bash
> # Actually build the compiled binding extensions...
> $ pip install --use-feature=in-tree-build ../mammoth
>
> # We've built in the tree, so now we need to do an editable install so it can find the extensions...
> $ pip install -e ../mammoth
> ```

</details>

## Helpers with vscode

To get the CMake plugin for vscode to play nicely with this package, I explored a few options:

- [Manually configure CMake in vscode](#manually-configure-vscode) as necessary. It takes a little work
  and isn't especially robust, but isn't terrible.
- Create a new venv via `nox`, adapted from [here](https://github.com/scikit-build/scikit-build-core/issues/635).
  This isn't the nicest because it can mix python versions, etc. It may still work, but use with care!

### Manually configure vscode

You need to add these values to your `.vscode/settings.json`:

```json
    "cmake.sourceDirectory": "${workspaceFolder}/mammoth-cpp",
    "cmake.configureSettings": {
        "CMAKE_PREFIX_PATH":"${workspaceRoot}/.venv-3.11/lib/python3.11/site-packages/pybind11/share/cmake/pybind11",
    },
    "cmake.configureArgs": [
        "-DSKBUILD_PROJECT_NAME=mammoth-cpp"
    ],
```

You'll have to adapt the virtualenv path as appropriate. This should allow the cmake plugin to configure properly.
