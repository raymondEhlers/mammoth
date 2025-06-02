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
> This package is developed for my own use. Framework and code is subject to change, and documentation is limited!

# Installation

To install mammoth, we have to 1) install the dependencies, and then 2) actually install the package.

## Dependencies

Although this is a python package, we also use c++ code, so we need to setup the proper dependencies (it's used for e.g. jet finding, that is then wrapped in python for use). This requires a compiler which supports c++17. We also have some setup steps that ease development. I'll assume you've setup your environment correctly, but you can also [check this quick reference](https://www.rehlers.com/posts/2025/software-2025-edition/) for some pointers.

We'll need to install:

1. [pachyderm](#pachyderm) [python support package useful for development]
2. [FastJet](#fastjet) [c++ package for jet finding]
3. [RooUnfold](#roounfold-optional) [optional - requires ROOT. Probably best to skip unless you know you need it]

### Pachyderm

Pachyderm is a python physics support library that I maintain. When installing mammoth in development mode, pachyderm needs to be available in the `external` directory. You can do this via:

```bash
$ cd external; git clone https://github.com/raymondEhlers/pachyderm.git; cd -
```

### FastJet

FastJet is the standard package for jet finding. It should be automatically installed when you install mammoth (see [below](#installing-mammoth)). However, if that fails for some reason, your first step should be to ensure that fastjet is installed correctly. There's a dedicated script to install it:

```bash
$ ./mammoth-cpp/external/install_fastjet.sh
```

### RooUnfold [Optional]

RooUnfold is used for unfolding. If you're not familiar with this, you can ignore this dependency. RooUnfold requires boost and ROOT to be installed and available in your environment. You can install RooUnfold via:

```bash
$ ./external/install_roounfold.sh
```

## Installing mammoth

The easiest way to install mammoth for development is to use `pdm`. `pdm` is preferred because it handles local dependencies better than most other build managers, which is particularly useful for development. Once you've setup the dependencies, you can install mammoth with:

```bash
$ pdm install -d
```

Note that the `pdm` lock file (`pdm.lock`) is included in the repository, ensuring a reproducible environment.

# Development helpers

## Playing nicely with VSCode

VSCode is a helpful tool, but requires some configuration to make it useful for development. For example, it won't know how to access the pybind11 headers. I document a few possible approaches here to address these kinds of problems - you can try them out to see what works best for you (this has gotten easier with time - I expect the first option will be enough).

- [Manually configure CMake for VSCode](#manually-configure-cmake).
- [Dedicated virtualenv for headers via `nox`](#dedicated-virtualenv-for-headers)

### Manually configure CMake

Here, we'll configure VSCode to pick up some package info (e.g., the `pybind11` headers used for accessing c++ functionality in python), which will make development easier. We'll also configure it to play nicely with `scikit-build-core`. It takes a bit of work and is a bit fragile, but isn't terrible. To set this up, you will need to add these values to your `.vscode/settings.json`:

```json
"cmake.sourceDirectory": "${workspaceFolder}/mammoth-cpp",
"cmake.configureSettings": {
    "CMAKE_PREFIX_PATH":"${workspaceRoot}/.venv-3.13/lib/python3.13/site-packages/pybind11/share/cmake/pybind11",
},
"cmake.configureArgs": [
    "-DSKBUILD_PROJECT_NAME=mammoth-cpp"
],
```

You will need to adjust the virtualenv path in `CMAKE_PREFIX_PATH` (part of `cmake.configureSettings`) as necessary. This should allow the CMake plugin to configure properly[^1]. This is the approach RJE uses as of May 2025.

[^1]: Configuring with CMake used to fail due to missing scikit-build-core variables, but they're now (May 2025) defined in the `CMakeLists.txt` when needed, so this should not be an issue anymore.

### Dedicated virtualenv for headers

Alternatively, you can create a dedicated virtualenv to contain the pybind11 headers. You can do this via `nox` (also useful for testing across python versions). An example is available at [this issue](https://github.com/scikit-build/scikit-build-core/issues/635), which you will have to adapt for this project.
