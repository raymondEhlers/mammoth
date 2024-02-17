# Mammoth

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

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


## Installation

Easiest is to use `pdm`, which will handle dependencies correctly. Just install with `pdm install -d` to include the dev dependencies.

<!-- insert detail collapsible block -->


<details>
 <summary>Old `pip` method: Install for other projects</summary>

> First, remove `pachyderm` from the dependencies, because it probably won't be picked up correctly by pip.
> Next, run
>
> ```bash
> # Actually build the extensions...
> $ pip install --use-feature=in-tree-build ../mammoth
>
> # We've built in the tree, so now we need to do an editable install so it can find the extensions...
> $ pip install -e ../mammoth
> ```

</details>
