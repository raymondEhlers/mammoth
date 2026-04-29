# Mammoth documentation

This is the main source of documentation for the Mammoth package.
Much of the documentation is included in the source files themselves. However, there is some dedicated documentation on:

- [Developing an analysis](developing_an_analysis.md) using the Mammoth analysis framework
- [Running an analysis](running_an_analysis.md) within the Mammoth analysis framework
- [The reclustered jet substructure / jet splitting skim format](jet_substructure_skim.md). n.b.: This is different from the track skim format!
- [Notes on unfolding](unfolding_notes.md)
- [(Brief) FAQ](FAQ.md)

Further information the repository structure is included below:

# Repository structure

The package is structured as follows:

### [`src/mammoth`](https://github.com/raymondEhlers/mammoth/tree/main/src/mammoth)

Main source directory for the package. Each directory focuses on a particular analysis or project, including all of the relevant code to both implement and run the analysis. Documentation on each analysis tends to be kept in these directories instead of in the project directories.

#### [`src/mammoth/framework`](https://github.com/raymondEhlers/mammoth/tree/main/src/mammoth/framework)

This is the key functionality of the mammoth package - it's a general framework for steering physics analysis tasks.
It includes capabilities to work with different job submission managers (parsl and dask).
Most of the code itself is reasonably well documented, at least at the function level.
See the decided documentation on how to [develop an analysis](developing_an_analysis.md) and [run an analysis](running_an_analysis.md).

### [`projects/`](https://github.com/raymondEhlers/mammoth/tree/main/projects)

Each directory corresponds to an analysis or project.
Generally speaking, these directories contain support files for each project, such as jupyter notebooks.
There are no definitive criteria, but think things that don't belong directly in the codebase.

Your best bet is to browse through the directories to see if there is content of interest to you.
For each analysis, these project dirs are best viewed in concert with the documentation stored in the analysis source code.

#### Jupyter notebooks

Jupyter notebooks are extremely useful, but aren't so convenient to store in version control.
I work around this by converting the notebooks into `.py` files using [jupytext](https://jupytext.readthedocs.io/en/latest/).
You should be able to convert one of those `.py` files back into a notebook via: `jupytext --sync <name_of_file.py>`.

> [!warning]
> Note that **not** all `.py` files in `projects/` directories are notebooks. To avoid confusion, open the file first to check what it is before proceeding. You can tell it's a notebook by the metadata and extensive use of "%%" to differentiate cell blocks.
