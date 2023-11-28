"""Steering for scale factors

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from typing import Any

from parsl.data_provider.files import File

from mammoth import job_utils
from mammoth.framework import production
from mammoth.framework.analysis import objects as analysis_objects
from mammoth.job_utils import python_app

logger = logging.getLogger(__name__)


@python_app
def _extract_scale_factors_from_hists(
    list_name: str,
    job_framework: job_utils.JobFramework,  # noqa: ARG001
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],  # noqa: ARG001
) -> analysis_objects.ScaleFactor:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same beyond switching to uproot.
    """
    from pathlib import Path

    from mammoth.alice import scale_factors as sf
    from mammoth.framework.analysis import objects as analysis_objects

    res = analysis_objects.ScaleFactor.from_hists(
        *sf.scale_factor_uproot(filenames=[Path(i.filepath) for i in inputs], list_name=list_name)
    )
    return res  # noqa: RET504


def setup_extract_scale_factors(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
) -> dict[int, Future[analysis_objects.ScaleFactor]]:
    """Extract scale factors from embedding or pythia hists.

    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Note:
        This is surprisingly fast.
    """
    # Setup
    scale_factors: dict[int, Future[analysis_objects.ScaleFactor]] = {}
    logger.info("Determining input files for extracting scale factors.")
    input_files_per_pt_hat_bin = prod.input_files_per_pt_hat()

    dataset_key = "signal_dataset" if "signal_dataset" in prod.config["metadata"] else "dataset"
    for pt_hat_bin, input_files in input_files_per_pt_hat_bin.items():
        logger.debug(f"pt_hat_bin: {pt_hat_bin}, filenames: {input_files}")
        if input_files:
            scale_factors[pt_hat_bin] = _extract_scale_factors_from_hists(
                inputs=[File(str(fname)) for fname in input_files],
                list_name=prod.config["metadata"][dataset_key]["list_name"],
                job_framework=job_framework,
            )

    return scale_factors


@python_app
def _write_scale_factors_to_yaml(
    scale_factors: Mapping[int, analysis_objects.ScaleFactor],
    job_framework: job_utils.JobFramework,  # noqa: ARG001
    inputs: Sequence[File] = [],  # noqa: ARG001
    outputs: Sequence[File] = [],
) -> bool:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    from pathlib import Path

    from pachyderm import yaml

    from mammoth.framework.analysis import objects as analysis_objects

    # Write them to YAML for later.
    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    output_dir = Path(outputs[0].filepath)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with output_dir.open("w") as f:
        y.dump(scale_factors, f)

    return True


def setup_write_scale_factors(
    prod: production.ProductionSettings,
    scale_factors: Mapping[int, Future[analysis_objects.ScaleFactor]],
    job_framework: job_utils.JobFramework,
) -> Future[bool]:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    """Write scale factors to YAML and to trees if necessary."""
    # First, we write to YAML.
    # We want to do this regardless of potentially writing the scale factor trees.
    logger.info("Writing scale factors to YAML. Jobs are executing, so this will take a minute...")
    output_filename = prod.scale_factors_filename
    parsl_output_file = File(str(output_filename))
    # NOTE: I'm guessing passing this is a problem because it's a class that's imported in an app, and then
    #       we're trying to pass the result into another app. I think we can go one direction or the other,
    #       but not both. So we just take the result.

    yaml_result = _write_scale_factors_to_yaml(
        # NOTE: We have to lie a bit about the typing here because passing a dask delayed object
        #       directly to another function will cause it to depend on it, which will fill in the
        #       value once it's executed. Parsl still have a future, so it needs to be handled separately.
        #       (Ideally, parsl would handle this correctly and provide the concrete value, but it doesn't seem
        #       seem to be able to do so...)
        scale_factors={  # type: ignore[arg-type]
            k: v.result() for k, v in scale_factors.items()
        } if job_framework == job_utils.JobFramework.parsl else scale_factors,
        job_framework=job_framework,
        outputs=[parsl_output_file],
    )

    return yaml_result  # noqa: RET504


@python_app
def _extract_pt_hat_spectra(
    scale_factors: Mapping[int, float],
    offsets: Mapping[int, int],
    list_name: str,
    yaml_exists: bool,  # noqa: ARG001
    job_framework: job_utils.JobFramework,  # noqa: ARG001
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> bool:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Args:
        scale_factors: pt_hat_bin to scale factor.
        offsets: pt_hat_bin to index where files in that pt hard bin start.
    """
    from pathlib import Path

    from mammoth.alice import scale_factors as sf

    # Convert back from parsl inputs
    offsets_values = list(offsets.values())
    filenames = {
        pt_hat_bin: [
            Path(f.filepath) for f in inputs[sum(offsets_values[:i]) : sum(offsets_values[: i + 1])]
        ]
        for i, pt_hat_bin in enumerate(offsets)
    }

    res = sf.pt_hat_spectra_from_hists(
        filenames=filenames,
        scale_factors=scale_factors,
        list_name=list_name,
        output_filename=Path(outputs[0].filepath),
    )
    return res  # noqa: RET504


def setup_check_pt_hat_spectra(
    prod: production.ProductionSettings,
    yaml_exists: bool,
    job_framework: job_utils.JobFramework,
) -> Future[bool]:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    logger.info("Checking pt hat spectra")
    # Input files
    input_files_per_pt_hat_bin = prod.input_files_per_pt_hat()

    # Must read the scale factors from file to get the properly scaled values.
    # NOTE: This means that we need to ensure externally that it's available.
    #       This is kind of hacking around the DAG, but it's convenient given the
    #       gymnastics that we need to do to support both parsl and dask
    scale_factors = prod.scale_factors()

    # Convert inputs to Parsl files.
    # Needs to be a list, so flatten them, and then unflatten in the App.
    parsl_files = []
    offsets = {}
    for pt_hat_bin, list_of_files in input_files_per_pt_hat_bin.items():
        converted_filenames = [File(str(f)) for f in list_of_files]
        offsets[pt_hat_bin] = len(converted_filenames)
        parsl_files.extend(converted_filenames)

    dataset_key = "signal_dataset" if "signal_dataset" in prod.config["metadata"] else "dataset"
    # We want to store it in the same directory as the scale factors, so it's easiest to just grab that filename.
    output_filename = prod.scale_factors_filename.parent / "pt_hat_spectra.yaml"

    results = _extract_pt_hat_spectra(
        scale_factors=scale_factors,
        offsets=offsets,
        list_name=prod.config["metadata"][dataset_key]["list_name"],
        yaml_exists=yaml_exists,
        job_framework=job_framework,
        inputs=parsl_files,
        outputs=[File(str(output_filename))],
    )

    return results  # noqa: RET504


def steer_extract_scale_factors(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
) -> list[Future[Any]]:
    # Validation
    if not prod.has_scale_factors:
        _msg = f"Invalid collision system for extracting scale factors: {prod.collision_system}"
        raise ValueError(_msg)

    # Attempt to bail out early if it's already been extracted
    scale_factors_filename = prod.scale_factors_filename
    if scale_factors_filename.exists():
        stored_scale_factors = prod.scale_factors()
        # We check if it's non-zero to avoid the case where it's accidentally empty
        if stored_scale_factors:
            logger.info("Scale factors already exist. Skipping extracting them again!")
            return []
    logger.info("Extracting scale factors...")

    # First, we need to extract the scale factors and keep track of the results
    all_results: list[Future[Any]] = []
    scale_factors = setup_extract_scale_factors(prod=prod, job_framework=job_framework)
    all_results.extend(list(scale_factors.values()))

    # Then, we need to write them
    all_results.append(
        setup_write_scale_factors(
            prod=prod,
            scale_factors=scale_factors,
            job_framework=job_framework
        )
    )
    # Store the result in a way that we can query later.
    if job_framework == job_utils.JobFramework.parsl:
        writing_yaml_success: bool = all_results[-1].result()
    elif job_framework == job_utils.JobFramework.dask_delayed:
        # We're lying about the type here because it's convenient
        writing_yaml_success = all_results[-1].compute()  # type: ignore[attr-defined]
    else:
        # We're lying about the type here because it's convenient
        writing_yaml_success = all_results[-1]  # type: ignore[assignment]

    if writing_yaml_success is not True:
        logger.warning("Some issue with the scale factor extraction! Check on them!")
        import IPython

        IPython.start_ipython(user_ns={**locals(), **globals()})  # type: ignore[no-untyped-call]
        # We want to stop here, so help ourselves out by raising the exception.
        _msg = "Some issue with the scale factor extraction!"
        raise ValueError(_msg)
    else:  # noqa: RET506
        # And then create the spectra (and plot them) to cross check the extraction
        all_results.append(
            setup_check_pt_hat_spectra(
                prod=prod,
                yaml_exists=writing_yaml_success,
                job_framework=job_framework,
            )
        )

    # NOTE: We don't want to scale_factor futures because they contain the actual scale factors.
    #       We'll just return the futures associated with writing to the file and the pt hat spectra cross check.
    # NOTE: If the case of dask, we only want to return the pt hat spectra cross check. Otherwise, it will run
    #       the writing to file task again
    return all_results[-1 if job_framework == job_utils.JobFramework.dask_delayed else -2:]
