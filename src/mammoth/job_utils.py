"""Functionality related to job submission

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import enum
import logging
import math
import os.path
import typing
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

import attrs
import dask
import dask.distributed
import parsl
from parsl.addresses import address_by_hostname, address_by_route
from parsl.app.app import python_app as parsl_python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.errors import WorkerLost
from parsl.executors.high_throughput.interchange import ManagerLost
from parsl.launchers import SingleNodeLauncher, SrunLauncher
from parsl.launchers.launchers import Launcher
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import LocalProvider, SlurmProvider

from mammoth import helpers

logger = logging.getLogger(__name__)


FACILITIES = Literal[
    "rehlers_mbp_m1pro",
    "rehlers_mbp_m1pro_multi_core",
    "ORNL_b587_short",
    "ORNL_b587_long",
    "ORNL_b587_loginOnly",
    "ORNL_b587_vip",
    "hiccup_quick",
    "hiccup_std",
    "hiccup_long",
    "hiccup_test",
]


def hours_in_walltime(walltime: str) -> int:
    """Extract the hours from a walltime string.

    Note:
        This can be convenient if you need to change queues based on the walltime,
        eg. a short vs long queue. This allows you to abstract some decisions by just
        specifying the walltime.

    Args:
        walltime: Walltime string in the format "HH:MM:SS".
    Returns:
        Number of hours in the walltime.
    """
    return int(walltime.split(":")[0])


def _expand_vars_in_work_dir(
    value: str | Path,
) -> Path:
    """Validate work dir."""
    _p = os.path.expandvars(str(value))
    p = Path(_p)
    return p  # noqa: RET504


@attrs.define
class TaskConfig:
    """Configuration for a single task.

    Attributes:
        name: Name of the task. It will be passed to parsl.
        n_cores_per_task: Number of cores required per task.
        memory_per_task: Memory required per task in GB.

    Note:
        We prefer to specify jobs by number of cores (easy to reason in, and it's usually our
        constraint), but some facilities require specifying the memory as well. This is only
        worth doing if absolutely required.
    """

    name: str
    n_cores_per_task: int
    memory_per_task: int | None = attrs.field(default=None)


@attrs.define
class NodeSpec:
    n_cores: int
    # Denoted in GB.
    memory: int


@attrs.define
class Facility:
    """Facility configuration.

    Attributes:
        name: Name of the facility.
        node_spec: Specification for a single node. This is needed to inform parsl
            about node constraints, what resources to request, and how to schedule jobs.
        partition_name: Name of the partition.
        target_allocate_n_cores: Target number of cores to allocate via slurm. This
            may be an entire node, or only part of one. Note that this is separate
            from the number of cores that are required for a particular task. Default: None,
            which corresponds to targeting allocating of the entire node.
        allocation_account: Name of allocation account, to be passed via slurm. Default: "".
        task_configs: Node configurations required for particular tasks. For example,
            for a jet energy loss calculation, or for hydro.
        node_work_dir: Work directory for where jobs are executed. This can be used to execute on
            local storage of a node. Default: Current directory.
        storage_work_dir: Work directory for where runs are stored. Default: Same as the node_work_dir.
        directories_to_mount_in_singularity: Directories to mount in singularity. Default: [].
        worker_init_script: Worker initialization script. Default: "".
        high_throughput_executor_additional_options: Additional keyword options to pass
            directly to the high throughput executor. Default: {}
        launcher: Launcher class to use with the high throughput executor. Default: SrunLauncher.
        parsl_config_additional_options: Additional keyword options to pass directly to
            the parsl config. Default: {}
    """

    name: str
    node_spec: NodeSpec
    partition_name: str
    # Number of cores to target allocating. Default: Full node.
    _target_allocate_n_cores: int | None = attrs.field(default=None)
    allocation_account: str = attrs.field(default="")
    task_configs: dict[str, TaskConfig] = attrs.Factory(dict)
    node_work_dir: Path = attrs.field(default=Path("."))  # noqa: PTH201
    storage_work_dir: Path = attrs.field(
        converter=_expand_vars_in_work_dir,
        default=Path("."),  # noqa: PTH201
    )
    directories_to_mount_in_singularity: list[Path] = attrs.Factory(list)
    worker_init_script: str = attrs.field(default="")
    high_throughput_executor_additional_options: dict[str, Any] = attrs.Factory(dict)
    launcher: Callable[[], Launcher] = attrs.field(default=SrunLauncher)
    parsl_config_additional_options: dict[str, Any] = attrs.Factory(dict)
    cmd_timeout: int = attrs.field(default=10)
    nodes_to_exclude: list[str] = attrs.Factory(list)

    @property
    def target_allocate_n_cores(self) -> int:
        if self._target_allocate_n_cores is None:
            return self.node_spec.n_cores
        return self._target_allocate_n_cores

    @property
    def allocate_full_node(self) -> bool:
        """True if we are allocating full nodes."""
        # If our target for allocating cores is equal to a single node, then we are allocating a full node.
        return self.node_spec.n_cores == self.target_allocate_n_cores


# Define the facility configurations.
# 587 cluster
_facilities_configs = {
    f"ORNL_b587_{queue}": Facility(
        name="b587",
        node_spec=NodeSpec(n_cores=11, memory=64),
        partition_name=queue,
        # Allocate full node:
        # target_allocate_n_cores=11 if queue != "loginOnly" else 6,
        # Allocate by core:
        target_allocate_n_cores=1,
        launcher=SingleNodeLauncher,
        # node_work_dir=Path("/tmp/parsl/$USER"),
        # storage_work_dir=Path("/alf/data/rehlers/jetscape/work_dir"),
        # Exclude login node
        nodes_to_exclude=[] if queue == "long" else ["pc059"],
    )
    for queue in ["short", "long", "loginOnly", "vip"]
}
# rehlers-MBP-m1pro
_facilities_configs.update(
    {
        f"rehlers_mbp_m1pro{'_multi_core' if multi_core else ''}": Facility(
            name=f"rehlers_mbp_m1pro{'_multi_core' if multi_core else ''}",
            node_spec=NodeSpec(n_cores=8, memory=12),
            partition_name="INVALID",
            target_allocate_n_cores=1 if multi_core is False else 8,
            launcher=SingleNodeLauncher,
            # node_work_dir=Path("/tmp/parsl/$USER"),
            # storage_work_dir=(Path.cwd() / Path("work_dir")).resolve(),
            directories_to_mount_in_singularity=[Path("/opt/scott")],
        )
        for multi_core in [False, True]
    }
)
# Hiccup at LBL
_facilities_configs.update(
    {
        f"hiccup_{queue}": Facility(
            name="hiccup",
            node_spec=NodeSpec(n_cores=20, memory=64),
            partition_name=queue,
            # Allocate full node:
            # target_allocate_n_cores=11 if queue != "loginOnly" else 6,
            # Allocate by core:
            target_allocate_n_cores=1,
            launcher=SingleNodeLauncher,
            # node_work_dir=Path("/tmp/parsl/$USER"),
            # storage_work_dir=Path("/alf/data/rehlers/jetscape/work_dir"),
            # Exclude login node
            nodes_to_exclude=[],
        )
        for queue in ["quick", "std", "long", "test"]
    }
)


class JobFramework(enum.Enum):
    dask_delayed = enum.auto()
    parsl = enum.auto()
    immediate_execution_debug = enum.auto()


P = ParamSpec("P")
R = TypeVar("R")


def python_app(func: Callable[P, R]) -> Callable[P, concurrent.futures.Future[R]]:
    """Helper for defining a python app for different job execution frameworks"""

    @wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> concurrent.futures.Future[R]:
        # Default to using parsl. Only use other frameworks if explicitly requested.
        # Grabbing it via the kwargs hurts discovering the option, but it's possible to implement the typing.
        # Note that we can't do better on the typing yet because we can't concatenate keyword parameters as of Dec 2022.
        # See: https://peps.python.org/pep-0612/#concatenating-keyword-parameters
        job_framework = kwargs.get("job_framework", JobFramework.parsl)
        if job_framework == JobFramework.dask_delayed:
            return dask.delayed(func)(*args, **kwargs)  # type: ignore[no-any-return]
        elif job_framework == JobFramework.parsl:  # noqa: RET505
            return parsl_python_app(func)(*args, **kwargs)  # type: ignore[no-any-return]
        elif job_framework == JobFramework.immediate_execution_debug:
            # NOTE: This is lying about the return value. But that's okay because this is just for
            #       immediate execution for debugging.
            return func(*args, **kwargs)  # type: ignore[return-value]
        else:
            _msg = f"Unrecognized job framework {job_framework}"
            raise ValueError(_msg)

    return inner


@typing.overload
def config(
    job_framework: Literal[JobFramework.parsl],
    facility: FACILITIES,
    task_config: TaskConfig,
    target_n_tasks_to_run_simultaneously: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: int | None = None,
    additional_worker_init_script: str = "",
) -> tuple[Config, Facility, list[helpers.LogMessage]]:
    ...


@typing.overload
def config(
    job_framework: Literal[JobFramework.dask_delayed],
    facility: FACILITIES,
    task_config: TaskConfig,
    target_n_tasks_to_run_simultaneously: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: int | None = None,
    additional_worker_init_script: str = "",
) -> tuple[dask.distributed.Client, Facility, list[helpers.LogMessage]]:
    ...


@typing.overload
def config(
    job_framework: JobFramework,
    facility: FACILITIES,
    task_config: TaskConfig,
    target_n_tasks_to_run_simultaneously: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: int | None = None,
    additional_worker_init_script: str = "",
) -> tuple[dask.distributed.client | Config, Facility, list[helpers.LogMessage]]:
    ...


def config(
    job_framework: JobFramework,
    facility: FACILITIES,
    task_config: TaskConfig,
    target_n_tasks_to_run_simultaneously: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: int | None = None,
    additional_worker_init_script: str = "",
) -> tuple[dask.distributed.client | Config, Facility, list[helpers.LogMessage]]:
    """Retrieve the appropriate parsl configuration for a facility and task.

    This is the main interface for retrieving these configurations.

    Args:
        facility: Name of facility. Possible values are in `FACILITIES`.
        task_config: Task configuration to be executed.
        n_tasks: Total number of tasks execute.
        walltime: Requested wall time for the job. Short times will (probably) be easier to schedule.
            Format: "hh:mm:ss".
        enable_monitoring: If True, enable parsl monitoring. Default: False, since I am unsure of how
            this will interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
        additional_worker_init_script: Additional script for initializing the worker. Default: ""

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Validation
    if facility not in _facilities_configs:
        _msg = f"Facility '{facility}' is invalid. Possible values: {_facilities_configs}"
        raise ValueError(_msg)
    _facility = _facilities_configs[facility]
    # Create the work directory once we know the facility.
    _facility.storage_work_dir.mkdir(parents=True, exist_ok=True)

    # Further validation
    return _define_config(
        job_framework=job_framework,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        task_config=task_config,
        facility=_facility,
        walltime=walltime,
        enable_monitoring=enable_monitoring,
        request_n_blocks=request_n_blocks,
        additional_worker_init_script=additional_worker_init_script,
    )


def _potentially_immediately_log_message(
    log_messages: list[helpers.LogMessage], immediately_log_messages: bool
) -> None:
    """If we can log immediately, let's do it. Otherwise, we leave it in place for later."""
    if immediately_log_messages:
        log_messages.pop().log()


def _define_config(
    job_framework: JobFramework,
    target_n_tasks_to_run_simultaneously: int,
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    request_n_blocks: int | None = None,
    additional_worker_init_script: str = "",
) -> tuple[Config, Facility, list[helpers.LogMessage]]:
    """Define the parsl config based on the facility and task.

    Args:
        job_framework: Job framework.
        target_n_tasks_to_run_simultaneously: Number of tasks to be executed simultaneously.
        task_config: Task configuration to be executed.
        facility: Facility configuration.
        walltime: Wall time for the job.
        enable_monitoring: If True, enable parsl monitoring. I am unsure of how this will
            interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
        additional_worker_init_script: Additional script for initializing the worker. Default: ""

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Setup
    log_messages: list[helpers.LogMessage] = []
    # If we're not dealing with parsl, there's no reason not to log immediately.
    immediately_log_messages = job_framework != JobFramework.parsl

    # Determine request properties.
    # Namely, we need to know:
    # 1. How many cores to request per block
    # 2. How much memory to request per block
    # 3. How many blocks are required to run all tasks.
    n_cores_required = int(target_n_tasks_to_run_simultaneously * task_config.n_cores_per_task)
    if n_cores_required <= facility.target_allocate_n_cores:
        # Only need a single block
        n_blocks = 1
        n_cores_to_allocate_per_block = n_cores_required
        n_tasks_per_block = target_n_tasks_to_run_simultaneously
    else:
        # Need multiple blocks.
        # Let's spread out as evenly as possible.
        # If we tried to pack into as few of blocks as possible, we would waste cores in our allocation.
        n_blocks = math.ceil(n_cores_required / facility.target_allocate_n_cores)
        # Need ceil here in case the number of required cores doesn't divide evenly.
        n_cores_to_allocate_per_block = math.ceil(n_cores_required / n_blocks)
        # Need to make sure that it fits within our core requirements, so round up to the
        # nearest multiple of n_cores_per_task
        n_cores_to_allocate_per_block = n_cores_to_allocate_per_block + (
            n_cores_to_allocate_per_block % task_config.n_cores_per_task
        )
        # Have to additional round because otherwise python will treat this as a float.
        n_tasks_per_block = round(n_cores_to_allocate_per_block / task_config.n_cores_per_task)

        # Cross check
        assert (
            n_tasks_per_block * n_blocks >= target_n_tasks_to_run_simultaneously
        ), f"Too many tasks per block. n_tasks_per_block: {n_tasks_per_block}, n_blocks: {n_blocks}, n_tasks: {target_n_tasks_to_run_simultaneously}"

    # Calculate the memory required per block
    # NOTE: type ignore because mypy apparently can't figure out that this is not None, even though the check is right there...
    memory_to_allocate_per_block = (
        n_tasks_per_block * task_config.memory_per_task if task_config.memory_per_task else None
    )

    log_messages.append(
        helpers.LogMessage(
            __name__,
            "info",
            f"Requesting {n_cores_to_allocate_per_block} core(s) in {n_blocks} block(s), with {n_tasks_per_block} tasks per block for {target_n_tasks_to_run_simultaneously} total tasks running simultaneously.",
        )
    )
    _potentially_immediately_log_message(log_messages=log_messages, immediately_log_messages=immediately_log_messages)
    log_messages.append(
        helpers.LogMessage(
            __name__,
            "debug",
            f"Requesting {n_cores_required} total cores, {memory_to_allocate_per_block * n_tasks_per_block if memory_to_allocate_per_block else 'no constraint on'} GB total memory.",
        )
    )
    _potentially_immediately_log_message(log_messages=log_messages, immediately_log_messages=immediately_log_messages)
    # Validate
    if request_n_blocks:
        if request_n_blocks > n_blocks:
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    f"Explicitly requested more blocks than needed. We'll ignore this request and take only the minimum. Requested n_blocks: {n_blocks}, required n blocks: {n_blocks}",
                )
            )
            _potentially_immediately_log_message(
                log_messages=log_messages, immediately_log_messages=immediately_log_messages
            )
        elif request_n_blocks < n_blocks:
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    f"Explicitly requested fewer blocks ({request_n_blocks}) than necessary ({n_blocks}) to run everything simultaneously. Tasks will run sequentially in the requested number of blocks.",
                )
            )
            _potentially_immediately_log_message(
                log_messages=log_messages, immediately_log_messages=immediately_log_messages
            )
            n_blocks = request_n_blocks

    if job_framework == JobFramework.immediate_execution_debug:
        # Debug option intentionally breaking typing
        job_framework_config, _additional_log_messages = None, []  # type: ignore[var-annotated]
    elif job_framework == JobFramework.parsl:
        # Parsl specific
        job_framework_config, _additional_log_messages = _define_parsl_config(
            task_config=task_config,
            facility=facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            n_blocks=n_blocks,
            n_tasks_per_block=n_tasks_per_block,
            n_cores_to_allocate_per_block=n_cores_to_allocate_per_block,
            memory_to_allocate_per_block=memory_to_allocate_per_block,
            additional_worker_init_script=additional_worker_init_script,
        )
    else:
        # Dask specific
        job_framework_config, _additional_log_messages = _define_dask_distributed_cluster(  # type: ignore[assignment]
            task_config=task_config,
            facility=facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            n_blocks=n_blocks,
            n_tasks_per_block=n_tasks_per_block,
            n_cores_to_allocate_per_block=n_cores_to_allocate_per_block,
            memory_to_allocate_per_block=memory_to_allocate_per_block,
            additional_worker_init_script=additional_worker_init_script,
        )

    # Store any further log messages
    log_messages.extend(_additional_log_messages)

    return job_framework_config, facility, log_messages  # type: ignore[return-value]


def _define_dask_distributed_cluster(
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    n_blocks: int,
    n_tasks_per_block: int,
    n_cores_to_allocate_per_block: int,
    memory_to_allocate_per_block: int | None,
    additional_worker_init_script: str = "",
) -> tuple[dask.distributed.SpecCluster, list[helpers.LogMessage]]:
    """Dask distributed cluster config"""
    # NOTE: We're fine to log directly here because we know that logging with dask is fine.
    if enable_monitoring:
        logger.debug("NOTE: Requested monitoring to be enabled, but it's always enabled for dask")

    # We want each worker to know how many cores it has available so we can then later tell dask how many cores each
    # task needs. This allows for multiple cores per task (assuming a worker has enough cores available).
    # This is based on the concept of "resources", described here:
    # https://distributed.dask.org/en/stable/resources.html#resources-are-applied-separately-to-each-worker-process ,
    # with an example here: https://github.com/dask/dask-jobqueue/issues/181#issuecomment-454390647
    # I can't immediately confirm that this works, but so far (Jan 2023), it seems to be fine.
    # NOTE: This is defined as a per worker quantity.
    # NOTE: This name is defined by convention. We need to add our specification in the resources when
    #       we submit jobs for processing.
    resources = {"n_cores": n_cores_to_allocate_per_block}

    # We need to treat the case of the local facility differently because
    # the cluster is different (ie. it's not slurm).
    if facility.partition_name == "INVALID" or "rehlers_mbp_m1pro" in facility.name:
        # Due to limitations of the LocalCluster, we need only to have only 1 task per worker.
        # Here, we'll modify the allocation logic to make this work.
        n_cores_to_allocate_per_block = task_config.n_cores_per_task
        # Update the resources available.
        resources = {"n_cores": n_cores_to_allocate_per_block}
        if n_tasks_per_block > 1:
            # It's possible that the config will generate more than one task per worker. In this case,
            # we need to create more workers to compensate.
            n_blocks = n_blocks * n_tasks_per_block
            logger.info(
                f"Since running local config, we can't have more than one task per worker (==block). Thus, change the number of workers to {n_tasks_per_block}"
            )

        # Ensure that we don't overload an individual system by requesting more cores than are available.
        if n_blocks * n_cores_to_allocate_per_block > facility.node_spec.n_cores:
            n_blocks = round(facility.node_spec.n_cores / n_cores_to_allocate_per_block)
            logger.info(
                f"Since running local config, we set the number of blocks ({n_blocks}) to the available number of cores to avoid overloading the system."
            )

        cluster = dask.distributed.LocalCluster(  # type: ignore[no-untyped-call]
            n_workers=n_blocks,
            threads_per_worker=1,
            processes=True,
            resources=resources,
        )
        # Actually request the jobs. Doing this or not can be made configurable later if needed, but the default
        # from parsl is to immediately allocate, so if nothing else, it's provides the same functionality.
        cluster.adapt(minimum=0, maximum=n_blocks, interval="10s")
    else:
        import dask_jobqueue

        cluster = dask_jobqueue.SLURMCluster(
            # Need to pass None - otherwise it will pass an invalid job script (worse, it doesn't breaks on this line, but the next one)
            account=facility.allocation_account if facility.allocation_account else None,
            queue=facility.partition_name,
            cores=n_cores_to_allocate_per_block,
            processes=round(n_cores_to_allocate_per_block / task_config.n_cores_per_task),
            # NOTE: Dask appears to require a unit, unlike parsl, so we add it here in GB, since we expect to pass GB.
            memory=f"{memory_to_allocate_per_block!s}GB",
            # string to prepend to #SBATCH blocks in the submit
            # Can add additional options directly to scheduler.
            job_extra_directives=[
                f"#SBATCH --exclude={','.join(facility.nodes_to_exclude)}" if facility.nodes_to_exclude else ""
            ],
            # Command to be run before starting a worker, such as:
            # 'module load Anaconda; source activate parsl_env'.
            job_script_prologue=[f"{facility.worker_init_script}; {additional_worker_init_script}"]
            if facility.worker_init_script
            else [additional_worker_init_script],
            walltime=walltime,
            # Apparently they dropped direct resources support (I can't fully trace it now), so we have to work around it by passing worker_extra_args
            worker_extra_args=["--resources " + ",".join([f"{k}={v}" for k, v in resources.items()])],
        )
        # Actually request the jobs. Doing this or not can be made configurable later if needed, but the default
        # from parsl is to immediately allocate, so if nothing else, it's provides the same functionality.
        # NOTE: This call uses "_jobs" arguments. These may be the same as straight maximum, but I think there's potentially
        #       a factor of the number of processes. See: https://github.com/dask/dask-jobqueue/blob/bee0e0c5444a4fecfa8e273ba0ff871679d9e9e1/dask_jobqueue/core.py#L828-L831 .
        #       Since it's slightly unclear, it's easier just to have separate calls rather than worrying about it!
        cluster.adapt(minimum_jobs=0, maximum_jobs=n_blocks, interval="10s")

    return cluster, []


def _default_parsl_config_kwargs(
    facility: Facility, workflow_name: str, enable_monitoring: bool = True
) -> dict[str, Any]:
    """Default parsl config keyword arguments.

    These are shared regardless of the facility.

    Args:
        enable_monitoring: If True, enable parsl monitoring. Default: True.

    Returns:
        Default config keyword arguments.
    """
    config_kwargs = {
        # This strategy is required to scale down blocks in the HTEX.
        "strategy": "htex_auto_scale",
        # Identify a node as being idle after 20 seconds.
        # This is a balance - if we're too aggressive, then the blocks may be stopped while we still
        # have work remaining. However, if we're not aggressive enough, then we're wasting our allocation.
        "max_idletime": 20,
    }

    # Setup
    # Monitoring Information
    if enable_monitoring:
        config_kwargs["monitoring"] = MonitoringHub(
            # For some reason, parsl doesn't seem to like address_by_hostname on hiccup. Maybe it's hiccup or maybe it's parsl.
            # In any case, this is a workaround
            hub_address=address_by_hostname() if "hiccup" not in facility.name else address_by_route(),
            monitoring_debug=False,
            resource_monitoring_interval=10,
            workflow_name=workflow_name,
        )

    return config_kwargs


def _define_parsl_config(
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    n_blocks: int,
    n_tasks_per_block: int,  # noqa: ARG001
    n_cores_to_allocate_per_block: int,
    memory_to_allocate_per_block: int | None,
    additional_worker_init_script: str = "",
) -> tuple[Config, list[helpers.LogMessage]]:
    # Setup
    log_messages: list[helpers.LogMessage] = []
    config_kwargs = _default_parsl_config_kwargs(
        facility=facility, workflow_name=task_config.name, enable_monitoring=enable_monitoring
    )

    # We need to treat the case of the local facility differently because
    # the provider is different (ie. it's not slurm).
    if facility.partition_name == "INVALID" or "test_local" in facility.name:
        # Ensure that we don't overload an individual system by requesting more cores than are available.
        if n_blocks * n_cores_to_allocate_per_block > facility.node_spec.n_cores:
            n_blocks = round(facility.node_spec.n_cores / n_cores_to_allocate_per_block)
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    "Since running local config, we set the number of blocks to the available number of cores to avoid overloading the system.",
                )
            )
            # NOTE: We don't try to log immediately here because know that we can't log immediately with parsl.
        provider: LocalProvider | SlurmProvider = LocalProvider(  # type: ignore[no-untyped-call]
            # One block is one node.
            nodes_per_block=1,
            # We want n_blocks initially because we will have work for everything immediately.
            # (useless explicitly requested otherwise).
            min_blocks=0,
            max_blocks=n_blocks,
            init_blocks=n_blocks,
            # NOTE: If we want to try scaling, we can select less core for the init. If we
            #       We need at least one block, so if set to just one core, n-1 would break.
            #       Consequently, we require at least one initial block.
            # init_blocks=max(n_cores - 1, 1),
            worker_init=f"{facility.worker_init_script}; {additional_worker_init_script}"
            if facility.worker_init_script
            else additional_worker_init_script,
            launcher=facility.launcher(),
        )
    else:
        provider = SlurmProvider(
            # This is how many cores and how much memory we'll request per node.
            cores_per_node=n_cores_to_allocate_per_block,
            mem_per_node=memory_to_allocate_per_block,
            # One block is one node.
            nodes_per_block=1,
            # We want n_blocks initially because we will have work for everything immediately.
            # (useless explicitly requested otherwise).
            min_blocks=0,
            max_blocks=n_blocks,
            init_blocks=n_blocks,
            partition=facility.partition_name,
            account=facility.allocation_account,
            # string to prepend to #SBATCH blocks in the submit
            # Can add additional options directly to scheduler.
            scheduler_options=f"#SBATCH --exclude={','.join(facility.nodes_to_exclude)}"
            if facility.nodes_to_exclude
            else "",
            # Command to be run before starting a worker, such as:
            # 'module load Anaconda; source activate parsl_env'.
            worker_init=f"{facility.worker_init_script}; {additional_worker_init_script}"
            if facility.worker_init_script
            else additional_worker_init_script,
            launcher=facility.launcher(),
            walltime=walltime,
            # If we're allocating full nodes, then we should request exclusivity.
            exclusive=facility.allocate_full_node,
            **facility.high_throughput_executor_additional_options,
        )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label=f"{facility.name}_HTEX",
                # For some reason, parsl doesn't seem to like address_by_hostname on hiccup. Maybe it's hiccup or maybe it's parsl.
                # In any case, this is a workaround
                address=address_by_hostname() if "hiccup" not in facility.name else address_by_route(),
                cores_per_worker=task_config.n_cores_per_task,
                # cores_per_worker=round(n_cores_per_node / n_workers_per_node),
                # NOTE: We don't want to set the `max_workers` because we want the number of workers to
                #       be determined by the number of cores per worker and the cores per node.
                working_dir=str(facility.node_work_dir),
                provider=provider,
            )
        ],
        **config_kwargs,
    )

    return config, log_messages


@typing.overload
def setup_job_framework(
    job_framework: Literal[JobFramework.dask_delayed],
    task_config: TaskConfig,
    facility: FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    additional_worker_init_script: str = "",
) -> tuple[dask.distributed.Client, dask.distributed.SpecCluster]:
    ...


@typing.overload
def setup_job_framework(
    job_framework: Literal[JobFramework.parsl],
    task_config: TaskConfig,
    facility: FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    additional_worker_init_script: str = "",
) -> tuple[parsl.DataFlowKernel, Config]:
    ...


@typing.overload
def setup_job_framework(
    job_framework: JobFramework,
    task_config: TaskConfig,
    facility: FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    additional_worker_init_script: str = "",
) -> tuple[parsl.DataFlowKernel, parsl.Config] | tuple[dask.distributed.Client, dask.distributed.SpecCluster]:
    ...


def setup_job_framework(
    job_framework: JobFramework,
    task_config: TaskConfig,
    facility: FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    additional_worker_init_script: str = "",
) -> tuple[parsl.DataFlowKernel, parsl.Config] | tuple[dask.distributed.Client, dask.distributed.SpecCluster]:
    # Basic setup: logging and parsl.
    # Setup job frameworks
    if job_framework != JobFramework.parsl:
        # As long as it's not parsl, it's fine to setup now!
        helpers.setup_logging(
            level=log_level,
        )
        # Dask is still pretty loud, so calm it down in most instances (unless we're trying to debug)
        if log_level != logging.DEBUG:
            logging.getLogger("dask").setLevel(logging.WARNING)
            logging.getLogger("distributed").setLevel(logging.WARNING)
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    job_framework_config, _facility_config, stored_messages = config(
        job_framework=job_framework,
        facility=facility,
        task_config=task_config,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        walltime=walltime,
        enable_monitoring=True,
        additional_worker_init_script=additional_worker_init_script,
    )
    if job_framework == JobFramework.immediate_execution_debug:
        # This is a debug option, so it will break typing
        return None, None  # type: ignore[return-value]
    elif job_framework == JobFramework.dask_delayed:  # noqa: RET505
        return dask.distributed.Client(job_framework_config), job_framework_config  # type: ignore[no-untyped-call,return-value]
    else:
        # Keep track of the dfk to keep parsl alive
        dfk = helpers.setup_logging_and_parsl(
            parsl_config=job_framework_config,
            level=log_level,
            stored_messages=stored_messages,
        )

        # Quiet down parsl
        logging.getLogger("parsl").setLevel(logging.WARNING)

        return dfk, job_framework_config


def _cancel_future(job: concurrent.futures.Future[Any]) -> None:
    """Cancel the given app future

    Taken from `coffea.processor.executor`

    Args:
        job: AppFuture to try to cancel
    """
    with contextlib.suppress(NotImplementedError):
        # NOTE: This is not implemented with parsl AppFutures
        job.cancel()


_T = TypeVar("_T")


def provide_results_as_completed(
    input_futures: Sequence[concurrent.futures.Future[_T]],
    timeout: float | None = None,
    running_with_parsl: bool = False,
) -> Iterable[_T]:
    """Provide results as futures are completed.

    Taken from `coffea.processor.executor`, with small modifications for parsl specific issues
    around cancelling jobs. Without this change, parsl always seems to hang.
    Their docs note that it is essentially the same as `concurrent.futures.as_completed`,
    but it makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.

    Args:
        input_futures: AppFutures which will eventually contain results
        timeout: Timeout to wait for results be bailing out. Passed directly to
            `concurrent.futures.wait`. Default: None.
        running_with_parsl: If True, don't wait for futures to cancel (since that's not
            implemented in parsl), and just raise the exception. Without this, parsl seems
            to hang. Default: False.

    Returns:
        Iterable containing the results from futures. They are yielded as the futures complete.
    """
    futures = set(input_futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    logger.warning(f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early")
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
                    except WorkerLost as e:
                        logger.warning(f"Lost worker: {e}")
                    except ManagerLost as e:
                        logger.warning(f"Manager lost: {e}")
            except KeyboardInterrupt as e:
                for job in futures:
                    _cancel_future(job)
                running = sum(job.running() for job in futures)
                logger.warning(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
                # parsl can't cancel, so we need to break out ourselves
                # It's most convenient to do this by just re-raising the ctrl-c
                if running_with_parsl:
                    raise e
    finally:
        running = sum(job.running() for job in futures)
        if running:
            logger.warning(f"Cancelling {running} running jobs (likely due to an exception)")
        while futures:
            _cancel_future(futures.pop())
