"""Functionality related to job submission

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
import math
import os.path
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import attr

from parsl.config import Config
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SingleNodeLauncher, SrunLauncher
from parsl.launchers.launchers import Launcher
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
from parsl.monitoring.monitoring import MonitoringHub

from mammoth import helpers

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

logger = logging.getLogger(__name__)


FACILITIES = Literal[
    "local",
    "test_local_rehlers",
    "ORNL_b587_short",
    "ORNL_b587_long",
    "ORNL_b587_loginOnly",
]


def _expand_vars_in_work_dir_(
    instance: "TaskConfig",
    attribute: attr.Attribute[Path],
    value: Path,
) -> None:
    """Validate work dir."""
    # We need to expand any variations, but return a Path
    _p = os.path.expandvars(str(value))
    p = Path(_p)
    # Don't create the directory because this is perform automatically for all facilities!
    # (ie. we'll create random directories all over the place...)
    setattr(instance, attribute.name, p)


def _use_existing_work_dir_if_not_set(
    instance: "TaskConfig",
    attribute: attr.Attribute[Path],
    value: Path,
) -> None:
    """If the work_dir isn't set, then use the standard node_work_dir."""
    if value is None:
        setattr(instance, attribute.name, instance.node_work_dir)


@attr.s
class TaskConfig:
    """Configuration for a single task.

    Attributes:
        n_cores_per_task: Number of cores required per task.
        memory_per_task: Memory required per task in GB.

    Note:
        We prefer to specify jobs by number of cores (easy to reason in, and it's usually our
        constraint), but some facilities require specifying the memory as well. This is only
        worth doing if absolutely required.
    """

    n_cores_per_task: int = attr.ib()
    memory_per_task: Optional[int] = attr.ib(default=None)


@attr.s
class NodeSpec:
    n_cores: int = attr.ib()
    # Denoted in GB.
    memory: int = attr.ib()


@attr.s
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

    name: str = attr.ib()
    node_spec: NodeSpec = attr.ib()
    partition_name: str = attr.ib()
    # Number of cores to target allocating. Default: Full node.
    _target_allocate_n_cores: Optional[int] = attr.ib(default=None)
    allocation_account: str = attr.ib(default="")
    task_configs: Dict[str, TaskConfig] = attr.ib(factory=dict)
    node_work_dir: Path = attr.ib(default=Path("."))
    storage_work_dir: Path = attr.ib(
        validator=[_use_existing_work_dir_if_not_set, _expand_vars_in_work_dir_], default=None
    )
    directories_to_mount_in_singularity: List[Path] = attr.ib(factory=list)
    worker_init_script: str = attr.ib(default="")
    high_throughput_executor_additional_options: Dict[str, Any] = attr.ib(factory=dict)
    launcher: Callable[[], Launcher] = attr.ib(default=SrunLauncher)
    parsl_config_additional_options: Dict[str, Any] = attr.ib(factory=dict)
    cmd_timeout: int = attr.ib(default=10)

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
_facilities_configs = {
    f"ORNL_b587_{queue}": Facility(
        name="b587",
        node_spec=NodeSpec(n_cores=8, memory=64),
        partition_name=queue,
        # Allocate full node:
        target_allocate_n_cores=9 if queue != "loginOnly" else 6,
        # Allocate by core:
        # target_allocate_n_cores=1,
        launcher=SingleNodeLauncher,
        #node_work_dir=Path("/tmp/parsl/$USER"),
        #storage_work_dir=Path("/alf/data/rehlers/jetscape/work_dir"),
    ) for queue in ["short", "long", "loginOnly"]
}


def _default_parsl_config_kwargs(enable_monitoring: bool = False) -> Dict[str, Any]:
    """Default parsl config keyword arguments.

    These are shared regardless of the facility.

    Args:
        enable_monitoring: If True, enable parsl monitoring. Default: False. It's False because
            I am unsure of how this will interact with the particular facilities. Ideally, it
            should be enabled.

    Returns:
        Default config keyword arguments.
    """
    config_kwargs = dict(
        # This strategy is required to scale down blocks in the HTEX. As of Feb 2021, it is only
        # available in the parsl master.
        strategy="htex_auto_scale",
        # Identify a node as being idle after 20 seconds.
        # This is a balance - if we're too aggressive, then the blocks may be stopped while we still
        # have work remaining. However, if we're not aggressive enough, then we're wasting our allocation.
        max_idletime=20,
    )

    # Setup
    # Monitoring Information
    if enable_monitoring:
        config_kwargs["monitoring"] = MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55055,
            monitoring_debug=False,
            resource_monitoring_interval=10,
            # TODO: Make this settable
            #workflow_name=name,
        )

    return config_kwargs


def config(
    facility: FACILITIES,
    task_config: TaskConfig,
    n_tasks: int,
    walltime: str,
    enable_monitoring: bool = False,
    request_n_blocks: Optional[int] = None,
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
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

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Validation
    if facility not in _facilities_configs:
        raise ValueError(f"Facility '{facility}' is invalid. Possible values: {_facilities_configs}")
    _facility = _facilities_configs[facility]
    # Create the work directory once we know the facility.
    _facility.storage_work_dir.mkdir(parents=True, exist_ok=True)

    # Further validation
    if "test_local" in facility:
        # We need to treat the case of the local facility differently because
        # the provide is different (ie. it's not slurm).
        return _define_local_config(
            n_tasks=n_tasks,
            task_config=task_config,
            facility=_facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            request_n_blocks=request_n_blocks,
        )
    else:
        return _define_config(
            n_tasks=n_tasks,
            task_config=task_config,
            facility=_facility,
            walltime=walltime,
            enable_monitoring=enable_monitoring,
            request_n_blocks=request_n_blocks,
        )


def _define_config(
    n_tasks: int,
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    request_n_blocks: Optional[int] = None,
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
    """Define the parsl config based on the facility and task.

    Args:
        n_tasks: Number of tasks to be executed.
        task_config: Task configuration to be executed.
        facility: Facility configuration.
        walltime: Wall time for the job.
        enable_monitoring: If True, enable parsl monitoring. I am unsure of how this will
            interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.

    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Setup
    log_messages: List[helpers.LogMessage] = []

    # Determine request properties.
    # Namely, we need to know:
    # 1. How many cores to request per block
    # 2. How much memory to request per block
    # 3. How many blocks are required to run all tasks.
    n_cores_required = int(n_tasks * task_config.n_cores_per_task)
    if n_cores_required <= facility.target_allocate_n_cores:
        # Only need a single block
        n_blocks = 1
        n_cores_to_allocate_per_block = n_cores_required
        n_tasks_per_block = n_tasks
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
            n_tasks_per_block * n_blocks >= n_tasks
        ), f"Too many tasks per block. n_tasks_per_block: {n_tasks_per_block}, n_blocks: {n_blocks}, n_tasks: {n_tasks}"

    # Calculate the memory required per block
    # NOTE: type ignore because mypy apparently can't figure out that this is not None, even though the check is right there...
    memory_to_allocate_per_block = n_tasks_per_block * task_config.memory_per_task if task_config.memory_per_task else None

    log_messages.append(
        helpers.LogMessage(
            __name__,
            "info",
            f"Requesting {n_cores_to_allocate_per_block} cores in {n_blocks} block(s), with {n_tasks_per_block} tasks per block for {n_tasks} total tasks.",
        )
    )
    log_messages.append(
        helpers.LogMessage(
            __name__,
            "debug",
            f"Requesting {n_cores_required} total cores, {memory_to_allocate_per_block * n_tasks_per_block if memory_to_allocate_per_block else 'no constraint on'} GB total memory.",
        )
    )
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
        elif request_n_blocks < n_blocks:
            log_messages.append(
                helpers.LogMessage(
                    __name__,
                    "warning",
                    f"Explicitly requested fewer blocks ({request_n_blocks}) than necessary ({n_blocks}) to run everything simultaneously. Tasks will run sequentially in the requested number of blocks.",
                )
            )
            n_blocks = request_n_blocks

    # Setup
    config_kwargs = _default_parsl_config_kwargs(enable_monitoring=enable_monitoring)

    config = Config(
        executors=[
            HighThroughputExecutor(
                label=f"Jetscape_{facility.name}_HTEX",
                address=address_by_hostname(),
                cores_per_worker=task_config.n_cores_per_task,
                # cores_per_worker=round(n_cores_per_node / n_workers_per_node),
                # NOTE: We don't want to set the `max_workers` because we want the number of workers to
                #       be determined by the number of cores per worker and the cores per node.
                working_dir=str(facility.node_work_dir),
                provider=SlurmProvider(
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
                    scheduler_options="""""",
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init=facility.worker_init_script,
                    launcher=facility.launcher(),
                    walltime=walltime,
                    # If we're allocating full nodes, then we should request exclusivity.
                    exclusive=facility.allocate_full_node,
                    **facility.high_throughput_executor_additional_options,
                ),
            )
        ],
        **config_kwargs,
    )

    return config, facility, log_messages


def _define_local_config(
    n_tasks: int,
    task_config: TaskConfig,
    facility: Facility,
    walltime: str,
    enable_monitoring: bool,
    request_n_blocks: Optional[int] = None,
) -> Tuple[Config, Facility, List[helpers.LogMessage]]:
    """Local parsl configuration via process pool.

    This allows for testing parsl locally without needing to be on the facilities or having access
    to a test slurm system. Practically, this means that we'll still use the HighThroughputExecutor,
    but it will be provided via local processes. Careful not to overload your system.

    Our execution scheme is as follows:

    - One block is defined as one node.
    - One node is one core.
    - One job (ie worker) is executed per node.

    Args:
        n_tasks: Number of tasks to be executed.
        task_config: Task configuration to be executed.
        facility: Facility configuration.
        walltime: Wall time for the job.
        enable_monitoring: If True, enable parsl monitoring. I am unsure of how this will
            interact with the particular facilities.
        request_n_blocks: Explicitly request n_blocks instead of the calculated number. This
            value is still validated and won't be blindly accepted. Default: None, which will
            use the calculated number of blocks.
    Returns:
        Tuple of: A parsl configuration for the facility - allocating enough blocks to immediately
            execute all tasks, facility config, stored log messages.
    """
    # Setup
    log_messages: List[helpers.LogMessage] = []
    n_blocks_exact = (n_tasks * task_config.n_cores_per_task) / facility.target_allocate_n_cores
    n_blocks = math.ceil(n_blocks_exact)
    log_messages.append(
        helpers.LogMessage(
            __name__,
            "info",
            f"Number of blocks required: {n_blocks_exact}, requesting: {n_blocks}. These need to be close, or we will waste resources on some facilities.",
        )
    )
    # NOTE: This ignores the request_n_blocks. For now, it's not worth the effort, since we can easily test on slurm.

    # Setup
    config_kwargs = _default_parsl_config_kwargs(enable_monitoring=enable_monitoring)
    n_cores = facility.target_allocate_n_cores

    local_config = Config(
        executors=[
            HighThroughputExecutor(
                label=f"Jetscape_{facility.name}_HTEX",
                address=address_by_hostname(),
                cores_per_worker=task_config.n_cores_per_task,
                provider=LocalProvider(
                    # One block is one node.
                    nodes_per_block=1,
                    # We want n_blocks initially because we will have work for everything immediately.
                    # (useless explicitly requested otherwise).
                    min_blocks=1,
                    max_blocks=n_cores,
                    # We try to select one core less for the init so we can see some scaling
                    # if we max out everything.
                    # NOTE: We need at least one block, so if set to just one core, n-1 would break.
                    #       Consequently, we require at least one initial block.
                    init_blocks=max(n_cores - 1, 1),
                    worker_init=facility.worker_init_script,
                    launcher=facility.launcher(),
                    walltime=walltime,
                ),
            )
        ],
        **config_kwargs,
    )

    return local_config, facility, log_messages
