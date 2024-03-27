"""Some tests for file management.

Based on the rsync provider. Modifications include:

- Removed remote system support - it only copies locally.
- Staged files are cleaned up after they are staged back.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import concurrent.futures
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import attrs
import parsl
from parsl.app.futures import DataFuture
from parsl.data_provider.data_manager import DataManager
from parsl.data_provider.files import File
from parsl.data_provider.staging import Staging
from parsl.utils import RepresentationMixin

logger = logging.getLogger(__name__)


@attrs.define(frozen=True)
class FileStaging:
    """Handles file staging within a job.

    The concept here is that there are two directories:
        - The permanent directory, which is where the files are stored permanently.
        - The worker node directory, which is where the files are stored on the worker node.
    Within this concept, we keep the directory structure as consistent as possible, even
    on the worker node, eg.:

    ```python
    fs = FileStaging(
        permanent_work_dir=Path("/path/to/permanent"),
        node_work_dir=Path("/path/to/worker_node"),
    )
    input_file = Path("/path/to/permanent/some_directory_path/input_file.txt")
    result = fs.stage_files_in(files_to_stage_in=[input_file])
    assert result[0] == Path(
        "/path/to/worker_node/input/some_directory_path/input_file.txt"
    )
    ```

    and then staging out will undo this, again as appropriate. The scheme is slightly involved,
    but I think it will be more intuitive in the long run.

    Attributes:
        permanent_work_dir: Permanent work directory for storing files.
        node_work_dir: Work directory on the worker node for storing files.
    """

    permanent_work_dir: Path
    node_work_dir: Path

    @property
    def node_work_dir_input(self) -> Path:
        """Input directory on the worker node.

        Nothing deeper here - just a useful convention.
        """
        return self.node_work_dir / "input"

    @property
    def node_work_dir_output(self) -> Path:
        """Output directory on the worker node.

        Nothing deeper here - just a useful convention.
        """
        return self.node_work_dir / "output"

    def stage_files_in(self, files_to_stage_in: list[Path]) -> list[Path]:
        """Stage files in.

        Args:
            files_to_stage_in: Files to stage in.
        Returns:
            Paths of the files that were staged out, in their worker node locations.
        """
        # Setup as necessary
        stage_in_dir = self.node_work_dir_input
        stage_in_dir.mkdir(parents=True, exist_ok=True)

        # Stage in the files, relative to the permanent directory.
        modified_paths = []
        for f in files_to_stage_in:
            p = stage_in_dir / f.relative_to(self.permanent_work_dir)
            modified_paths.append(p)
            shutil.copy(f, p)

        return modified_paths

    def _stage_files_out(self, files_to_stage_out: list[Path]) -> list[Path]:
        """Stage files out implementation.

        Args:
            files_to_stage_out: Files to stage out.
        Returns:
            Paths of the files that were staged out, in their permanent locations.
        """
        # Stage out the files, relative to the permanent directory.
        modified_paths = []
        for f in files_to_stage_out:
            p = self.permanent_work_dir / f.relative_to(self.node_work_dir)
            try:
                modified_paths.append(p)
                shutil.copy(f, p)
            except OSError as e:
                logger.exception(e)
                continue
            # If we've succeeded in copying, we can remove the existing file.
            f.unlink()

        return modified_paths

    def stage_all_files_out(self) -> list[Path]:
        """Stage out all files in the output directory.

        Returns:
            Paths of the files that were staged out, in their permanent locations.
        """
        # NOTE: Could also use shutil.copytree(src, dest, dir_exist_ok=True), but
        #       this is more convenient since we already implemented the copying, so
        #       just leave it as is for now...
        all_files_to_stage_out = list(self.node_work_dir_output.rglob("*"))
        return self._stage_files_out(files_to_stage_out=all_files_to_stage_out)

    def stage_files_out(self, output_files: list[Path]) -> list[Path]:
        """Stage out the provided files.

        Args:
            output_files: Files to stage out.
        Returns:
            Paths of the files that were staged out, in their permanent locations.
        """
        return self._stage_files_out(files_to_stage_out=output_files)


def retrieve_working_dir(dm: DataManager, executor: str) -> str:
    """Retrieve the working directory for the executor.

    This is a convenience function to retrieve the working directory for the executor.
    It is a property of the HTEX executor, so it is used here to avoid repeating the
    same code in multiple places.

    Args:
        dm: The data manager.
        executor: The executor for which to retrieve the working directory.

    Returns:
        The working directory for the executor.
    """
    # This is a property of the HTEX executor. It will fail otherwise!
    if not hasattr(dm.dfk.executors[executor], "working_dir"):
        msg = f"Executor {executor} does not have a working_dir property! Needs HTEX"
        raise ValueError(msg)
    working_dir: str = dm.dfk.executors[executor].working_dir  # type: ignore[attr-defined]
    return working_dir


class RSyncStagingForParsl(Staging, RepresentationMixin):
    """Sync locally accessible files between worker nodes and storage directories.

    Based on RSyncStaging from parsl, with modifications for our purposes
    to focus on syncing local files.

    This staging provider will execute rsync on worker nodes to stage in files
    from a remote location.
    """

    def __init__(self) -> None:
        pass

    def can_stage_in(self, file: File) -> bool:
        return file.scheme == "file"

    def can_stage_out(self, file: File) -> bool:
        return file.scheme == "file"

    def stage_in(
        self,
        dm: DataManager,
        executor: str,
        file: parsl.File,
        parent_fut: concurrent.futures.Future[Any] | None,  # noqa: ARG002
    ) -> DataFuture | None:
        """Prepare for stage-in.

        Copied from the Staging class docs:

        This call gives the staging provider an opportunity to prepare for
        stage-in and to launch arbitrary tasks which must complete as part
        of stage-in.

        This call will be made with a fresh copy of the File that may be
        modified for the purposes of this particular staging operation,
        rather than the original application-provided File. This allows
        staging specific information (primarily localpath) to be set on the
        File without interfering with other stagings of the same File.

        The call can return a:
          - DataFuture: the corresponding task input parameter will be
            replaced by the DataFuture, and the main task will not
            run until that DataFuture is complete. The DataFuture result
            should be the file object as passed in.
          - None: the corresponding task input parameter will be replaced
            by a suitable automatically generated replacement that container
            the File fresh copy, or is the fresh copy.
        """
        # we need to make path an absolute path, because
        # rsync remote name needs to include absolute path
        file.path = str(Path(file.path).resolve())

        working_dir = retrieve_working_dir(dm, executor)

        if working_dir:
            file.local_path = str(Path(working_dir) / file.filename)
        else:
            file.local_path = file.filename

        return None

    def stage_out(
        self,
        dm: DataManager,
        executor: str,
        file: File,
        app_fu: concurrent.futures.Future[Any],  # noqa: ARG002
    ) -> concurrent.futures.Future[Any] | None:
        """Prepare for stage-out.

        Copied from the Staging class docs:

        This call gives the staging provider an opportunity to prepare for
        stage-out and to launch arbitrary tasks which must complete as
        part of stage-out.

        Even though it should set up stageout, it will be invoked before
        the task executes. Any work which needs to happen after the main task
        should depend on app_fu.

        For a given file, either return a Future which completes when stageout
        is complete, or return None to indicate that no stageout action need
        be waited for. When that Future completes, parsl will mark the relevant
        output DataFuture complete.

        Note the asymmetry here between stage_in and stage_out: this can return
        any Future, while stage_in must return a DataFuture.
        """

        file.path = str(Path(file.path).resolve())

        working_dir = retrieve_working_dir(dm, executor)

        if working_dir:
            file.local_path = str(Path(working_dir) / file.filename)
        else:
            file.local_path = file.filename

        return None

    def replace_task(
        self, dm: DataManager, executor: str, file: File, func: Callable[[Any], Any]
    ) -> Callable[[Any], Any] | None:
        logger.debug(f"Replacing task for rsync stagein, {file}")
        working_dir = retrieve_working_dir(dm, executor)
        return in_task_stage_in_wrapper(func, file, working_dir)

    def replace_task_stage_out(
        self, dm: DataManager, executor: str, file: File, func: Callable[[Any], Any]
    ) -> Callable[[Any], Any] | None:
        logger.debug(f"Replacing task for rsync stageout, {file}")
        working_dir = retrieve_working_dir(dm, executor)
        return in_task_stage_out_wrapper(func, file, working_dir)


def in_task_stage_in_wrapper(
    func: Callable[[Any], Any],
    file: File,
    working_dir: str,
) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import logging
        import os
        import subprocess
        from pathlib import Path

        logger = logging.getLogger(__name__)
        logger.debug("rsync in_task_stage_in_wrapper start")

        if working_dir:
            # Also need to update the working dir
            # NOTE: We need a new variable name because working_dir is defined
            #       via the closure...
            _expanded_working_dir = os.path.expandvars(working_dir)
            Path(_expanded_working_dir).mkdir(exist_ok=True)

        # We need to expand environment variables for the files local_path,
        # but we need to do it in a way that is persistent. To do so, we update
        # the inputs list in the kwargs.
        # NOTE: Since we're staging in, we want to look at the inputs.
        # NOTE: By waiting until here to expand the variables, we know we're on
        #       the node and therefore have the right environment.
        existing_directories = [Path(working_dir)] if working_dir else []
        for input_file in kwargs.get("inputs", []):
            input_file.local_path = os.path.expandvars(input_file.local_path)
            # Further, we may need to create a directory for the file. We first
            # check against existing directories to avoid unnecessary I/O
            p = Path(input_file.local_path)
            if p.parent not in existing_directories:
                p.parent.mkdir(parents=True, exist_ok=True)
                existing_directories.append(p)

        # Validation
        assert file.local_path, "file.local_path must be set!"

        logger.debug("rsync in_task_stage_in_wrapper calling rsync")
        # Move the file from the permanent location to the worker location
        try:
            subprocess.run(["rsync", file.path, file.local_path], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Most likely own't get the logging info from here, so better to encode it in the error message
            logger.exception(e)
            msg = f"Failed to execute rsync in subprocess with args: {file.path} {file.local_path}"
            msg += f"\nstdout: {e.stdout.decode()}"
            msg += f"\nstderr: {e.stderr.decode()}"
            raise RuntimeError(msg) from e
        logger.debug("rsync in_task_stage_in_wrapper calling wrapped function")
        result = func(*args, **kwargs)
        logger.debug("rsync in_task_stage_in_wrapper returned from wrapped function")
        return result

    return wrapper


def in_task_stage_out_wrapper(
    func: Callable[[Any], Any],
    file: File,
    working_dir: str,  # noqa: ARG001
) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import logging
        import os
        import subprocess
        from pathlib import Path

        logger = logging.getLogger(__name__)
        logger.debug("rsync in_task_stage_out_wrapper start")

        # We need to expand environment variables for the files local_path,
        # but we need to do it in a way that is persistent. To do so, we update
        # the outputs list in the kwargs.
        # NOTE: Since we're staging out, we want to look at the outputs.
        # NOTE: By waiting until here to expand the variables, we know we're on
        #       the node and therefore have the right environment.
        for output in kwargs.get("outputs", []):
            output.local_path = os.path.expandvars(output.local_path)

        # Validation
        assert file.local_path, "file.local_path must be set!"

        # Execute the wrapper function, running the actual app.
        logger.debug("rsync in_task_stage_out_wrapper calling wrapped function")
        result = func(*args, **kwargs)
        logger.debug("rsync in_task_stage_out_wrapper returned from wrapped function, calling rsync")
        # Move the file from the worker location to the permanent location
        try:
            subprocess.run(["rsync", file.local_path, file.path], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Most likely own't get the logging info from here, so better to encode it in the error message
            logger.exception(e)
            msg = f"Failed to execute rsync in subprocess with args: {file.local_path} {file.path}"
            msg += f"\nstdout: {e.stdout.decode()}"
            msg += f"\nstderr: {e.stderr.decode()}"
            raise RuntimeError(msg) from e
        logger.debug("rsync in_task_stage_out_wrapper returned from rsync")

        # Now, cleanup by removing the staged file now that it's been synced.
        try:
            # NOTE: We need to expand the variables again because this seems to be a copy
            #       of the files that are in outputs. Which is to say that the local_path
            #       isn't expanded in the `file`.
            Path(os.path.expandvars(file.local_path)).unlink()
        except FileNotFoundError:
            # No point in raising here. The purpose was to cleanup. It didn't work
            # for some reason, which may be a bit surprising so we log it, but no
            # need to break execution.
            logger.warning(f"Unable to remove file: {file.local_path}")

        return result

    return wrapper
