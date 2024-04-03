"""Some tests for file management.

Based on the rsync provider. Modifications include:

- Removed remote system support - it only copies locally.
- Staged files are cleaned up after they are staged back.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import concurrent.futures
import logging
import os.path
import shutil
import signal
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, overload

import attrs
import parsl
from parsl.app.futures import DataFuture
from parsl.data_provider.data_manager import DataManager
from parsl.data_provider.files import File
from parsl.data_provider.staging import Staging
from parsl.utils import RepresentationMixin

logger = logging.getLogger(__name__)


@attrs.define(frozen=True)
class FileStagingSettings:
    """Settings for file staging.

    NOTE:
        The unique ID ensures that jobs don't interfere with each other. It costs some in IO,
        but it should be worth it to avoid potential conflicts (especially for staging out files).

    Attributes:
        permanent_work_dir: Permanent work directory for storing files.
        node_work_dir: Work directory on the worker node for storing files with the unique ID.
        _node_work_dir: Work directory on the worker node for storing files without the unique ID.
            You usually want it **with** the unique ID (hence this being a private attribute).
        _unique_id: Unique ID for the node work directory.
    """

    permanent_work_dir: Path
    _node_work_dir: Path
    _unique_id: str = attrs.field(init=False, factory=lambda: str(uuid.uuid4())[:10])

    @property
    def node_work_dir(self) -> Path:
        return self._node_work_dir / self._unique_id

    def make_unique_copy(self, expand_env_vars_in_node_work_dir: bool) -> FileStagingSettings:
        """Create a unique copy of the path manager.

        Args:
            expand_env_vars_in_node_work_dir: Whether to expand environment variables in the node work dir.
                Usually, you'll want to save this for when you're on the worker node.
        Returns:
            A new `FileStagingPaths` object with a unique ID.
        """
        # NOTE: We intentionally retrieve the node work dir without the unique ID, since we want
        #       a new value in the copy.
        node_work_dir = self._node_work_dir
        if expand_env_vars_in_node_work_dir:
            node_work_dir = Path(os.path.expandvars(node_work_dir))
        return FileStagingSettings(permanent_work_dir=self.permanent_work_dir, node_work_dir=node_work_dir)

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

    def translate_input_permanent_to_node_path(self, permanent_path: Path) -> Path:
        """Translate a file path from the permanent directory to the worker node directory.

        Args:
            permanent_path: Path to a file in the permanent directory.
        Returns:
            Path to the file in the worker node directory.
        """
        return self.node_work_dir_input / permanent_path.relative_to(self.permanent_work_dir)

    def translate_output_permanent_to_node_path(self, permanent_path: Path) -> Path:
        """Translate a predetermined output path from the permanent directory to the worker node directory.

        Args:
            permanent_path: Path to a file in the permanent directory.
        Returns:
            Path to the file in the worker node directory.
        """
        return self.node_work_dir_output / permanent_path.relative_to(self.permanent_work_dir)

    def translate_output_node_to_permanent_path(self, node_path: Path) -> Path:
        """Translate a file path in the worker node directory to the permanent directory.

        Args:
            node_path: Path to a file in the worker node directory.
        Returns:
            Path to the file in the permanent directory.
        """
        return self.permanent_work_dir / node_path.relative_to(self.node_work_dir_output)


@attrs.define(frozen=True)
class FileStager:
    """Performs file staging within a job according to the provided settings.

    Best used in conjunction with the `FileStagingManager` context manager! Otherwise,
    there's a lot details to manage (e.g. file cleanup), that aren't handled by this class.
    All it does is move files and some basic cleanup (but only in cases where we can be
    100% confident that it's safe).

    The concept here is that there are two directories:
        - The permanent directory, which is where the files are stored permanently.
        - The worker node directory, which is where the files are stored on the worker node.
    Within this concept, we keep the directory structure as consistent as possible, even
    on the worker node, eg.:

    ```python
    fs = FileStaging(
        settings=FileStagingSettings(
            permanent_work_dir=Path("/path/to/permanent"),
            node_work_dir=Path("/path/to/worker_node"),
        )
    )
    # Stage in
    input_file = Path("/path/to/permanent/some_directory_path/input_file.txt")
    result = fs.stage_files_in(files_to_stage_in=[input_file])
    assert result[0] == Path(
        "/path/to/worker_node/<unique_ID>/input/some_directory_path/input_file.txt"
    )
    ```

    and then staging out will undo this, again as appropriate. e.g.:

    ```python
    fs = FileStaging(
        settings=FileStagingSettings(
            permanent_work_dir=Path("/path/to/permanent"),
            node_work_dir=Path("/path/to/worker_node"),
        )
    )
    # Stage out
    output_file = Path(
        "/path/to/worker_node/<unique_ID>/output/some_output_directory_path/output_file.txt"
    )
    result = fs.stage_files_out(files_to_stage_out=[output_file])
    assert result[0] == Path(
        "/path/to/permanent/<unique_ID>/some_output_directory_path/output_file.txt"
    )
    ```

    Attributes:
        settings: File staging settings.
    """

    settings: FileStagingSettings

    def stage_files_in(self, files_to_stage_in: list[Path]) -> list[Path]:
        """Stage files in.

        Args:
            files_to_stage_in: Files to stage in.
        Returns:
            Paths of the files that were staged in, in their worker node locations.
        """
        logger.info("Staging in files")
        # Setup as necessary
        stage_in_dir = self.settings.node_work_dir_input
        stage_in_dir.mkdir(parents=True, exist_ok=True)

        # Stage in the files, relative to the permanent directory.
        modified_paths: list[Path] = []
        # Group the directories to create together to optimize IO
        directories_to_create: set[Path] = set()

        for f in files_to_stage_in:
            p = self.settings.translate_input_permanent_to_node_path(f)
            modified_paths.append(p)
            directories_to_create.add(p.parent)

        # Create the directories
        for d in directories_to_create:
            d.mkdir(parents=True, exist_ok=True)

        # Finally, move the files
        for f, p in zip(files_to_stage_in, modified_paths, strict=True):
            logger.debug(f"Copying permanent:'{f.relative_to(self.settings.permanent_work_dir)}' -> node:'{p}'")
            shutil.copy(f, p)

        return modified_paths

    def stage_files_out(self, files_to_stage_out: list[Path] | None = None) -> list[Path]:
        """Stage out the provided files.

        Args:
            files_to_stage_out: Files to stage out, using their paths on the worker node.
                Default: None. In that case, we stage out al files in the output directory.
        Returns:
            Paths of the files that were staged out, in their permanent locations.
        """
        # We won't always known the output files beforehand. In that case, we want
        # to just stage out all files in the output directory.
        if files_to_stage_out is None:
            # NOTE: Could also use shutil.copytree(src, dest, dir_exist_ok=True), but
            #       this is more convenient since we already implemented the copying, so
            #       just leave it as is for now...
            # NOTE: We have to take the hit on the is_file check because otherwise it will include
            #       directories in the glob.
            files_to_stage_out = [v for v in self.settings.node_work_dir_output.rglob("*") if v.is_file()]
        return self._stage_files_out(files_to_stage_out=files_to_stage_out)

    def _stage_files_out(self, files_to_stage_out: list[Path]) -> list[Path]:
        """Stage files out implementation.

        Args:
            files_to_stage_out: Files to stage out.
        Returns:
            Paths of the files that were staged out, in their permanent locations.
        """
        logger.info("Staging out files")
        # Stage out the files, relative to the permanent directory.
        modified_paths: list[Path] = []
        # Group the directories to create together to optimize IO
        directories_to_create = set()
        # Calculate the output paths
        for f in files_to_stage_out:
            p = self.settings.translate_output_node_to_permanent_path(f)
            modified_paths.append(p)
            directories_to_create.add(p.parent)

        # Create the directories
        for d in directories_to_create:
            d.mkdir(parents=True, exist_ok=True)

        # Finally, move the files out
        for f, p in zip(files_to_stage_out, modified_paths, strict=True):
            try:
                logger.debug(f"Copying node:'{f}' -> permanent:'{p.relative_to(self.settings.permanent_work_dir)}'")
                shutil.copy(f, p)
            except OSError as e:
                logger.exception(e)
                continue
            # If we've succeeded in copying, we can remove the existing file on the worker node.
            f.unlink()
        try:
            # Double check that there are not any files left in the tree before removing the whole thing.
            # NOTE: This does take some IO, but it's on the node, so it should be relatively cheap.
            remaining_files = [f for f in self.settings.node_work_dir_output.rglob("*") if f.is_file()]
            if any(remaining_files):
                msg = (
                    f"Wanted to remove node output directory: {self.settings.node_work_dir_output}"
                    f", but it still contains files: {remaining_files}"
                )
                raise RuntimeError(msg)
            # If there aren't any files left, then we're all good to remove the directory
            shutil.rmtree(self.settings.node_work_dir_output)
        except (RuntimeError, OSError) as e:
            # If this fails, it doesn't prevent further operations, so we just log it and move on.
            logger.exception(e)
            # Add some additional information, so we can figure out how it went wrong in the future.
            msg = (
                f"Failed to remove node output work directory: {self.settings.node_work_dir_output}"
                f" containing {list(self.settings.node_work_dir_output.rglob('*'))}"
            )
            logger.warning(msg)

        return modified_paths


@attrs.define
class FileStagingManager:
    """Manage file staging.

    This is the preferred interface for actually staging files. It takes care of many details
    that you would otherwise have to worry about in e.g. cleaning up files as appropriate.

    Note that we assume that we will clean up the staged in files. This is almost certainly
    what you want to do.

    NOTE:
        If file staging is disabled, then we just provide the manager and don't do
        anything else. i.e. in that case, it's effectively a no-op.

    The file list attributes are marked as private because we don't want the user to try to
    interact directly with them and get confused. It's better if they have to make explicit
    (but obvious) choices to get the right paths.

    There's no `_output_files_post_stage_out` because we don't need to keep track of them.
    Further, the context manager will be out of scope by then, so we can't access this value
    anyway by the time we would be interested. So it's better to just not keep track of it.

    Attributes:
        file_staging: File staging object.
        _input_files: List of input files to potentially stage in from the permanent directory
            to the worker node.
        _output_files: List of output files to potentially stage out from the worker node to
            the permanent directory. Default: None, which means that we will stage out all files
            that are in the output directory.
        _input_files_post_stage_in: List of input files that were staged in, after the stage in operation.
    """

    file_stager: FileStager | None
    _input_files: list[Path]
    _output_files: list[Path] | None = None
    _input_files_post_stage_in: list[Path] = attrs.field(init=False, factory=list)

    @classmethod
    def from_settings(
        cls, settings: FileStagingSettings | None, input_files: list[Path], output_files: list[Path] | None = None
    ) -> FileStagingManager:
        if settings is None:
            # We don't want the manager to do anything, so don't provide a stager. This will ensure
            # that this manager is effectively a no-op. We pass the other arguments for completeness,
            # but they aren't needed.
            return cls(file_stager=None, input_files=input_files, output_files=output_files)
        return cls(
            file_stager=FileStager(settings=settings.make_unique_copy(expand_env_vars_in_node_work_dir=True)),
            input_files=input_files,
            output_files=output_files,
        )

    @property
    def staging_enabled(self) -> bool:
        return self.file_stager is not None

    def translate_input_paths(self, paths: list[Path]) -> list[Path]:
        """Translate input file paths to the path to use for the task.

        Args:
            paths: Paths to the input files.
        Returns:
            Translated paths to the input files.
        """
        # Translate as appropriate if we are staging.
        if self.file_stager:
            return [self.file_stager.settings.translate_input_permanent_to_node_path(f) for f in paths]
        # If we're not staging, then we don't need to do anything
        return paths

    def translate_output_paths(self, paths: list[Path]) -> list[Path]:
        """Translate output file paths to the path to use for the task.

        Args:
            paths: Paths to the output files.
        Returns:
            Translated paths to the output files.
        """
        if self.file_stager:
            return [self.file_stager.settings.translate_output_permanent_to_node_path(f) for f in paths]
        return paths

    def translate_paths(self, *, input_files: list[Path], output_files: list[Path]) -> tuple[list[Path], list[Path]]:
        """Convenience function to translate both input and output paths.

        NOTE:
            Thus function requires both input and output files. If you only need one,
            then just use the dedicated functions.

        Args:
            input_files: Paths to the input files.
            output_files: Paths to the output files.
        Returns:
            Translated paths to the input and output files.
        """
        return self.translate_input_paths(input_files), self.translate_output_paths(output_files)

    def __enter__(self) -> FileStagingManager:
        """Entering the context manager, staging in files if appropriate."""
        # Register ctrl-c handler
        signal.signal(signal.SIGINT, self._sigint_handler)
        # If not, there's nothing to be done
        if self.file_stager:
            # If file staging is valid, then ensure we stage in.
            self._input_files_post_stage_in = self.file_stager.stage_files_in(files_to_stage_in=self._input_files)
            # And that we create the basic output dir. Some codes may forget to do this,
            # so it's convenient to take care of it here.
            # NOTE: We only want to do it if file_staging is valid since it's likely to exist
            #       if we're not actually staging. So may as well save the IO.
            self.file_stager.settings.node_work_dir_output.mkdir(parents=True, exist_ok=True)
        # Then we return self to keep track of it in the context manager
        return self

    @overload
    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        ...

    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exiting the context manager, staging out files if appropriate."""
        if self.file_stager:
            # First, clean up the staged in files so we don't forget.
            self._clean_up_staged_in_files_after_task()
            # And stage out, using the provided output files if appropriate.
            self.file_stager.stage_files_out(files_to_stage_out=self._output_files)
            # And then the last bit of cleanup - removing the node work dir itself
            try:
                self.file_stager.settings.node_work_dir.rmdir()
            except OSError as e:
                # If this fails, it doesn't prevent further operations, so we just log it and move on.
                logger.exception(e)
                # Add some additional information, so we can figure out how it went wrong in the future.
                msg = (
                    f"Failed to remove node origin work directory: {self.file_stager.settings.node_work_dir}"
                    f" containing {list(self.file_stager.settings.node_work_dir.rglob('*'))}"
                )
                logger.warning(msg)
        # If we're not staging, there's nothing to be done

    def _sigint_handler(self, signal_received: int, frame: FrameType | None) -> None:  # noqa: ARG002
        """Handle if ctrl-c is pressed.

        Based on: https://aalvarez.me/posts/gracefully-exiting-python-context-managers-on-ctrl-c/

        This ensures that we can clean up the staged in and out files if the task is interrupted.
        """
        logger.warning("Detected Ctrl + C handler called. Performing remaining staging...")
        # Calling __exit__ should take care of all of the staging and cleanup.
        self.__exit__(None, None, None)

        sys.exit(0)

    def _clean_up_staged_in_files_after_task(self) -> None:
        """Clean up the staged in files after the task completes."""
        if self.file_stager:
            for f in self._input_files_post_stage_in:
                f.unlink()
            # NOTE: Although we've remove the files, this still leaves the whole directory structure,
            #       so we also want to remove the input directory itself. This will only fail on
            #       read-only files, so it would be easy to lose important data if we're not careful.
            #       To address this, we do a check for remaining files and raise an error if there are
            #       any. This of course has an IO cost, but again, we're on the local storage of a
            #       worker node, so it should be relatively cheap.
            try:
                # Double check that there are not any files left in the tree before removing the whole thing.
                # NOTE: This does take some IO, but it's on the node, so it should be relatively cheap.
                remaining_files = [f for f in self.file_stager.settings.node_work_dir_input.rglob("*") if f.is_file()]
                if any(remaining_files):
                    msg = (
                        f"Wanted to remove node input directory: {self.file_stager.settings.node_work_dir_input}"
                        f", but it still contains files: {remaining_files}"
                    )
                    raise RuntimeError(msg)
                # If there aren't any files left, then we're all good to remove the directory
                shutil.rmtree(self.file_stager.settings.node_work_dir_input)
            except (RuntimeError, OSError) as e:
                # If this fails, it doesn't prevent further operations, so we just log it and move on.
                logger.exception(e)
                # Add some additional information, so we can figure out how it went wrong in the future.
                msg = (
                    f"Failed to remove node input work directory: {self.file_stager.settings.node_work_dir_input}"
                    f" containing {list(self.file_stager.settings.node_work_dir_input.rglob('*'))}"
                )
                logger.warning(msg)

            # And clear the list since we're done.
            self._input_files_post_stage_in = []


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
