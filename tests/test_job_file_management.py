"""Tests for job file management.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mammoth import job_file_management


def test_stage_files_in() -> None:
    """Test staging in files."""
    # Create a temporary directory with a nested structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        permanent_work_dir = temp_dir_path / "permanent"
        node_work_dir = temp_dir_path / "worker_node"
        permanent_work_dir.mkdir(parents=True)
        node_work_dir.mkdir(parents=True)

        # Create test files at different depths
        files = [
            permanent_work_dir / "dir1" / "dir2" / "file1.txt",
            permanent_work_dir / "dir1" / "file2.txt",
            permanent_work_dir / "file3.txt",
        ]
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()
        # Create an instance of FileStaging
        fs = job_file_management.FileStaging(permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir)

        # Stage in the files
        staged_files = fs.stage_files_in(files)

        # Verify that the files are staged in to the correct location
        expected_staged_files = [fs.node_work_dir_input / file.relative_to(permanent_work_dir) for file in files]
        assert staged_files == expected_staged_files
        # And verify that they actually exist!
        for staged_file in staged_files:
            assert staged_file.exists()

        # Clean up the staged files
        fs.clean_up_staged_in_files_after_task()

        # Verify that the staged files are cleaned up
        assert not any(staged_file.exists() for staged_file in staged_files)


def test_stage_files_out() -> None:
    """Test staging out files."""
    # Create a temporary directory with a nested structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        permanent_work_dir = temp_dir_path / "permanent"
        node_work_dir = temp_dir_path / "worker_node"
        permanent_work_dir.mkdir(parents=True)
        node_work_dir.mkdir(parents=True)

        # Create an instance of FileStaging
        fs = job_file_management.FileStaging(permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir)
        # Create test files at different depths in the node work dir
        node_files = [
            fs.node_work_dir_output / "dir1" / "dir2" / "file1.txt",
            fs.node_work_dir_output / "dir1" / "file2.txt",
            fs.node_work_dir_output / "file3.txt",
        ]
        for f in node_files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
        # Stage out the files
        staged_out_files = fs.stage_files_out(node_files)
        # Verify that the files are staged out to the correct location
        expected_staged_files = [
            permanent_work_dir / "dir1" / "dir2" / "file1.txt",
            permanent_work_dir / "dir1" / "file2.txt",
            permanent_work_dir / "file3.txt",
        ]
        assert staged_out_files == expected_staged_files
        # And verify that they actually exist!
        for staged_file in staged_out_files:
            assert staged_file.exists()
        # Verify that the staged files are cleaned up
        assert not any(node_file.exists() for node_file in node_files)


# TODO: Test staging out all files vs the list!
@pytest.mark.parametrize("staging_options", ["context_manager", "wo_task_wrapper", "wo_task_wrapper_glob_output_files"])
def test_integration(staging_options: str) -> None:
    """Integration test for staging in and staging out."""
    # Create a temporary directory with a nested structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        temp_dir_path = Path(temp_dir)
        permanent_work_dir = temp_dir_path / "permanent"
        node_work_dir = temp_dir_path / "worker_node"
        permanent_work_dir.mkdir(parents=True)
        node_work_dir.mkdir(parents=True)

        # Create test files at different depths in the permanent work dir
        permanent_files = [
            permanent_work_dir / "dir1" / "dir2" / "file1.txt",
            permanent_work_dir / "dir1" / "file2.txt",
            permanent_work_dir / "file3.txt",
        ]
        for f in permanent_files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()

        # Create a task to be wrapped.
        # In this case, we use it to create output files as it's only task.
        # However, it could be doing anything...
        def generate_output_files(
            input_files: list[Path],  # noqa: ARG001
            output_dir: Path,
        ) -> list[Path]:
            """Generate output files.

            This is a stand in for a more complex task that would be wrapped by
            the FileStaging class.
            """
            output_files = [
                output_dir / "output1.txt",
                output_dir / "output2.txt",
                output_dir / "dir_new" / "output3.txt",
            ]
            for f in output_files:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.touch()
            return output_files

        # Create an instance of FileStaging
        fs = job_file_management.FileStaging(permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir)
        if staging_options == "context_manager":
            with job_file_management.potential_file_staging(file_staging=fs, input_files=permanent_files):
                node_path_files_to_stage_out = generate_output_files(
                    input_files=permanent_files, output_dir=fs.node_work_dir_output
                )
                # NOTE: We call the result the output_files, but that's just for convenience in checking
                #       the test. It could be any result! In fact, we generally wouldn't want to return
                #       the output_files since they would point at the node work dir, which is temporary.
                #       (this won't be an issue in our framework because we can take care of this for a user).

                # We'll have to derive the staged_in_files and staged_out_files since we don't
                # have direct access to the outputs when we use the wrapper.
                node_path_files_staged_in = fs._staged_in_files
                staged_out_files = [fs.translated_node_to_permanent_path(f) for f in node_path_files_to_stage_out]
        else:
            # Stage in the permanent files
            node_path_files_staged_in = fs.stage_files_in(files_to_stage_in=permanent_files)
            # Generate the output files
            node_path_files_to_stage_out = generate_output_files(
                input_files=node_path_files_staged_in, output_dir=fs.node_work_dir_output
            )
            # Stage out the output files
            if "glob_output_files" in staging_options:
                staged_out_files = fs.stage_all_files_out()
            else:
                staged_out_files = fs.stage_files_out(files_to_stage_out=node_path_files_to_stage_out)
            # Clean up the staged files
            fs.clean_up_staged_in_files_after_task()

        # Verify that the files are staged in and staged out to the correct locations
        expected_staged_in_files = sorted([fs.translate_permanent_to_node_path(f) for f in permanent_files])
        expected_staged_out_files = sorted(
            [
                permanent_work_dir / "output1.txt",
                permanent_work_dir / "output2.txt",
                permanent_work_dir / "dir_new" / "output3.txt",
            ]
        )
        assert sorted(node_path_files_staged_in) == expected_staged_in_files
        assert sorted(staged_out_files) == expected_staged_out_files
        # And verify that the staged out files actually exist on the permanent storage!
        assert all(staged_file.exists() for staged_file in staged_out_files)

        # Also check that the input files haven't been distributed
        assert all(permanent_file.exists() for permanent_file in permanent_files)

        # Verify that the staged in files are cleaned up
        # NOTE: The staged_in_files won't exist because the wrapper has already taken
        #       care of the cleanup!
        assert not any(staged_file.exists() for staged_file in node_path_files_staged_in)
        # Verify that the staged out files on the node are cleaned up
        assert not any(staged_file.exists() for staged_file in node_path_files_to_stage_out)
