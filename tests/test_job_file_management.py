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
        settings = job_file_management.FileStagingSettings(
            permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir
        )
        fs = job_file_management.FileStager(settings=settings)

        # Stage in the files
        staged_files = fs.stage_files_in(files)

        # Verify that the files are staged in to the correct location
        expected_staged_files = [fs.settings.translate_input_permanent_to_node_path(file) for file in files]
        assert staged_files == expected_staged_files
        # And verify that they actually exist!
        for staged_file in staged_files:
            assert staged_file.exists()

        # Clean up the staged files
        # NOTE: The manager is now responsible for cleaning up the staged files!
        #       We'll cheat by creating a manager here and setting it up manually
        staging_manager = job_file_management.FileStagingManager(file_stager=fs, input_files=files)
        staging_manager._input_files_post_stage_in = staged_files
        staging_manager._clean_up_staged_in_files_after_task()

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
        settings = job_file_management.FileStagingSettings(
            permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir
        )
        fs = job_file_management.FileStager(settings=settings)
        # Create test files at different depths in the node work dir
        node_files = [
            fs.settings.node_work_dir_output / "dir1" / "dir2" / "file1.txt",
            fs.settings.node_work_dir_output / "dir1" / "file2.txt",
            fs.settings.node_work_dir_output / "file3.txt",
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

        # We now use the manager, so we don't need to clean up the staged files here
        # since it's __NOT__ the responsibility of the FileStager to clean up the files.
        # assert not any(node_file.exists() for node_file in node_files)


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
            input_files: list[Path],
            relevant_stager_settings: job_file_management.FileStagingSettings,
            output_dir: Path,
        ) -> list[Path]:
            """Generate output files.

            This is a stand in for a more complex task that would be wrapped by
            the FileStaging class.
            """
            # Check that we have the node worker input files!
            expected_staged_in_files = sorted(
                [relevant_stager_settings.translate_input_permanent_to_node_path(f) for f in permanent_files]
            )
            assert sorted(input_files) == expected_staged_in_files

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
        settings = job_file_management.FileStagingSettings(
            permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir
        )
        if staging_options == "context_manager":
            with job_file_management.FileStagingManager.from_settings(
                settings=settings, input_files=permanent_files
            ) as staging_manager:
                translated_input_files = staging_manager.translate_input_paths(paths=permanent_files)
                node_path_files_to_stage_out = generate_output_files(
                    input_files=translated_input_files,
                    # NOTE: It's important that we pass the staging_manager settings here rather than the
                    #       general settings because `from_settings` will create a unique settings object,
                    #       which won't match up with the standard settings object. In fact, we'll cross
                    #       check that below.
                    relevant_stager_settings=staging_manager.file_stager.settings,
                    output_dir=staging_manager.translate_output_paths([permanent_work_dir])[0],
                )
                # NOTE: We call the result the output_files, but that's just for convenience in checking
                #       the test. It could be any result! In fact, we generally wouldn't want to return
                #       the output_files since they would point at the node work dir, which is temporary.
                #       (this won't be an issue in our framework because we can take care of this for a user).

                # We'll have to derive the staged_in_files and staged_out_files since we don't
                # have direct access to the outputs when we use the wrapper.
                node_path_files_staged_in = staging_manager._input_files_post_stage_in
                staged_out_files = [
                    staging_manager.file_stager.settings.translate_output_node_to_permanent_path(f)
                    for f in node_path_files_to_stage_out
                ]
                # Cross check the statement above about the right manager settings
                assert staging_manager.file_stager.settings.node_work_dir.parent == settings.node_work_dir.parent
                assert staging_manager.file_stager.settings.node_work_dir != settings.node_work_dir
                assert staging_manager.file_stager.settings.node_work_dir_input != settings.node_work_dir_input

                # From here, we'll update the settings to use the manager settings to avoid further confusion in the checks below...
                settings = staging_manager.file_stager.settings
        else:
            fs = job_file_management.FileStager(settings=settings)
            # Stage in the permanent files
            node_path_files_staged_in = fs.stage_files_in(files_to_stage_in=permanent_files)
            # Generate the output files
            node_path_files_to_stage_out = generate_output_files(
                input_files=node_path_files_staged_in,
                relevant_stager_settings=settings,
                output_dir=fs.settings.node_work_dir_output,
            )
            # Stage out the output files
            if "glob_output_files" in staging_options:
                staged_out_files = fs.stage_files_out()
            else:
                staged_out_files = fs.stage_files_out(files_to_stage_out=node_path_files_to_stage_out)
            # Clean up the staged files
            # NOTE: The manager is now responsible for cleaning up the staged files!
            #       We'll cheat by creating a manager here and setting it up manually
            staging_manager = job_file_management.FileStagingManager(file_stager=fs, input_files=permanent_files)
            staging_manager._input_files_post_stage_in = node_path_files_staged_in
            staging_manager._clean_up_staged_in_files_after_task()

        # Verify that the files are staged in and staged out to the correct locations
        expected_staged_in_files = sorted([settings.translate_input_permanent_to_node_path(f) for f in permanent_files])
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


@pytest.mark.parametrize("actually_stage_files", [True, False])
def test_staging_manager(actually_stage_files: bool) -> None:
    """Tests for the staging manager.

    Includes testing that it does nothing when the file staging is disabled
    """
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

        settings = None
        if actually_stage_files:
            settings = job_file_management.FileStagingSettings(
                permanent_work_dir=permanent_work_dir, node_work_dir=node_work_dir
            )

        with job_file_management.FileStagingManager.from_settings(
            settings=settings, input_files=permanent_files
        ) as staging_manager:
            translated_input_files = staging_manager.translate_input_paths(paths=permanent_files)
            translated_output_files = staging_manager.translate_output_paths([permanent_work_dir])

            if actually_stage_files:
                # NOTE: It's important that we use the settings in the staging_manager here rather than the
                #       general settings because `from_settings` will create a unique settings object,
                #       which won't match up with the standard settings object. We update the settings to use
                #       the manager settings to avoid further confusion in the checks below...
                settings = staging_manager.file_stager.settings
                # Input
                comparison_files_input = {settings.translate_input_permanent_to_node_path(f) for f in permanent_files}
                assert set(translated_input_files) == comparison_files_input
                # Output
                assert set(translated_output_files) == {settings.node_work_dir_output}
            else:
                # Input
                comparison_files_input = set(permanent_files)
                assert set(translated_input_files) == comparison_files_input
                # Output
                assert set(translated_output_files) == {permanent_work_dir}
