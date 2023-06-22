""" Download LEGO train output file list stored in YAML

Ported from jet_substructure. The code here isn't amazing (nor is the pachyderm interface),
but it works, so no point in messing with it (as of June 2023).

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path

from pachyderm import yaml
from pachyderm.alice import download

import mammoth.helpers

logger = logging.getLogger(__name__)


def create_file_pairs(input_filename: Path = Path("files_to_download.yaml")) -> list[download.FilePair]:
    """Create file pairs from a input YAML file.

    Reads a YAML file consisting of a dict with entries of:
    alien_file_path: local_file_path

    Args:
        input_filename: Filename of the YAML file containing the file pairs
    Returns:
        List of file pairs
    """
    y = yaml.yaml()
    with input_filename.open() as f:
        file_list_input = y.load(f)

    file_pairs = []
    for alien_file, local_file in file_list_input.items():
        file_pairs.append(download.FilePair(source=alien_file, target=local_file))

    logger.debug(f"file_pairs: {file_pairs}")

    return file_pairs


def run() -> None:
    """Download the file list.

    Not very sophisticated or configurable, but good enough.

    NOTE:
        It assumes the YAML file with the file pairs is called `files_to_download.yaml` in
        the current working directory.
    """
    mammoth.helpers.setup_logging(level=logging.INFO)

    file_pairs = create_file_pairs()

    # Setup the queue and filler, and then start downloading.
    q: download.FilePairQueue = queue.Queue()
    queue_filler = download.FileListDownloadFiller(pairs=file_pairs, q=q)
    download.download(queue_filler=queue_filler, q=q)


if __name__ == "__main__":
    run()
