"""File sources for IO.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import copy
import inspect
from functools import partial
from typing import Any

from mammoth.framework import sources
from mammoth.framework.io import HF_tree, jet_extractor, jetscape, jewel_from_laura, track_skim

file_source_registry: dict[str, sources.DelayedSource] = {
    "HF_tree": HF_tree.FileSource,
    "HF_tree_at_LBL": HF_tree.FileSource,
    "jet_extractor_jewel": jet_extractor.JEWELFileSource,
    "jetscape": jetscape.FileSource,
    "jewel_from_laura": jewel_from_laura.FileSource,
    "track_skim": track_skim.FileSource,
}

def file_source(
    file_source_config: dict[str, Any]
) -> sources.SourceFromFilename:
    """Factory to create a file source from a configuration.

    Args:
        file_source_config: Configuration for the file source.

    Returns:
        File source with the filename as the remaining argument.
    """
    skim_type = file_source_config["skim_type"]
    try:
        FileSource = file_source_registry[skim_type]
    except KeyError as e:
        msg = f"Unknown skim type: {skim_type}"
        raise ValueError(msg) from e

    # We need to pass only valid args to the file source.
    # The rest of the info will be stored in the metadata.
    # This is a bit of a hack, but I don't want to make the interface too strict, so good enough.
    metadata = copy.deepcopy(file_source_config)
    kwargs = {}
    relevant_kwargs_for_file_source = list(inspect.signature(FileSource).parameters.keys())
    for k in relevant_kwargs_for_file_source:
        v = metadata.pop(k, None)
        if v is not None:
            kwargs[k] = v
    kwargs["metadata"] = metadata

    return partial(FileSource, **kwargs)
