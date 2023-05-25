
import logging
from typing import Any, BinaryIO, Mapping

logger = logging.getLogger(__name__)


def merge_results(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """Merge job results together.

    By convention, we merge into the first dict to try to avoid unnecessary copying.

    Although this should generically work for any object which implements `__add__`,
    it's geared towards histograms.

    Note:
        For the first result, it's often convenient to start with a variable containing an
        empty dict as the argument to a. That way, the merged results will be stored in
        a persistent variable.

    Args:
        a: Job result to be merged into.
        b: Result to be merged with.

    Returns:
        Merged histograms
    """
    # Short circuit if nothing to be done
    if not b and a:
        logger.debug("Returning a since b is None")
        return a
    if not a and b:
        logger.debug("Returning b since a is None")
        return b

    # Ensure we don't miss anything in either dict
    all_keys = set(a) | set(b)

    for k in all_keys:
        a_value = a.get(k)
        b_value = b.get(k)
        # Nothing to be done
        if a_value and b_value is None:
            logger.debug(f"b_value is None for {k}. Skipping")
            continue
        # Just take the b value and move on
        if a_value is None and b_value:
            logger.debug(f"a_value is None for {k}. Assigning")
            a[k] = b_value
            continue
        # At this point, both a_value and b_value should be not None
        assert a_value is not None
        assert b_value is not None

        # Recursive on dict
        if isinstance(a_value, dict):
            logger.debug(f"Recursing on dict for {k}")
            a[k] = merge_results(a_value, b_value)
        else:
            # Otherwise, merge
            logger.debug(f"Merging for {k}")
            a[k] = a_value + b_value

    return a


def write_hists_to_file(hists: Mapping[Any, Any], f: BinaryIO, prefix: str = "") -> bool:
    """Recursively write histograms to a given file.

    Args:
        hists: Hists to be written.
        f: File to be written to. Usually a file opened with uproot.
        prefix: Prefix to append to all keys. Default: "". This is rarely set by the user
            directly. Instead, it's used when recursing to keep track of the path.

    Returns:
        True if writing was successful.
    """
    for k, v in hists.items():
        if isinstance(v, dict):
            write_hists_to_file(hists=v, f=f, prefix=f"{prefix}_{k}")
        else:
            write_name = str(k) if not prefix else f"{prefix}_{k}"
            f[write_name] = v  # type: ignore[index]

    return True