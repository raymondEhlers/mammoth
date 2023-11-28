from __future__ import annotations

import functools
from typing import Any, Callable, Union

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from mammoth_cpp._ext import TrackingEfficiencyEventActivity as ALICETrackingEfficiencyEventActivity
from mammoth_cpp._ext import TrackingEfficiencyPeriod as ALICETrackingEfficiencyPeriod
from mammoth_cpp._ext import fast_sim_tracking_efficiency as alice_fast_sim_tracking_efficiency

# Need this to tell mypy that this symbol is explicitly exported. If the name didn't change,
# it would work, but I think the ALICE label needs to be specified for clarity.
__all__ = ["alice_fast_sim_tracking_efficiency"]


def inverse_sample_decorator(
    distribution: Callable[..., Union[npt.NDArray[Union[np.float32, np.float64]], float]]
) -> Callable[..., Union[float, npt.NDArray[Union[np.float32, np.float64]]]]:
    """Decorator to perform inverse transform sampling.

    Based on: https://stackoverflow.com/a/64288861/12907985
    """

    @functools.wraps(distribution)
    def wrapper(
        n_samples: int,
        x_min: float,
        x_max: float,
        n_distribution_samples: int = 100_000,
        **kwargs: Any,
    ) -> Union[npt.NDArray[Union[np.float32, np.float64]], float]:
        # Validation
        x = np.linspace(x_min, x_max, int(n_distribution_samples))
        cumulative = np.cumsum(distribution(x, **kwargs))
        cumulative -= cumulative.min()
        # This is an inverse of the CDF
        # See: https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
        f = interp1d(cumulative / cumulative.max(), x)
        rng = np.random.default_rng()
        return f(rng.uniform(size=n_samples))  # type: ignore[no-any-return]

    return wrapper


def x_exp(
    x: Union[float, npt.NDArray[Union[np.float32, np.float64]]], scale: float
) -> npt.NDArray[Union[np.float32, np.float64]]:
    return x * np.exp(-x / scale)


sample_x_exp = inverse_sample_decorator(x_exp)


@attrs.define
class ALICEFastSimParameters:
    period: ALICETrackingEfficiencyPeriod
    event_activity: ALICETrackingEfficiencyEventActivity
