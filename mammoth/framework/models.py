import functools
from typing import Any, Callable, Union

import numpy as np
from scipy.interpolate import interp1d


def inverse_sample_decorator(
    distribution: Callable[..., Union[np.ndarray, float]]
) -> Callable[..., Union[float, np.ndarray]]:
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
    ) -> Union[np.ndarray, float]:
        # Validation
        x = np.linspace(x_min, x_max, int(n_distribution_samples))
        cumulative = np.cumsum(distribution(x, **kwargs))
        cumulative -= cumulative.min()
        # This is an inverse of the CDF
        # See: https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
        f = interp1d(cumulative / cumulative.max(), x)
        rng = np.random.default_rng()
        return f(rng.uniform(size=n_samples))  # type: ignore

    return wrapper


def x_exp(x: Union[float, np.ndarray], scale: float) -> np.ndarray:
    return x * np.exp(-x / scale)  # type: ignore


sample_x_exp = inverse_sample_decorator(x_exp)
