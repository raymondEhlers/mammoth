from typing import Any, List, Optional, Tuple, TypeVar, Union

import awkward as ak
import numpy as np

_T = TypeVar("_T")

ArrayLike = Union[np.ndarray, List[_T]]

class ConstituentSubtractionSettings:
    alpha: float
    r_max: float
    def __init__(self, r_max: float = ..., alpha: float = ...) -> None: ...  # noqa: E704

class OutputWrapper:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
    @property
    def constituent_indices(self) -> ArrayLike[ArrayLike[int]]: ...  # noqa: E704
    @property
    def jets(self) -> Tuple[ArrayLike[float], ArrayLike[float], ArrayLike[float], ArrayLike[float]]: ...  # noqa: E704
    @property
    def subtracted_info(self) -> Optional[Tuple[Tuple[ArrayLike[float], ArrayLike[float], ArrayLike[float], ArrayLike[float]], ArrayLike[int]]]: ...  # noqa: E704

def find_jets(px: ArrayLike[float], py: ArrayLike[float], pz: ArrayLike[float], E: ArrayLike[float], jet_R: float, jet_algorithm: str = ..., eta_range: Tuple[float, float] = ..., min_jet_pt: float = ..., background_subtraction: bool = ..., constituent_subtraction: Optional[ConstituentSubtractionSettings] = ...) -> ak.Array: ...  # noqa: E704
