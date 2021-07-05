from typing import Any, List, Optional, Tuple, TypeVar, Union

import awkward as ak
import numpy as np

_T = TypeVar("_T")

ArrayLike = Union[np.ndarray, List[_T]]

class ConstituentSubtractionSettings:
    alpha: float
    r_max: float
    def __init__(self, r_max: float = ..., alpha: float = ...) -> None: ...

class OutputWrapper:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def constituent_indices(self) -> ArrayLike[ArrayLike[int]]: ...
    @property
    def jets(self) -> Tuple[ArrayLike[float],ArrayLike[float],ArrayLike[float],ArrayLike[float]]: ...
    @property
    def subtracted_info(self) -> Optional[Tuple[Tuple[ArrayLike[float],ArrayLike[float],ArrayLike[float],ArrayLike[float]],ArrayLike[int]]]: ...

def find_jets(*args, **kwargs) -> ak.Array: ...
