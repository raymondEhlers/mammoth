""" Typing helpers for the full package

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""

from __future__ import annotations

import typing
from typing import Collection, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

# Typing helpers
# Generic for passing types through
_T = TypeVar("_T", covariant=True)
# Generic numpy scalar
# See: https://stackoverflow.com/a/71126857/12907985
Scalar = TypeVar("Scalar", bound=np.generic, covariant=True)
# Generc number
# See: https://stackoverflow.com/a/60617044/12907985
Number = Union[float, int, np.number]

# Using `class AwkwardArray(Protocol[_T]):` caused mypy to hang as of August 2022, but
# for some reason, Collection is fine. Presumably there's a bug somewhere, but not worth worrying about,
# especially given that Collection sems to work okay too
# NOTE: Protocol[_T] is equivalent to `class AwkwardArray(Protocol, Generic[_T])``
# NOTE: Perhaps it's because Collections may be treated in a special way. See the note under
#       the example here: https://mypy.readthedocs.io/en/stable/generics.html#defining-sub-classes-of-generic-classes .
#       The note is copied below:
#       > You have to add an explicit Mapping base class if you want mypy to consider a user-defined
#       > class as a mapping (and Sequence for sequences, etc.). This is because mypy doesn’t use
#       > structural subtyping for these ABCs, unlike simpler protocols like Iterable, which use
#       > structural subtyping.
# NOTE: Some further useful info is:
#       - https://stackoverflow.com/a/48314895/12907985
#       - https://mypy.readthedocs.io/en/stable/generics.html#generic-protocols
class AwkwardArray(Collection[_T]):
    @typing.overload
    def __getitem__(self, key: AwkwardArray[bool]) -> AwkwardArray[_T]: ...

    @typing.overload
    def __getitem__(self, key: AwkwardArray[int]) -> AwkwardArray[_T]: ...

    @typing.overload
    def __getitem__(self, key: Tuple[slice, slice]) -> AwkwardArray[_T]: ...

    @typing.overload
    def __getitem__(self, key: npt.NDArray[Scalar]) -> AwkwardArray[_T]: ...

    @typing.overload
    def __getitem__(self, key: bool) -> _T: ...

    @typing.overload
    def __getitem__(self, key: int) -> _T: ...

    def __getitem__(self, key): ...  # type: ignore

    def __add__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __radd__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __sub__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __rsub__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __mul__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __rmul__(self, other: Union[AwkwardArray[_T], int, float]) -> AwkwardArray[_T]: ...

    def __truediv__(self, other: Union[AwkwardArray[_T], float]) -> AwkwardArray[_T]: ...

    def __lt__(self, other: Union[AwkwardArray[_T], float]) -> AwkwardArray[bool]: ...

    def __le__(self, other: Union[AwkwardArray[_T], float]) -> AwkwardArray[bool]: ...

    def __gt__(self, other: Union[AwkwardArray[_T], float]) -> AwkwardArray[bool]: ...

    def __ge__(self, other: Union[AwkwardArray[_T], float]) -> AwkwardArray[bool]: ...

# Sometimes, it could be either
ArrayOrScalar = Union[AwkwardArray[_T], _T]
