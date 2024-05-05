from __future__ import annotations

from collections.abc import Generator
from typing import ClassVar, Protocol, overload

import numpy as np
import numpy.typing as npt

DEFAULT_RAPIDITY_MAX: float

class cpp_redirect_stream(Protocol):
    """Redirect cpp stdout/stderr with context manager"""
    def __call__(self, /, stdout: str = ..., stderr: str = ...) -> Generator[None, None, None]: ...

class AreaSettings:
    area_type: str
    ghost_area: float
    rapidity_max: float
    def __init__(
        self,
        area_type: str = ...,
        ghost_area: float = ...,
        rapidity_max: float = ...,
        repeat_N_ghosts: int = ...,
        grid_scatter: float = ...,
        kt_scatter: float = ...,
        kt_mean: float = ...,
        random_seed: list[int] = ...,
    ) -> None: ...

class NegativeEnergyRecombiner:
    def __init__(self, identifier_index: int = ...) -> None: ...
    @property
    def identifier_index(self) -> int: ...

class JetFindingSettings:
    R: float
    area_settings: AreaSettings | None
    recombiner: NegativeEnergyRecombiner | None
    def __init__(
        self,
        R: float,
        algorithm: str,
        pt_range: tuple[float, float],
        eta_range: tuple[float, float],
        recombination_scheme: str = ...,
        strategy: str = ...,
        area_settings: AreaSettings | None = ...,
        recombiner: NegativeEnergyRecombiner | None = ...,
        additional_algorithm_parameter: str | None = ...,
    ) -> None: ...

class JetMedianBackgroundEstimator:
    compute_rho_m: bool
    constituent_pt_max: float
    exclude_n_hardest_jets: int
    use_area_four_vector: bool
    def __init__(
        self,
        jet_finding_settings: JetFindingSettings,
        compute_rho_m: bool = ...,
        use_area_four_vector: bool = ...,
        exclude_n_hardest_jets: int = ...,
        constituent_pt_max: float = ...,
    ) -> None: ...

class GridMedianBackgroundEstimator:
    grid_spacing: float
    rapidity_max: float
    def __init__(self, rapidity_max: float = ..., grid_spacing: float = ...) -> None: ...

class BackgroundSubtractionType:
    disabled: ClassVar[BackgroundSubtractionType] = ...
    event_wise_constituent_subtraction: ClassVar[BackgroundSubtractionType] = ...
    jet_wise_constituent_subtraction: ClassVar[BackgroundSubtractionType] = ...
    rho: ClassVar[BackgroundSubtractionType] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class RhoSubtractor:
    use_rho_M: bool
    use_safe_mass: bool
    def __init__(self, use_rho_M: bool = ..., use_safe_mass: bool = ...) -> None: ...

class ConstituentSubtractor:
    alpha: float
    distance_measure: str
    r_max: float
    rapidity_max: float
    def __init__(
        self, r_max: float = ..., alpha: float = ..., rapidity_max: float = ..., distance_measure: str = ...
    ) -> None: ...

class BackgroundSubtraction:
    def __init__(
        self,
        type: BackgroundSubtractionType,
        estimator: JetMedianBackgroundEstimator | GridMedianBackgroundEstimator | None = ...,
        subtractor: RhoSubtractor | ConstituentSubtractor | None = ...,
    ) -> None: ...

class ColumnarSplittings:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def delta_R(self) -> npt.NDArray[np.float32]: ...
    @property
    def kt(self) -> npt.NDArray[np.float32]: ...
    @property
    def parent_index(self) -> npt.NDArray[np.short]: ...
    @property
    def tau(self) -> npt.NDArray[np.float32]: ...
    @property
    def z(self) -> npt.NDArray[np.float32]: ...

class ColumnarSubjets:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def constituent_indices(self) -> list[list[int]]: ...
    @property
    def part_of_iterative_splitting(self) -> npt.NDArray[np.bool_]: ...
    @property
    def splitting_node_index(self) -> npt.NDArray[np.int64]: ...

class JetSubstructureSplittings:
    def __init__(self, *args, **kwargs) -> None: ...
    def splittings(self) -> ColumnarSplittings: ...
    def subjets(self) -> ColumnarSubjets: ...

class OutputWrapperDouble:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def constituents_user_index(self) -> list[list[int]]: ...
    @property
    def jets(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float64]: ...
    @property
    def rho_value(self) -> np.float64: ...
    @property
    def subtracted_info(
        self,
    ) -> (
        tuple[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.int64],
        ]
        | None
    ): ...

class OutputWrapperFloat:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def constituents_user_index(self) -> list[list[int]]: ...
    @property
    def jets(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float32]: ...
    @property
    def rho_value(self) -> np.float32: ...
    @property
    def subtracted_info(
        self,
    ) -> (
        tuple[
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
            npt.NDArray[np.int64],
        ]
        | None
    ): ...

@overload
def find_jets(  # type: ignore[overload-overlap]
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    pz: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    jet_finding_settings: JetFindingSettings,
    background_px: npt.NDArray[np.float32],
    background_py: npt.NDArray[np.float32],
    background_pz: npt.NDArray[np.float32],
    background_E: npt.NDArray[np.float32],
    background_subtraction: BackgroundSubtraction,
    user_index: npt.NDArray[np.int64] | None,
    release_gil: bool,
) -> OutputWrapperFloat: ...
@overload
def find_jets(
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    jet_finding_settings: JetFindingSettings,
    background_px: npt.NDArray[np.float64],
    background_py: npt.NDArray[np.float64],
    background_pz: npt.NDArray[np.float64],
    background_E: npt.NDArray[np.float64],
    background_subtraction: BackgroundSubtraction,
    user_index: npt.NDArray[np.int64] | None,
    release_gil: bool = ...,
) -> OutputWrapperDouble: ...
@overload
def recluster_jet(
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    pz: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    jet_finding_settings: JetFindingSettings,
    user_index: npt.NDArray[np.int64] | None,
    store_recursive_splittings: bool = ...,
    release_gil: bool = ...,
) -> JetSubstructureSplittings: ...
@overload
def recluster_jet(
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    jet_finding_settings: JetFindingSettings,
    user_index: npt.NDArray[np.int64] | None,
    store_recursive_splittings: bool = ...,
    release_gil: bool = ...,
) -> JetSubstructureSplittings: ...

# ALICE
# Fast sim
class TrackingEfficiencyEventActivity:
    central_00_10: ClassVar[TrackingEfficiencyEventActivity] = ...
    inclusive: ClassVar[TrackingEfficiencyEventActivity] = ...
    invalid: ClassVar[TrackingEfficiencyEventActivity] = ...
    mid_central_10_30: ClassVar[TrackingEfficiencyEventActivity] = ...
    peripheral_50_90: ClassVar[TrackingEfficiencyEventActivity] = ...
    semi_central_30_50: ClassVar[TrackingEfficiencyEventActivity] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class TrackingEfficiencyPeriod:
    LHC11a: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC11h: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC15o: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC18qr: ClassVar[TrackingEfficiencyPeriod] = ...
    disabled: ClassVar[TrackingEfficiencyPeriod] = ...
    pA: ClassVar[TrackingEfficiencyPeriod] = ...
    pp: ClassVar[TrackingEfficiencyPeriod] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def fast_sim_tracking_efficiency(
    track_pt: npt.NDArray[np.float64],
    track_eta: npt.NDArray[np.float64],
    event_activity: TrackingEfficiencyEventActivity,
    period: TrackingEfficiencyPeriod,
) -> object: ...
def find_event_activity(value: float) -> TrackingEfficiencyEventActivity: ...
def smooth_array(arr: npt.NDArray[np.float64], n_times: int = ...) -> npt.NDArray[np.float64]: ...
def smooth_array_f(arr: npt.NDArray[np.float32], n_times: int = ...) -> npt.NDArray[np.float32]: ...
