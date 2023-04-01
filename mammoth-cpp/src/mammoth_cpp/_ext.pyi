from __future__ import annotations

from typing import ClassVar, Generator, List, Optional, Protocol, Tuple, Union, overload

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
    def __init__(  # noqa: E301,E704
        self,
        area_type: str = ...,
        ghost_area: float = ...,
        rapidity_max: float = ...,
        repeat_N_ghosts: int = ...,
        grid_scatter: float = ...,
        kt_scatter: float = ...,
        kt_mean: float = ...,
        random_seed: List[int] = ...,
    ) -> None: ...

class NegativeEnergyRecombiner:
    def __init__(self, identifier_index: int = ...) -> None: ...  # noqa: E301,E704
    @property
    def identifier_index(self) -> int: ...  # noqa: E301,E704

class JetFindingSettings:
    R: float
    recombiner: NegativeEnergyRecombiner | None
    def __init__(  # noqa: E301,E704
        self,
        R: float,
        algorithm: str,
        pt_range: Tuple[float, float],
        eta_range: Tuple[float, float],
        recombination_scheme: str = ...,
        strategy: str = ...,
        area_settings: Optional[AreaSettings] = ...,
        recombiner: Optional[NegativeEnergyRecombiner] = ...,
    ) -> None: ...

class JetMedianBackgroundEstimator:
    compute_rho_m: bool
    constituent_pt_max: float
    exclude_n_hardest_jets: int
    use_area_four_vector: bool
    def __init__(  # noqa: E301,E704
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
    def __init__(self, rapidity_max: float = ..., grid_spacing: float = ...) -> None: ...  # noqa: E301,E704

class BackgroundSubtractionType:
    disabled: ClassVar[BackgroundSubtractionType] = ...
    event_wise_constituent_subtraction: ClassVar[BackgroundSubtractionType] = ...
    jet_wise_constituent_subtraction: ClassVar[BackgroundSubtractionType] = ...
    rho: ClassVar[BackgroundSubtractionType] = ...
    def __init__(self, value: int) -> None: ...  # noqa: E301,E704
    def __eq__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __ge__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __getstate__(self) -> int: ...  # noqa: E301,E704
    def __gt__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __hash__(self) -> int: ...  # noqa: E301,E704
    def __index__(self) -> int: ...  # noqa: E301,E704
    def __int__(self) -> int: ...  # noqa: E301,E704
    def __le__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __lt__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __ne__(self, other: object) -> bool: ...  # noqa: E301,E704
    def __setstate__(self, state: int) -> None: ...  # noqa: E301,E704
    @property
    def name(self) -> str: ...  # noqa: E301,E704
    @property
    def value(self) -> int: ...  # noqa: E301,E704

class RhoSubtractor:
    use_rho_M: bool
    use_safe_mass: bool
    def __init__(self, use_rho_M: bool = ..., use_safe_mass: bool = ...) -> None: ...  # noqa: E301,E704

class ConstituentSubtractor:
    alpha: float
    distance_measure: str
    r_max: float
    rapidity_max: float
    def __init__(  # noqa: E301,E704
        self, r_max: float = ..., alpha: float = ..., rapidity_max: float = ..., distance_measure: str = ...
    ) -> None: ...

class BackgroundSubtraction:
    def __init__(  # noqa: E301,E704
        self,
        type: BackgroundSubtractionType,
        estimator: Optional[Union[JetMedianBackgroundEstimator, GridMedianBackgroundEstimator]] = ...,
        subtractor: Optional[Union[RhoSubtractor, ConstituentSubtractor]] = ...,
    ) -> None: ...

class ColumnarSplittings:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]  # noqa: E704
    @property
    def delta_R(self) -> npt.NDArray[np.float32]: ...  # noqa: E301,E704
    @property
    def kt(self) -> npt.NDArray[np.float32]: ...  # noqa: E301,E704
    @property
    def parent_index(self) -> npt.NDArray[np.float32]: ...  # noqa: E301,E704
    @property
    def z(self) -> npt.NDArray[np.float32]: ...  # noqa: E301,E704

class ColumnarSubjets:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]  # noqa: E704
    @property
    def constituent_indices(self) -> List[List[int]]: ...  # noqa: E301,E704
    @property
    def part_of_iterative_splitting(self) -> npt.NDArray[np.bool_]: ...  # noqa: E301,E704
    @property
    def splitting_node_index(self) -> npt.NDArray[np.int64]: ...  # noqa: E301,E704

class JetSubstructureSplittings:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]  # noqa: E704
    def splittings(self) -> ColumnarSplittings: ...  # noqa: E301,E704
    def subjets(self) -> ColumnarSubjets: ...  # noqa: E301,E704

class OutputWrapperDouble:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]  # noqa: E704
    @property
    def constituent_indices(self) -> List[List[int]]: ...  # noqa: E301,E704
    @property  # noqa: E301
    def jets(  # noqa: E301,E704
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float64]: ...  # noqa: E301,E704
    @property
    def rho_value(self) -> np.float64: ...  # noqa: E301,E704
    @property  # noqa: E301
    def subtracted_info(  # noqa: E301,E704
        self,
    ) -> Optional[
        Tuple[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.int64],
        ]
    ]: ...

class OutputWrapperFloat:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]  # noqa: E704
    @property
    def constituent_indices(self) -> List[List[int]]: ...  # noqa: E704
    @property  # noqa: E301
    def jets(  # noqa: E704
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float32]: ...  # noqa: E704
    @property
    def rho_value(self) -> np.float32: ...  # noqa: E704
    @property  # noqa: E301
    def subtracted_info(  # noqa: E704
        self,
    ) -> Optional[
        Tuple[
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
            npt.NDArray[np.int64],
        ]
    ]: ...

@overload
def find_jets(   # type: ignore[misc]  # noqa: E704
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
) -> OutputWrapperFloat: ...
@overload
def find_jets(  # noqa: E704
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
) -> OutputWrapperDouble: ...
@overload
def recluster_jet(  # noqa: E704
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    pz: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    jet_finding_settings: JetFindingSettings,
    store_recursive_splittings: bool = ...,
) -> JetSubstructureSplittings: ...
@overload
def recluster_jet(  # noqa: E704
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    jet_finding_settings: JetFindingSettings,
    store_recursive_splittings: bool = ...,
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
    def __init__(self, value: int) -> None: ...  # noqa: E704
    def __eq__(self, other: object) -> bool: ...  # noqa: E704
    def __ge__(self, other: object) -> bool: ...  # noqa: E704
    def __getstate__(self) -> int: ...  # noqa: E704
    def __gt__(self, other: object) -> bool: ...  # noqa: E704
    def __hash__(self) -> int: ...  # noqa: E704
    def __index__(self) -> int: ...  # noqa: E704
    def __int__(self) -> int: ...  # noqa: E704
    def __le__(self, other: object) -> bool: ...  # noqa: E704
    def __lt__(self, other: object) -> bool: ...  # noqa: E704
    def __ne__(self, other: object) -> bool: ...  # noqa: E704
    def __setstate__(self, state: int) -> None: ...  # noqa: E704
    @property
    def name(self) -> str: ...  # noqa: E704
    @property
    def value(self) -> int: ...  # noqa: E704

class TrackingEfficiencyPeriod:
    LHC11a: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC11h: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC15o: ClassVar[TrackingEfficiencyPeriod] = ...
    LHC18qr: ClassVar[TrackingEfficiencyPeriod] = ...
    disabled: ClassVar[TrackingEfficiencyPeriod] = ...
    pA: ClassVar[TrackingEfficiencyPeriod] = ...
    pp: ClassVar[TrackingEfficiencyPeriod] = ...
    def __init__(self, value: int) -> None: ...  # noqa: E704
    def __eq__(self, other: object) -> bool: ...  # noqa: E704
    def __ge__(self, other: object) -> bool: ...  # noqa: E704
    def __getstate__(self) -> int: ...  # noqa: E704
    def __gt__(self, other: object) -> bool: ...  # noqa: E704
    def __hash__(self) -> int: ...  # noqa: E704
    def __index__(self) -> int: ...  # noqa: E704
    def __int__(self) -> int: ...  # noqa: E704
    def __le__(self, other: object) -> bool: ...  # noqa: E704
    def __lt__(self, other: object) -> bool: ...  # noqa: E704
    def __ne__(self, other: object) -> bool: ...  # noqa: E704
    def __setstate__(self, state: int) -> None: ...  # noqa: E704
    @property
    def name(self) -> str: ...  # noqa: E704
    @property
    def value(self) -> int: ...  # noqa: E704

def fast_sim_tracking_efficiency(  # noqa: E704
    track_pt: npt.NDArray[np.float64],
    track_eta: npt.NDArray[np.float64],
    event_activity: TrackingEfficiencyEventActivity,
    period: TrackingEfficiencyPeriod,
) -> object: ...
def find_event_activity(value: float) -> TrackingEfficiencyEventActivity: ...  # noqa: E704
