from typing import ClassVar, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt

DEFAULT_RAPIDITY_MAX: float

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
        random_seed: List[int] = ...,
    ) -> None: ...

class JetFindingSettings:
    R: float
    def __init__(
        self,
        R: float,
        algorithm: str,
        pt_range: Tuple[float, float],
        eta_range: Tuple[float, float],
        recombination_scheme: str = ...,
        strategy: str = ...,
        area_settings: Optional[AreaSettings] = ...,
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
        estimator: Optional[Union[JetMedianBackgroundEstimator, GridMedianBackgroundEstimator]] = ...,
        subtractor: Optional[Union[RhoSubtractor, ConstituentSubtractor]] = ...,
    ) -> None: ...

class ColumnarSplittings:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore
    @property
    def delta_R(self) -> npt.NDArray[np.float32]: ...
    @property
    def kt(self) -> npt.NDArray[np.float32]: ...
    @property
    def parent_index(self) -> npt.NDArray[np.float32]: ...
    @property
    def z(self) -> npt.NDArray[np.float32]: ...

class ColumnarSubjest:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore
    @property
    def constituent_indices(self) -> List[List[int]]: ...
    @property
    def part_of_iterative_splitting(self) -> npt.NDArray[np.bool_]: ...
    @property
    def splitting_node_index(self) -> npt.NDArray[np.int64]: ...

class ConstituentSubtractionSettings:
    alpha: float
    r_max: float
    def __init__(self, r_max: float = ..., alpha: float = ...) -> None: ...

class JetSubstructureSplittings:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore
    def splittings(self) -> ColumnarSplittings: ...
    def subjets(self) -> ColumnarSubjest: ...

class OutputWrapperDouble:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore
    @property
    def constituent_indices(self) -> List[List[int]]: ...
    @property
    def jets(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float64]: ...
    @property
    def subtracted_info(
        self,
    ) -> Optional[
        Tuple[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.int64],
        ]
    ]: ...

class OutputWrapperFloat:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore
    @property
    def constituent_indices(self) -> List[List[int]]: ...
    @property
    def jets(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...
    @property
    def jets_area(self) -> npt.NDArray[np.float32]: ...
    @property
    def subtracted_info(
        self,
    ) -> Optional[
        Tuple[
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
            npt.NDArray[np.int64],
        ]
    ]: ...

@overload
def find_jets_new(   # type: ignore
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
) -> OutputWrapperFloat: ...
@overload
def find_jets_new(
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
) -> OutputWrapperDouble: ...
@overload
def find_jets(px: npt.NDArray[np.float32], py: npt.NDArray[np.float32], pz: npt.NDArray[np.float32], E: npt.NDArray[np.float32], background_px: npt.NDArray[np.float32], background_py: npt.NDArray[np.float32], background_pz: npt.NDArray[np.float32], background_E: npt.NDArray[np.float32], jet_R: float, jet_algorithm: str, area_settings: AreaSettings, eta_range: Tuple[float, float] = ..., fiducial_acceptance: bool = ..., min_jet_pt: float = ..., background_subtraction: bool = ..., constituent_subtraction: Optional[ConstituentSubtractionSettings] = ...) -> OutputWrapperFloat: ...  # type: ignore
@overload
def find_jets(
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    background_px: npt.NDArray[np.float64],
    background_py: npt.NDArray[np.float64],
    background_pz: npt.NDArray[np.float64],
    background_E: npt.NDArray[np.float64],
    jet_R: float,
    jet_algorithm: str,
    area_settings: AreaSettings,
    eta_range: Tuple[float, float] = ...,
    fiducial_acceptance: bool = ...,
    min_jet_pt: float = ...,
    background_subtraction: bool = ...,
    constituent_subtraction: Optional[ConstituentSubtractionSettings] = ...,
) -> OutputWrapperDouble: ...
@overload
def recluster_jet_new(
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    pz: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    jet_finding_settings: JetFindingSettings,
    store_recursive_splittings: bool = ...,
) -> JetSubstructureSplittings: ...
@overload
def recluster_jet_new(
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    jet_finding_settings: JetFindingSettings,
    store_recursive_splittings: bool = ...,
) -> JetSubstructureSplittings: ...
@overload
def recluster_jet(
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
    pz: npt.NDArray[np.float32],
    E: npt.NDArray[np.float32],
    jet_R: float = ...,
    jet_algorithm: str = ...,
    area_settings: Optional[AreaSettings] = ...,
    eta_range: Tuple[float, float] = ...,
    store_recursive_splittings: bool = ...,
) -> JetSubstructureSplittings: ...
@overload
def recluster_jet(
    px: npt.NDArray[np.float64],
    py: npt.NDArray[np.float64],
    pz: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    jet_R: float = ...,
    jet_algorithm: str = ...,
    area_settings: Optional[AreaSettings] = ...,
    eta_range: Tuple[float, float] = ...,
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
