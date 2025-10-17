from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency in some deployments
    from src.estimators.ifm import (  # type: ignore[import-untyped]
        gaussian_ifm_corr as _gaussian_ifm_corr,
    )
except ImportError:  # pragma: no cover - fallback exercised in app runtime
    _gaussian_ifm_corr = None

from src.estimators.student_t import (  # noqa: E402
    student_t_ifm,
    student_t_pmle,
)
from src.estimators.tau_inversion import (  # noqa: E402
    choose_nu_from_tail,
    rho_matrix_from_tau_gaussian,
    rho_matrix_from_tau_student_t,
    theta_from_tau_amh,
    theta_from_tau_clayton,
    theta_from_tau_frank,
    theta_from_tau_gumbel,
    theta_from_tau_joe,
)
from src.models.copulas.archimedean import (  # noqa: E402
    AMHCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from src.utils.dependence import (  # noqa: E402
    average_kendall_tau,
    average_tail_dep_upper,
    kendall_tau_matrix,
)
from src.utils.modelsel import (  # noqa: E402
    gaussian_pseudo_loglik,
    information_criteria,
    student_t_pseudo_loglik,
)
from src.utils.results import FitResult  # noqa: E402
from src.utils.types import FloatArray  # noqa: E402

try:  # pragma: no cover - scipy optional in some deployments
    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
    from scipy.stats import norm  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - handled by fallback implementation
    minimize_scalar = None  # type: ignore[assignment]
    norm = None

GaussianMatrixFunc = Callable[[FloatArray], FloatArray]


def _gaussian_ifm_corr_fallback(U: FloatArray) -> FloatArray:
    if norm is None:
        raise ImportError("scipy.stats.norm is required for Gaussian IFM")

    u_array = np.asarray(U, dtype=np.float64)
    if u_array.ndim != 2:
        raise ValueError("U must be a two-dimensional array.")
    n_obs, dim = u_array.shape
    if n_obs < 2 or dim < 2:
        raise ValueError(
            "At least two observations and two dimensions are required."
        )
    if np.any((u_array <= 0.0) | (u_array >= 1.0)):
        raise ValueError("U entries must lie strictly between 0 and 1.")

    clipped = np.clip(u_array, 1e-12, 1.0 - 1e-12)
    z = norm.ppf(clipped)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    return corr


ifm_callable: GaussianMatrixFunc | None = _gaussian_ifm_corr


def _estimate_student_nu(
    U: FloatArray, corr: NDArray[np.float64]
) -> float:
    if minimize_scalar is None:
        lambda_upper = average_tail_dep_upper(U)
        return choose_nu_from_tail(lambda_upper)

    def objective(nu_val: float) -> float:
        if nu_val <= 2.05:
            return float("inf")
        return -student_t_pseudo_loglik(U, corr, nu_val)

    result = minimize_scalar(
        objective, bounds=(2.05, 60.0), method="bounded"
    )
    if not result.success:
        lambda_upper = average_tail_dep_upper(U)
        return choose_nu_from_tail(lambda_upper)
    return float(result.x)


def gaussian_ifm_corr(U: FloatArray) -> FloatArray:
    """Return the IFM correlation matrix with a graceful fallback."""

    if ifm_callable is not None:
        corr = ifm_callable(U)
        return np.asarray(corr, dtype=np.float64)
    return _gaussian_ifm_corr_fallback(U)


def _flatten_corr(
    corr: FloatArray,
    labels: Tuple[str, ...] | None,
) -> Tuple[Dict[str, float], Tuple[str, ...]]:
    matrix = np.asarray(corr, dtype=np.float64)
    dim = matrix.shape[0]
    params: Dict[str, float] = {}
    display: list[str] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            key = f"rho_{i + 1}_{j + 1}"
            params[key] = float(matrix[i, j])
            if labels is None:
                lhs = f"rho({i + 1},{j + 1})"
            else:
                lhs = f"rho({labels[i]}, {labels[j]})"
            display.append(f"{lhs}={matrix[i, j]:.4f}")
    return params, tuple(display)


def _build_corr(
    params: Mapping[str, float], dim: int
) -> NDArray[np.float64] | None:
    matrix = np.eye(dim, dtype=np.float64)
    found = False
    for key, value in params.items():
        if not key.startswith("rho_"):
            continue
        parts = key.split("_")
        if len(parts) != 3:
            continue
        try:
            i = int(parts[1]) - 1
            j = int(parts[2]) - 1
        except ValueError:
            continue
        if not (0 <= i < dim and 0 <= j < dim):
            continue
        matrix[i, j] = matrix[j, i] = float(value)
        found = True
    return matrix if found else None


@dataclass(frozen=True)
class CalibrationOutcome:
    """Container for a calibrated copula and presentation helpers."""

    result: FitResult
    display: Tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSpec:
    """Describe a copula family, method, and its calibration routine."""

    family: str
    method: str
    calibrate: Callable[
        [FloatArray, Tuple[str, ...] | None],
        CalibrationOutcome,
    ]
    min_dim: int = 2
    max_dim: int | None = None

    def supports_dim(self, dim: int) -> bool:
        if dim < self.min_dim:
            return False
        if self.max_dim is not None and dim > self.max_dim:
            return False
        return True


def _calibrate_gaussian_tau(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    tau_matrix = kendall_tau_matrix(U)
    corr = rho_matrix_from_tau_gaussian(tau_matrix)
    loglik = gaussian_pseudo_loglik(U, corr)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    fit = FitResult(
        family="Gaussian",
        params=params,
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


def _calibrate_gaussian_ifm(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    corr = gaussian_ifm_corr(U)
    loglik = gaussian_pseudo_loglik(U, corr)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    fit = FitResult(
        family="Gaussian",
        params=params,
        method="IFM",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


def _calibrate_student_tau(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    tau_matrix = kendall_tau_matrix(U)
    corr = rho_matrix_from_tau_student_t(tau_matrix)
    nu = _estimate_student_nu(U, corr)
    loglik = student_t_pseudo_loglik(U, corr, nu)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    display = display + (f"nu={nu:.4f}",)
    fit = FitResult(
        family="Student t",
        params=params,
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


def _calibrate_student_ifm(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    corr, nu = student_t_ifm(U)
    loglik = student_t_pseudo_loglik(U, corr, nu)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    display = display + (f"nu={nu:.4f}",)
    fit = FitResult(
        family="Student t",
        params=params,
        method="IFM",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


def _calibrate_student_pmle(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    corr, nu, loglik = student_t_pmle(U)
    k_params = U.shape[1] * (U.shape[1] - 1) // 2 + 1
    aic, bic = information_criteria(loglik, k_params=k_params, n=U.shape[0])
    params, display = _flatten_corr(corr, labels)
    params["nu"] = float(nu)
    display = display + (f"nu={nu:.4f}",)
    fit = FitResult(
        family="Student t",
        params=params,
        method="PMLE",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


ArchimedeanBuilder = Callable[
    [float, int],
    AMHCopula | ClaytonCopula | FrankCopula | GumbelCopula | JoeCopula,
]


def _calibrate_archimedean(
    U: FloatArray,
    family: str,
    builder: ArchimedeanBuilder,
    theta_from_tau: Callable[[float], float],
    labels: Tuple[str, ...] | None,
) -> CalibrationOutcome:
    avg_tau = average_kendall_tau(U)
    theta = theta_from_tau(avg_tau)
    copula = builder(theta, U.shape[1])
    density = copula.pdf(U)
    if np.any(density <= 0.0):
        raise ValueError("Copula density returned non-positive values.")
    loglik = float(np.sum(np.log(density)))
    aic, bic = information_criteria(loglik, k_params=1, n=U.shape[0])
    params = {"theta": float(theta)}
    display = (f"theta={theta:.4f}",)
    fit = FitResult(
        family=family,
        params=params,
        method="Tau inversion",
        loglik=loglik,
        aic=aic,
        bic=bic,
    )
    return CalibrationOutcome(result=fit, display=display)


def _calibrate_clayton(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    del labels

    def builder(theta: float, dim: int) -> ClaytonCopula:
        return ClaytonCopula(theta=theta, dim=dim)

    return _calibrate_archimedean(
        U,
        "Clayton",
        builder,
        theta_from_tau_clayton,
        None,
    )


def _calibrate_gumbel(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    del labels

    def builder(theta: float, dim: int) -> GumbelCopula:
        return GumbelCopula(theta=theta, dim=dim)

    return _calibrate_archimedean(
        U,
        "Gumbel",
        builder,
        theta_from_tau_gumbel,
        None,
    )


def _calibrate_frank(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    del labels

    def builder(theta: float, dim: int) -> FrankCopula:
        return FrankCopula(theta=theta, dim=dim)

    return _calibrate_archimedean(
        U,
        "Frank",
        builder,
        theta_from_tau_frank,
        None,
    )


def _calibrate_joe(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    del labels

    def builder(theta: float, dim: int) -> JoeCopula:
        return JoeCopula(theta=theta, dim=dim)

    return _calibrate_archimedean(
        U,
        "Joe",
        builder,
        theta_from_tau_joe,
        None,
    )


def _calibrate_amh(
    U: FloatArray, labels: Tuple[str, ...] | None
) -> CalibrationOutcome:
    del labels
    if U.shape[1] != 2:
        raise ValueError("The AMH copula currently supports two dimensions.")

    def builder(theta: float, dim: int) -> AMHCopula:
        del dim
        return AMHCopula(theta=theta)

    return _calibrate_archimedean(
        U,
        "AMH",
        builder,
        theta_from_tau_amh,
        None,
    )


_CALIBRATION_SPECS: Tuple[CalibrationSpec, ...] = (
    CalibrationSpec("Gaussian", "Tau inversion", _calibrate_gaussian_tau),
    CalibrationSpec("Gaussian", "IFM", _calibrate_gaussian_ifm),
    CalibrationSpec("Student t", "Tau inversion", _calibrate_student_tau),
    CalibrationSpec("Student t", "IFM", _calibrate_student_ifm),
    CalibrationSpec("Student t", "PMLE", _calibrate_student_pmle),
    CalibrationSpec("Clayton", "Tau inversion", _calibrate_clayton),
    CalibrationSpec("Gumbel", "Tau inversion", _calibrate_gumbel),
    CalibrationSpec("Frank", "Tau inversion", _calibrate_frank),
    CalibrationSpec("Joe", "Tau inversion", _calibrate_joe),
    CalibrationSpec("AMH", "Tau inversion", _calibrate_amh, max_dim=2),
)

_SPEC_INDEX: Dict[Tuple[str, str], CalibrationSpec] = {
    (spec.family, spec.method): spec for spec in _CALIBRATION_SPECS
}

_FAMILY_ORDER: Tuple[str, ...] = (
    "Gaussian",
    "Student t",
    "Clayton",
    "Gumbel",
    "Frank",
    "Joe",
    "AMH",
)


def list_family_names() -> Tuple[str, ...]:
    """Return the default ordered list of copula families."""

    return _FAMILY_ORDER


def get_calibration_specs() -> Tuple[CalibrationSpec, ...]:
    """Return all registered calibration specifications."""

    return _CALIBRATION_SPECS


def get_specs_for_family(
    family: str, dim: int | None = None
) -> Tuple[CalibrationSpec, ...]:
    """Return calibration specs for the requested family and dimension."""

    if family not in _FAMILY_ORDER:
        raise ValueError(f"Unknown copula family: {family}")
    specs = tuple(spec for spec in _CALIBRATION_SPECS if spec.family == family)
    if dim is None:
        return specs
    return tuple(spec for spec in specs if spec.supports_dim(dim))


def get_specs_for_dimension(dim: int) -> Tuple[CalibrationSpec, ...]:
    """Return calibration specs compatible with the provided dimension."""

    return tuple(spec for spec in _CALIBRATION_SPECS if spec.supports_dim(dim))


def run_spec(
    spec: CalibrationSpec,
    U: FloatArray,
    labels: Tuple[str, ...] | None = None,
) -> CalibrationOutcome:
    """Execute the calibration routine for a given specification."""

    return spec.calibrate(U, labels)


def run_calibration(
    family: str,
    method: str,
    U: FloatArray,
    labels: Tuple[str, ...] | None = None,
) -> CalibrationOutcome:
    """Calibrate a copula for the requested family and method."""

    try:
        spec = _SPEC_INDEX[(family, method)]
    except KeyError as exc:  # pragma: no cover - guarded by UI selections
        raise ValueError(
            f"Unknown calibration request: {family} / {method}"
        ) from exc
    return run_spec(spec, U, labels)


def reconstruct_corr(
    params: Mapping[str, float], dim: int
) -> NDArray[np.float64] | None:
    """Reconstruct a correlation matrix from stored parameter entries."""

    return _build_corr(params, dim)


__all__ = [
    "CalibrationOutcome",
    "CalibrationSpec",
    "gaussian_ifm_corr",
    "get_calibration_specs",
    "get_specs_for_dimension",
    "get_specs_for_family",
    "list_family_names",
    "reconstruct_corr",
    "run_calibration",
    "run_spec",
]
