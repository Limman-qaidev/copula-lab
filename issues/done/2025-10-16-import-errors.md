# Issue: Streamlit calibration import errors

## Summary
- Opening the Calibrate or Study pages raises ImportError exceptions for
  `gaussian_ifm_corr` and `rho_matrix_from_tau_gaussian`.
- Root cause: stale imports in the Streamlit pages point to legacy symbol
  names that are no longer exposed by `src.estimators` after the recent
  refactor. The Student t utilities also need to avoid importing matrix
  helpers at module import time to prevent circular dependencies.
- Scope: Streamlit calibration workflow and estimator helper exports.

## Tasks
- [x] Update Streamlit pages to consume the public `src.estimators` API and
      add compatibility fallbacks so legacy attribute names keep working.
- [x] Export the tau inversion matrix helper through the estimator package
      to avoid ImportError on cold imports.
- [x] Ensure IFM estimators expose correlation-matrix helpers expected by the
      UI (add alias or wrapper if necessary).
- [x] Add regression tests for the alias exports to avoid future breakage.

## Acceptance criteria
- Running `streamlit run app/streamlit_app.py` no longer raises import
  errors on the Calibrate or Study pages.
- New tests document the estimator exports used by the UI.
- Static analysis (isort, black, flake8, mypy --strict, pytest) continues to
  pass without modifications to global configuration.
