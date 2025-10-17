## Summary
- Calibrate page still raises import errors for Gaussian IFM correlation helpers and tau matrix builders on some environments.
- Data, Compare, and Study tabs need UX polish (responsive charts/dataframes) and better session persistence so uploaded datasets remain available across navigation.
- Multi-dimensional workflows must keep the pseudo-observations and dataset metadata synchronized for diagnostics and comparisons.

## Tasks
- [x] Restore compatibility exports for Gaussian IFM and tau inversion helpers consumed by the Streamlit app and legacy namespaces.
- [x] Harden Streamlit rendering helpers to fall back gracefully when the new `width` keyword is unavailable.
- [x] Persist uploaded datasets in session state so calibrate/compare/study tabs reuse the same data without requiring a reload.
- [x] Refresh tab layouts to ensure tables/charts remain legible across themes and provide clearer navigation guidance.

## Acceptance criteria
- Calibrate, Compare, and Study pages load without import errors after a clean restart.
- Uploading a dataset once makes it available across tabs without re-uploading.
- Charts/tables render without warnings on both legacy and current Streamlit releases.
- Static analysis (isort, black, flake8, mypy --strict, pytest) continues to pass.
