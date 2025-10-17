# Copula Lab v1.0

Copula Lab is an end-to-end environment for constructing, calibrating, and
studying multivariate copulas. The repository combines a typed Python toolkit
with a Streamlit interface so quantitative analysts can explore theory,
validate models, and generate synthetic datasets without leaving the browser.

## Key capabilities

- **Interactive Streamlit laboratory** with dedicated pages for data
  preparation, calibration, model comparison, theoretical study, and an
  advanced sandbox for experimentation.
- **Comprehensive copula support** covering Gaussian, Student t, Clayton,
  Gumbel, Frank, Joe, and Ali–Mikhail–Haq families across arbitrary
  dimensions.
- **Multiple calibration strategies** including Kendall tau inversion,
  inference functions for margins (IFM), pseudo maximum likelihood, and
  log-likelihood optimization with automatic model ranking via AIC/BIC.
- **Diagnostic visualisations** featuring empirical vs. model density overlays,
  Rosenblatt transforms, tail dependence summaries, and convergence traces.
- **Custom dataset generation** through preset copulas or user-defined
  densities with configurable marginal distributions, correlation matrices, and
  export-ready CSV output.
- **Extensive documentation** lives inside `docs/` and mirrors the
  Streamlit workflow with derivations, references, and practice notes.

## Installation

Copula Lab targets Python 3.10+. Use an isolated environment,
then install the locked dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Streamlit app

Launch the laboratory from the project root:

```bash
streamlit run app/streamlit_app.py
```

The application persists uploaded datasets and calibration results in session
state so you can move freely between the **Data**, **Calibrate**, **Compare**,
**Study**, and **Sandbox** modules.

## Testing and quality gates

Every change must pass the strict quality suite enforced by continuous
integration:

```bash
pytest
mypy --strict .
flake8
black --check .
isort --check .
```

The configuration (see `pyproject.toml`) pins the maximum line length to 79
characters and enables full `mypy --strict` semantics.

## Repository layout

- `app/`: Streamlit entry point and page definitions (`1_Data.py` through
  `5_Sandbox.py`).
- `src/`: Reusable Python package with copula models, calibration workflows,
  estimation utilities, and diagnostic helpers.
- `tests/`: End-to-end and unit tests that exercise every calibration routine,
  density visual, sampling helper, and transform.
- `docs/`: MkDocs knowledge base covering theory, formulas, and worked
  examples.
- `notebooks/`: Reproducible research notebooks for exploratory studies.
- `issues/`: Markdown snapshots of resolved issues archived for auditability.
- `assets/`: Static images shared by the documentation and the Streamlit UI.

## Copula families and calibration methods

The following catalogue summarises the supported families:

- **Gaussian** — correlation matrix parameters with tau inversion, IFM,
  and log-likelihood estimators.
- **Student t** — correlation matrix plus degrees of freedom calibrated
  via tau inversion, IFM, and pseudo/log-likelihood optimisers.
- **Clayton** — scalar theta estimated through tau inversion.
- **Gumbel** — scalar theta estimated through tau inversion.
- **Frank** — scalar theta estimated through tau inversion.
- **Joe** — scalar theta estimated through tau inversion.
- **Ali–Mikhail–Haq (AMH)** — scalar theta estimated through tau inversion.

The calibration registry in `src/workflows/calibration.py` orchestrates data
validation, parameter reconstruction, numerical optimisation, and model
selection scores so that every estimator can be invoked uniformly from the UI
or from Python scripts.

## Sandbox workflows

The Sandbox page supports two complementary flows:

1. **Preset builder** – choose any supported family, configure the correlation
   matrix (positive-definiteness checks included) and draw samples of
   any size.
2. **Custom density designer** – provide symbolic expressions for `C(u)` and
   `c(u)`, map marginal distributions per dimension, and generate pseudo
   observations via importance resampling. Diagnostics include Matplotlib-based
   KDEs, pair plots, heatmaps, and downloadable CSV exports.

## Documentation and references

The MkDocs site under `docs/` mirrors the application flow:

- `01_foundations/`: probability integral transforms, Sklar’s theorem, and
  dependence metrics.
- `02_families/`: closed-form densities, generators, and tail behaviour for
  each supported copula.
- `03_calibration/`: derivations for tau inversion, IFM, and likelihood methods
  with worked numerical examples.
- `04_diagnostics/`: Rosenblatt transforms, density overlays, and QQ analyses.
- `05_simulation/`: sampling algorithms and variance reduction strategies.
- `06_practice_notes/`: implementation checklists and operational guidance.

## Release checklist

This repository marks the `v1.0` release of Copula Lab. All historical issues
are closed, regression tests pass on CI, and the theory documented in `docs/`
is implemented in the application. Future enhancements can branch from the
`develop` target without additional setup.

