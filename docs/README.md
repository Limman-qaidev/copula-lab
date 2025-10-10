# Copula Lab — Theoretical Documentation

This directory hosts the **complete theoretical component** of the
**Copula Lab** project. It explains every mathematical and statistical
aspect required to understand, calibrate, and evaluate copula models.

---

## Purpose

Copula Lab provides an interactive and reproducible environment to
calibrate, compare, and study dependence models in multivariate data.
The documentation here serves as the *textbook* behind the app — it
covers all families, measures, algorithms, and theoretical results in
detail.

---

## Folder Structure

```
docs/
  README.md                # This file (overview and main references)
  SUMMARY.md               # Navigation index (for mkdocs or Streamlit links)
  01_foundations/
    01_overview.md         # What is a copula?
    02_sklar_theorem.md    # Sklar's theorem and its implications
    03_rank_measures.md    # Spearman's rho, Kendall's tau
    04_tail_dependence.md  # Tail dependence coefficients λ_U, λ_L
  02_families/
    01_elliptical_gaussian.md
    02_elliptical_student_t.md
    10_arch_clayton.md
    11_arch_gumbel.md
    12_arch_frank.md
    13_arch_joe.md
  03_calibration/
    01_pseudo_mle.md
    02_ifm.md
    03_tau_inversion.md
    04_distance_min_cvm.md
  04_diagnostics/
    01_rosenblatt_pit.md
    02_gof_ks_cvm.md
    03_dependence_summaries.md
  05_simulation/
    01_random_variates.md
    02_reproducibility.md
  06_practice_notes/
    01_model_risk_pitfalls.md
    02_finance_use_cases.md
  assets/
    figures/
    tables/
```
---

## Math and Notation

- Vectors: **bold** or $\mathbf{U}$, dimension $d$.
- Copula: $C:[0,1]^d\to[0,1]$, density $c(u_1,\dots,u_d)$.
- Marginals: $F_i$, quantiles $F_i^{-1}$.
- Parameters: $\rho, \Sigma, \nu, \theta$.
- Dependence measures: Kendall’s $\tau$, Spearman’s $\rho$, tails
  $\lambda_U$, $\lambda_L$.

---

## Figures and Tables

All figures go to `docs/assets/figures/`, tables to
`docs/assets/tables/`. Each section will include visual aids:

| Type | Description | Example filename |
|------|--------------|------------------|
| Figure | Sklar theorem diagram | sklar_construction.png |
| Figure | Gaussian vs t-copula contours | gauss_vs_t_contour.png |
| Figure | Archimedean generator ψ examples | archimedean_generators.png |
| Table | Family summary (τ, λ, symmetry) | copula_families_summary.csv |

---


## References

1. Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
2. Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
3. McNeil, A. J., Frey, R., & Embrechts, P. (2015).
   *Quantitative Risk Management*, 2nd ed. Princeton Univ. Press.
4. Haugh, M. *An Introduction to Copulas*. Columbia University.
5. Schmidt, T. (2007). *Tail Dependence*. University of Cologne.
