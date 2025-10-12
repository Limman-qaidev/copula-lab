# 4. Tail Dependence

## 4.1 Context and Motivation

Linear correlation does not describe joint extremes.
Tail dependence coefficients quantify the probability that one variable
is extreme given that another is.

---

## 4.2 Definitions

Upper tail dependence:
$$
\lambda_U
 = \lim_{q\uparrow 1}
   \Pr\!\big(X_2>F_2^{-1}(q)\mid X_1>F_1^{-1}(q)\big),
\tag{7}
$$

Lower tail dependence:
$$
\lambda_L
 = \lim_{q\downarrow 0}
   \Pr\!\big(X_2\le F_2^{-1}(q)\mid X_1\le F_1^{-1}(q)\big).
\tag{8}
$$

When the limits exist, $0\le \lambda_L,\lambda_U\le1$.

---

## 4.3 Examples

- **Gaussian copula:** $\lambda_U=\lambda_L=0$ (asymptotic independence).  
- **Student-t copula:** symmetric, $\lambda_U=\lambda_L>0$, with

$$
\lambda = 2\,T_{\nu+1}
\!\left(-\sqrt{\tfrac{\nu+1}{1+\rho}(1-\rho)}\right),
\tag{9}
$$

where $T_{\nu+1}$ is the CDF of a t-distribution with $\nu+1$ d.f.

---

## 4.4 Illustration

**Figure 1:** *Tail dependence curves λ(ρ, ν) for t-copula*  
![t-copula lambda dependece all](../assets/figures/01_foundations/t_lambda_dependence_all.svg)
Each curve shows how heavier tails (smaller ν) imply stronger tail
dependence.

**Figure 2:** *Contour comparison (Gaussian vs t-copula)*  
*(to be generated via notebook 04_tail_dependence.ipynb)*  
The t-copula contours remain dense in the corners,
while the Gaussian vanish — illustrating asymptotic independence.

---

## 4.5 Remarks

- Tail dependence is invariant under strictly increasing transforms.  
- For Archimedean families, upper and lower tails may differ (asymmetry).  
- In risk modeling, λ informs joint default or co-exceedance
  probabilities beyond correlation.

---

## 4.6 References

- Schmidt T. (2007). *Tail Dependence*, Univ. of Cologne.  
- Nelsen (2006), §5.4.  
- McNeil, Frey & Embrechts (2015), §5.5.
