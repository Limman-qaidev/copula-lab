# 2. Sklar’s Theorem

## 2.1 Context and Motivation

Sklar’s theorem (1959) provides the theoretical foundation of copulas:
it formalizes the separation between marginals and dependence.

---

## 2.2 Statement

Let $F$ be a $d$-dimensional CDF with marginals $F_1,\ldots,F_d$.
Then there exists a copula $C$ such that

$$
F(x_1,\ldots,x_d)
  = C\!\big(F_1(x_1),\ldots,F_d(x_d)\big).
\tag{2}
$$

If $F_i$ are continuous, $C$ is unique.  
Conversely, for any copula $C$ and univariate CDFs $F_i$, the
composition (2) defines a valid joint distribution.

If densities exist, the factorization reads

$$
f_X(x_1,\ldots,x_d)
  = c\!\big(F_1(x_1),\ldots,F_d(x_d)\big)
    \prod_{i=1}^d f_i(x_i),
\tag{3}
$$

where $c=\partial^d C / (\partial u_1\cdots\partial u_d)$ is the copula
density.

---

## 2.3 Interpretation

Equation (3) expresses that **dependence enters only through $c$**.
Inference may thus proceed via:
- *Inference Functions for Margins* (IFM): estimate $F_i$ first,
  then $C$.
- *Pseudo-MLE*: replace $F_i$ by empirical $\hat F_i$ directly.

---

## 2.4 Illustration

**Figure:** *Functional composition diagram*  
![Sklar composition](../assets/figures/01_foundations/sklar_functional_composition.svg)

Arrows depict the transformation
$x_i \mapsto u_i=F_i(x_i)$ feeding the copula $C$, producing the joint
$F(x)$. This visualizes the functional composition that underlies all
copula modeling.

---

## 2.5 References

- Sklar, A. (1959). *Fonctions de répartition à n dimensions...*  
- Nelsen (2006), Ch. 2.  
- Haugh (M.), *An Introduction to Copulas*, Columbia Univ.

