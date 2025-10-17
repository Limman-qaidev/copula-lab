# Instructions for Agents

This document defines the guidelines and
conventions that Copilot should follow when
 generating content (code comments,
 documentation) for quantitative analysis
 and risk modeling projects. The goal is to
 ensure consistency, quality, and aligment
 with the advanced level required.

 ---

 ## 1- General Objective

 - Provide automatic code assistance in **Python 3**
 that meets professional, academic, and production standards.q  

 - Generate code snippets that include:
    1. **Step-by-step mathematical derivations**, with clear contextualization
    of each variable and its meaning.
    2. **Concise documentation** (docstrings and comments) explaining assumptions,
    limitations, and purpose of functions.
    3. **Practical examples** only when explicitly requested.
    4. references to key academic papers if applicable (e.g., DOI, authors, year).

---

## 2. Code Style Guide

### 2.1 Language and Version

- **Python 3** (at least 3.8).
- Use **`typing`** for static type annotations in all functions and classes.

### 2.2 PEP 8 Conventions

- Maximum line length: **79 characters**.
- Identitation: **4 spaces**, no tabs.
- Variable and function names in **snake_case**.
- Class names in **CamelCase**.
- Avoid importing whole modules; import only necessary symbols:

```python
from typing import List, Tuple, Optional
```

- Separate logical blocks and code sections with blank lines according PEP 8.

### 2.3 Code Quality Tools Compliance

**ALL generated code MUST pass the following tools without any errors or warnings:**

#### 2.3.1 Mypy (Static Type Checking)
- **Zero tolerance for type errors**: All code must pass `mypy --strict`
- Use proper type annotations for all function parameters and return values
- Handle Optional types explicitly with proper None checks
- Use generics correctly: `List[str]`, `Dict[str, int]`, etc.
- Import typing symbols at module level, never inside functions

```python
from typing import List, Optional, Dict, Any
import numpy as np

def calculate_metrics(data: List[float]) -> Dict[str, Optional[float]]:
    """All parameters and returns must be properly typed."""
    if not data:
        return {"mean": None, "std": None}
    return {"mean": np.mean(data), "std": np.std(data)}
```

#### 2.3.2 Pylance (VS Code Language Server)
- **Zero errors**: Code must not show red underlines in VS Code
- **Minimize warnings**: Address yellow underlines when possible
- Use explicit imports, avoid wildcard imports (`from module import *`)
- Ensure all variables are defined before use
- Handle potential `None` values properly

#### 2.3.3 Black (Code Formatting)
- **Automatic formatting compliance**: Code must pass `black --check`
- Use Black's 88-character line limit OR 79 for PEP 8 strict projects
- Let Black handle string quotes, spacing, and line breaks
- Never manually format code that conflicts with Black

#### 2.3.4 Flake8 (Linting and Style)
- **Zero violations**: Code must pass `flake8` with no output
- Address all error codes: E, W, F, C, N series
- Common violations to avoid:
  - `E501`: Line too long (max 79 characters)
  - `F401`: Imported but unused
  - `F841`: Local variable assigned but never used
  - `W293`: Blank line contains whitespace
  - `E302`: Expected 2 blank lines before class/function

```python
# Good: Flake8 compliant
import logging
from typing import List

logger = logging.getLogger(__name__)


class VaRCalculator:
    """Two blank lines before class definition."""
    
    def __init__(self, confidence: float = 0.95) -> None:
        """One blank line between methods."""
        self.confidence = confidence
        
    def calculate(self, returns: List[float]) -> float:
        """No trailing whitespace, proper spacing."""
        if not returns:
            raise ValueError("Returns list cannot be empty")
        # Implementation here...
        return 0.0
```

#### 2.3.5 Pre-commit Quality Gates
All code must pass these checks before commit:

```bash
# Required passing commands:
mypy --strict your_file.py          # No type errors
pylance (via VS Code)               # No red underlines  
black --check your_file.py          # Properly formatted
flake8 your_file.py                 # No style violations
```

**Enforcement Rules:**
1. **Never commit code** that fails any of these tools
2. **Fix all issues** before submitting pull requests  
3. **Use type: ignore sparingly** and only with justification
4. **Document any exceptions** in code comments with reasoning

### 2.4 Modularity

- Each Python file should focus on a single purpose (e.g., VaR calculation,
parameter, calibration, visualization).
- Split into functions or classes if a single logic grows beyond ~50 lines.
- Always export with __all__ or include a section at the end of each module indicating
what is publicly imported.

### 2.5 Docstrings and Comments

- Use **Google-style docstrings**.
Example:

```python
def calculate_var(
    return_series: List[float],
    confidence_level: float
) -> float:
    """
    Calculates historical Value at Risk (VaR) at a given confidence level.

    Args:
    return_series (List[float])
        List of daily protfolio returns.
    confidence_level (float): 
        Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    float: Estimated historical VaR.

    Raises:
        ValueError: If 'return_series' is empty or 'confidence_level' 
        is not in (0, 1).
    """
    # Step-by-step percentile calculation
    ...
```

- Each critical variable must be briefly described in the docstring
in the `Args`: section or in a inline comment
- Include **assumptions** (e.g., stationarity, independence of returns)
and **limitations** of the method.
- Document **exceptions** in a Raises sectoin:
    - Specify exception type and condition triggering it.
    - Example:

    ```python
    Raises
    ------
    ValueError
        If `p` or `q` parameters are negative.
    ```

- Reference **logging** practicesin comments:
    - Use Python's built-in logging module.
    - At the top of each module, instantiate a module-level logger:

    ```python
    import logging

    logger = logging.getLogger(__name__)
    ```

    - In code, insert log statements to record key steps, warnings, or errors.

    ```python
    if not return_series:
        logger.error("Empty return_series provided to calculate_var")
        raise ValueError("return_series must contain at least one value")
    ```

    - Comments should mention which log level to use (e.g., logger.debug, logger.info,
    logger.warning, logger.error).

## 3. Mathematical Structure and Reporting

### 3.1 Step-by-Step Derivations

- When Copilot generates derivations (e.g., parametric VaR calculation, derivative of
density functions, parameter optimization), it must:

    1. **Contextualize** each symbol (e.g., "Let $X_t$ be the position value at time $t$)
    2. Explain ** underlying assumptions** (e.g., normal distribution, linear correlation).
    3. Display equations with **LaTeX notation** if included in comments or documentation-do
    not render HTML, only inline LaTeX (e.g., returns in percentage).
    4. Indicate the **unit** of each variable when applicable (e.g., returns in percentage).

### 3.2 Academic References

- If logic or formulas originate from a paper or book, add a brief reference in the docstring
or comment:

```
Reference:

    - Jorion, P. (2007). `Value at Risk: The New Benchmark for Managing Financial Risk.` McGraw-Hill.
    - Hull. J. (2018). `Options, Futures, and Other Derivatives (10th ed.)`. Pearson
```

- Copilot may suggest a DOI or links to arXiv preprints if available.

---

## 4. Project and Module Structure

### 4.1. Folder Organization

project/
│
├── data/            
│   ├── raw/          # Original unprocessed data
│   └── processed/    # Transformed data ready for analysis
│
├── notebooks/        # Jupyter notebooks for exploration and demostrations
│   └── var_exploration.ipynb
│
├── src/              # Source code
│   ├── estimators/  # Parameter estimation function
│   │   └── parametric_var.py
│   ├── modelos/      # Model definitions (e.g., Monte Carlo, GARCH)
│   │   └── garch.py
│   ├── utils/        # Utility functions (e.g., data loading, visualizatoins)
│   │   └── io.py
│   └── main.py       # Main script to run pipelines
│
├── tests/            # Unit tests (pytest)
│   ├── test_var.py
│   └── test_estimators.py
│
├── docs/             # Documentation, technical notes, and results
│   └── var_report.md
│
├── requirements.txt
└── copilot-instructions.md  # ← This file

### 4.2. Unit Testing

- Copilot should generate tests using **pytest**.
- Each critical function musth have at least:
- One nominal test case (expected values).
- One edge-case test (e.g., confidence level near 0 or 1).
- Name tests files with prefix `test_` and test funcitons as `test_<function_name>`.

---

## 5. Visualization with Matplotlib

- Generate plots with **Matplotlib** without specifying color palettes (use default style).
- Each figure must include:
    - Axis labels with units (e.g., "Return (%)" vs. "Density").
    - Descriptive title and legend if multiple series exist.
    - Brief inline comment explaining the plot interpretation.
- Minimal example in docstring:

```python
import matplotlib.pyplot as plt
from typing import List

def plot_return_distribution(returns: List[float]) -> None:
    """
    Plots the empirical distribution of returns.

    Args: 
        returns (List[float]): Daily historical returns.
    """
    plt.figure()
    plt.hist(returns, bins=50, density=True)
    plt.xlabel("Return (%)")
    plt.ylabel("Density")
    plt.title("Empirical Return Distribution")
    plt.show()
```

---

## 6. Tone and Lebel of Detail

- **Formal**, **scientific**, **and direct**
- Avoid unnecessary "`ffluff`": get straight to the point.
- Always adopt a **skeptical and curious** attitude. If Copilot suggests a method, it should:
    1. Point out assumptions and limitations.
    2. Suggest possible extensions or improvements.

- If a topic may require data verification or updating, include a comment such as:

```
"`Verify recent updates in market risk literature (2023-2025) to confirm assumptions`"
```

--- 

## 7. Assumptions and Limitations

- Always document:
    - **Satistical assumptions**  (normality, independence, stationarity).
    - **Data limitations** (sample size, data, quality).
    - **Regulatory implications** if applicable (e.g., Basel III requirements).
- Example:

```python
def calibrate_garch(
    return_series: List[float],
    p: int = 1,
    q: int = 1
) -> Tuple[List[float], List[float]]:
    """
    Calibrates a GARCH(p, q) model to the return series.

    Assumptions:
    - Returns are mean-zero.
    - Conditional volatility modeled as GARCH.
    - No abrupt jumps in the data sereis.

    Limitations:
    - Does not capture heavy-tail effects without extensions
    (e.g., t-student GARCH).
    - Requires sufficient historical data (minimum ~500 observations).

    Raises:
        ValueError: If `return_series` length is below required threshold.
    
    """
    ...
```

---

## 8. Code Quality Enforcement

### 8.1 Mandatory Compliance

**ALL CODE MUST PASS ALL QUALITY TOOLS BEFORE SUBMISSION:**

1. **Mypy**: `mypy --strict filename.py` → **ZERO ERRORS**
2. **Pylance**: No red underlines in VS Code → **ZERO ERRORS** 
3. **Black**: `black --check filename.py` → **PROPERLY FORMATTED**
4. **Flake8**: `flake8 filename.py` → **ZERO VIOLATIONS**

### 8.2 Quality Gates

```bash
# Pre-commit checklist (ALL must pass):
mypy --strict *.py                  # ✅ Type checking
black --check *.py                  # ✅ Code formatting  
flake8 *.py                        # ✅ Style compliance
python -m pytest tests/            # ✅ Unit tests
```

### 8.3 Zero Tolerance Policy

- **No code should be committed** that fails any of these tools
- **All warnings must be addressed** or explicitly justified with comments
- **Type hints are mandatory** for all function signatures
- **Line length must not exceed 79 characters** (PEP 8 compliance)
- **All imports must be used** (no unused imports)

**Copilot should generate code that passes all these checks by default.**