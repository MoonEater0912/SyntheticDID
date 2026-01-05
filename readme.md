# SyntheticDID

`SyntheticDID` is a Python implementation of the **Synthetic Difference-in-Differences (SDID)** estimator. 

> [!NOTE]
> This package is still under development.

### Introduction

While other implementations in Python like `pysynthdid` exist, this package has (aims to have) several advantages:

* **Time-Varying Covariates**: Support for including dynamic controls in estimation.
* **ATT Estimation**: Uses the weighted two-way fixed-effect regression to estimate the Average Treatment Effect on the Treated (ATT), while also keeping the ATT estimated by diff-in-diff. 
* **Relaxed Dependencies**: Updated to be compatible with the latest Python data science stack.
* **Other Features (under construction)**: 
    * Enhanced support for **Event Studies**.
    * Support for **Staggered Adoption** (multiple treatment timing).


### Next Step

* Support for inference
* Support for event studies (estimation of dynamic effects)
* Support for staggered adoption


### References & Resources

**Original Paper**
Arkhangelsky, D., Imbens, G. W., Lei, L., & Hirshberg, D. A. (2021). Synthetic Difference-in-Differences. American Economic Review. https://www.aeaweb.org/articles?id=10.1257/aer.20190159

**Related Paper**
* SDID: https://docs.iza.org/dp15907.pdf
* Covariates: https://github.com/skranz/xsynthdid/blob/main/paper/synthdid_with_covariates.pdf
* SDID event: https://arxiv.org/abs/2407.09565

**Other Implementations**
* R: https://synth-inference.github.io/synthdid/
* Stata: https://github.com/Daniel-Pailanir/sdid 
* Python (pysynthdid): https://github.com/MasaAsami/pysynthdid/tree/main
* Python (synthdid): https://github.com/d2cml-ai/synthdid.py

---

## Installation

Install the development version from GitHub:

```bash
pip install git+https://github.com/MoonEater0912/SyntheticDID.git
```


## Quick Start

### Model Setup

To get started, your data frame should contain at least four columns: 'unit', 'time', 'outcome' and 'treated indicator'. You can specify the column names when fitting the model. The treatment should be absorbed, meaning that the treated indicator should never revert to 0 after treatment. Also, the panel should be balanced, i.e., all units have same number of observations.

You can also specify covariates when fitting the model. The way the package controls for covariates is described in this [working paper](https://github.com/skranz/xsynthdid/blob/main/paper/synthdid_with_covariates.pdf).

Currently, this package only supports same-time adoption.

```python
import pandas as pd
from sdid import SyntheticDID as SDID

# Load your panel data in pd.DataFrame
# it should contain: unit, time, outcome, treatment indicator, and other covariates (if any)
dt = pd.read_csv("your_data.csv")

# initialize the model in default setting
model = SDID()

# fit the model with your data
model.fit(
    dt,
    unit="unit",
    time="time",
    outcome="outcome",
    treated="treated",
    covariates=["x1", "x2"]
)
```
Actually, you can set hyperparameters for optimization process (computing $\zeta$, $\omega$ weights and $\lambda$ weights) during initialization.

```python
model = SDID(
    zeta_omega="base",      # "base" (default), "inf" or a non-negative float
    zeta_lambda=0,          # 0 (default) or a non-negative float, or "inf"
    omega_type="parallel",  # "parallel" (default) or "match"
    negative_omega=False.   # True or False (default)
    max_iter=500,           # Maximum iterations for optimization
    tol=1e-5,               # Tolerance for termination
    sparse_threshold=0      # non-negative float; if zero, it will be ignored
)
```
By default, the model implement the algorithm proposed in the original paper (automatically calculating zeta_omega based on an empirical formula). Set zeta_omega = "inf" and zeta_lambda = "inf" to degrade the model to a standard DID estimator ("inf" means the regularization penalty dominates the optimization, forcing the weights to be uniform). Set zeta_omega=0 and zeta_lambda to ignore the regularization penalty, making the units weights more sparse.  For falied optimazation, consider increasing max_iter and relaxing tol. The underlying optimization is powered by scipy.optimize.minimize (SLSQP).

Set omega_type="match" to degrade the model to a Synthetic Control estimator (getting rid of the intercept, i.e., $\omega_0$). However, when omega_type is set to 'match', the optimizer may fail to converge if the treatment group's characteristics lie outside the convex hull of the donor pool (where it will likely return uniform weights, i.e., degraded to a DID estimator). In such scenarios, one might consider relaxing the non-negativity constraint to allow for negative unit weights (negative_omega = True). However, it is critical to note that this approach introduces the risk of arbitrary extrapolation, which may undermine the structural validity of the synthetic control.

To achieve sparser omega weights to improve interpretability, set sparse_threshold to a positive number `k` (like 0.1) and zeta_omega=0. Once the weights are estimated, any value below the threshold of `k / len(omegas)` is set to zero, and the remaining weights are re-scaled to sum to 1. Note: This approach violates the standard SDID assumptions, so use it sparingly.

### Estimation of ATT

After calling model.fit(), you can access the ATT calculated via different methods::

```python
# ATT via Weighted Two-Way Fixed Effects (The standard SDID result)
print(model.ATT)

# simple weighted Diff-in-Diff result
print(model.ATT_diff)
```

The difference between these two ATTs is: `model.ATT` controls for the unit and the time fixed effects; whicle `model.ATT_diff` simply calculates $\frac{1}{T_{post}}\sum(\frac{1}{N_{tr}}\sum Y_{i,t} - \sum \omega_i Y_{i,t}) - \sum \lambda_t (\frac{1}{N_{tr}}\sum Y_{i,t} - \sum \omega_i Y_{i,t})$. Neither of them provides standard error or confidence interval, which should be accessed through `model.infer()`, where you can specify hyperparameters for the inference model.

You can easily plot the outcome trajectories for treated and synthetic control groups. You can set line features for the control group and the treatment group separately via parameters starting with "co_" or "tr_", like "co_color='red'", "tr_label='treated states'", etc.

```python
model.plot(
    ax=None, show=True, time_weights=True,
    xlabel = "Time",
    ylabel = "Outcome",
    title = "Synthetic Difference-in-Differences: Trajectories",
    **kwargs
)
```

The plots show the outputs of SDID (left) and SC (with negative weights) (right) on california_prop99 dataset.

<p align="center">
  <img src="images/eg_sdid.png" width="45%" />
  <img src="images/eg_sc.png" width="45%" />
</p>


### Inference

