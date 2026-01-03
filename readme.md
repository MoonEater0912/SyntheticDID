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

* Support for sparse estimation
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

```python
import pandas as pd
from sdid import SyntheticDID as SDID

# Load your panel data in pd.DataFrame
# it should contain: unit, time, outcome, treatment indicator, and other covariates (if any)
dt = pd.read_csv("your_data.csv")

# initialize the model
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
    zeta_omega="base",   # "base" (default) or a non-negative float
    zeta_lambda=0,       # 0 (default) or a non-negative float
    omega_type="parallel", # "parallel" (default) or "match"
    negative_omega=False. # True or False (default)
    max_iter=500,        # Maximum iterations for optimization
    tol=1e-6             # Tolerance for termination
)
```
By default, the model implement the algorithm proposed in the original paper. Set a large zeta_omega to degrade the model to a standard DID estimator. Set zeta_omega=0 to ignore the regularization penalty, making the units weights set more sparse. Set omega_type="match" to degrade the model to a Synthetic Control estimator. For large datasets, consider increasing max_iter and relaxing tol. The underlying optimization is powered by scipy.optimize.minimize (SLSQP).

When omega_type is set to 'match', the optimizer may fail to converge if the treatment group's characteristics lie outside the convex hull of the donor pool (where it will likely return uniform weights, i.e., degraded to a DID estimator). In such scenarios, one might consider relaxing the non-negativity constraint to allow for negative unit weights (negative_omega = True). However, it is critical to note that this approach introduces the risk of arbitrary extrapolation, which may undermine the structural validity of the synthetic control.

After calling model.fit(), you can access the ATT calculated via different methods::

```python
# ATT via Weighted Two-Way Fixed Effects (The standard SDID result)
print(model.ATT)

# simple Diff-in-Diff result
print(model.ATT_diff)
```

You can easily plot the outcome trajectories for treated and synthetic control groups:

```python
model.plot(
    ax=None, show=True
)
```

The plots show the outputs of SDID (left) and SC (with negative weights) (right) on california_prop99 dataset.

<p align="center">
  <img src="images/eg_sdid.png" width="45%" />
  <img src="images/eg_sc.png" width="45%" />
</p>
