import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

from sdid._optimizer import Optimizer
from sdid._plotter import Plotter
from sdid._inferer import Inferer

class SyntheticDID:
    def __init__(
            self, 
            zeta_omega="base", # ["base", "inf" float]
            zeta_lambda=0,     # ["inf", float]
            omega_type="parallel", # ["match", "parallel"]
            negative_omega=False, # [True, False]
            random_state=42,
            max_iter=500,
            tol=1e-5,
            sparse_threshold: float = 0 # only controls the sparsity of omega 
    ):
        
        self.random_state = random_state

        # optimizer
        self._optimizer = Optimizer(
            zeta_omega_type=zeta_omega, omega_type=omega_type, negative_omega=negative_omega,
            max_iter=max_iter, tol=tol,
            sparse_threshold=sparse_threshold
        )

        # data
        self._raw_data = None
        self._data = None
        self._wide_data = None
        self._covariates = None
        
        # param
        self._zeta_omega = None
        self._zeta_lambda = zeta_lambda
        self._omega = None
        self._lambda_ = None
        self._att = None
        self._att_diff = None

        # flag
        self._is_fitted = False

        # plotter
        self._plotter = Plotter()

        # inferer
        self._inferer = Inferer(random_state = random_state)

    def fit(
            self,
            data: pd.DataFrame,
            outcome_col: str,
            unit_col: str,
            time_col: str,
            treated_col: str, 
            covariate_cols: Optional[List[str]] = None,
    ):

        self._is_fitted = False  # reset the flag

        # print("check data and transform ...")
        variables = ["outcome", "unit", "time", "treated"]
        if covariate_cols:
            variables.extend(covariate_cols)
        self._data = (
            data
            .rename(columns={
                outcome_col: 'outcome',
                unit_col: 'unit',
                time_col: 'time',
                treated_col: 'treated'
            })
            .copy()[variables]
        )

        self._sort_rows()
        self._check_panel() # check if balanced, non-staggered and absorbed

        self._raw_data = self._data  # raw data: without adjusting for Y

        if covariate_cols: # adjust for outcomes by time-varying covariates
            self._covariates = covariate_cols
            self._adjust_outcomes()

        self._transform_to_wide() # make wide table, row=time, col=unit

        # print("optimizing ...")
        self._zeta_omega = self._optimizer.est_zeta(
            self.wide_data, 
            self.treated_units, 
            self.post_treatment_terms, 
        )

        self._omega = self._optimizer.est_omega(
            self.wide_data,
            self.treated_units,
            self.post_treatment_terms,
            self._zeta_omega
        )

        self._lambda_ = self._optimizer.est_lambda(
            self.wide_data,
            self.treated_units,
            self.post_treatment_terms,
            self._zeta_lambda
        )

        # print("estimating ...")
        self._att_diff = self._est_att_diff()
        self._make_reg_data()
        self._att = self._est_att()

        self._is_fitted = True
        # print("Done!")

        return self
        

    def _adjust_outcomes(self):
        df = self._data[self._data["treated"] == 0].copy()
        df = df.set_index(['unit', 'time'])

        covariate_cols = [name for name in self.data.columns if name not in ["unit", "time", "outcome", "treated", "is_treated"]]
        covariate_formula = " + ".join(covariate_cols)
        reg_model = PanelOLS.from_formula(
            f'outcome ~  1 + {covariate_formula} + EntityEffects + TimeEffects', 
            data=df,
            drop_absorbed=True
        )
        reg_res = reg_model.fit(cov_type='clustered', cluster_entity=True)
        effective_covariates = [c for c in covariate_cols if c in reg_res.params.index]
        reg_params = reg_res.params[effective_covariates]

        self._data['outcome'] = self._data['outcome'] - (self._data[effective_covariates] @ reg_params)

    def _check_panel(self):
        if self.data.duplicated(subset=["unit", "time"]).any():
            raise ValueError("Data contains duplicated observations (unit, year).")

        period_counts = self.data.groupby("unit")["time"].nunique()
        expected_count = self.data["time"].nunique()

        if not (period_counts == expected_count).all():
            raise ValueError("Data is not a balanced panel. Please make sure all units have same periods of observation.")
        
        if set(self.data["treated"].to_list()) != {0, 1}:
            raise ValueError("Treatment status column contains values other than 0/1 or there's only one treatment status in the dataset.")
        
        status_diff = self.data.groupby("unit")["treated"].diff()
        if (status_diff == -1).any():
            raise ValueError("Data contains non-absorbed treatement. Please make sure treated units do not revert their treatement status.")
        
        treated_durations = self.data.groupby("unit")["treated"].sum()
        actual_treated_durations = treated_durations[treated_durations > 0]
        
        if len(actual_treated_durations) > 0:
            if actual_treated_durations.nunique() != 1:
                raise ValueError("Data contains staggered setup, please use sdid.model.SDIDEvent.")
            if (actual_treated_durations == expected_count).any():
                raise ValueError("Units are treated from the very beginning (1st period), which is not allowed in SDID estimation.")
        else:
            raise ValueError("Data doesn't contain treated units.")   

    def _sort_rows(self):
        group_map = self.data.groupby("unit")["treated"].max().rename('is_treated')
        self._data = self.data.merge(group_map, on="unit")
        self._data = self.data.sort_values(
            by=['is_treated', "unit", "time"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    def _transform_to_wide(self):
        self._wide_data = self.data.pivot(index="time", columns="unit", values="outcome")
        self._wide_data = self.wide_data.loc[self.data["time"].unique(), self.data["unit"].unique()]

    @property
    def treated_units(self):
        return set(self.data.loc[self.data["treated"] == 1, "unit"].unique())
    
    @property
    def post_treatment_terms(self):
        return set(self.data.loc[self.data["treated"] == 1, "time"].unique())

    @property
    def wide_data(self):
        return self._wide_data
    
    @property
    def data(self):
        return self._data

    @property
    def covariates(self):
        self._check_fitted()
        return self._covariates

    def _check_fitted(self):
        if self._is_fitted == False:
            raise ValueError("The model is not fitted yet, or the last fitting failed.")

    def _est_att_diff(self):
        T_post = len(self.post_treatment_terms)
        lambda_full = np.concatenate([
            self._lambda_[1:],
            np.ones(T_post)/(-T_post)
        ])
        pre_post_diff = self.wide_data.T @ lambda_full

        N_tr = len(self.treated_units)
        if self._optimizer.omega_type == "match":
            omega_full = np.concatenate([
                self._omega,
                np.ones(N_tr)/(-N_tr)
            ])
        elif self._optimizer.omega_type == "parallel":
            omega_full = np.concatenate([
                self._omega[1:],
                np.ones(N_tr)/(-N_tr)
            ])
        return pre_post_diff @ omega_full

    @property
    def ATT_diff(self):
        self._check_fitted()
        return self._att_diff

    def _make_reg_data(self):
        N_tr = len(self.treated_units)
        if self._optimizer.omega_type == "parallel":
            omega_full = np.concatenate([
                self._omega[1:],
                np.ones(N_tr)/N_tr
            ])
        elif self._optimizer.omega_type == "match":
            omega_full = np.concatenate([
                self._omega,
                np.ones(N_tr)/N_tr
            ])

        unit2weight = dict(zip(
            self._data.unit.unique(),
            omega_full
        ))

        T_post = len(self.post_treatment_terms)
        lambda_full = np.concatenate([
            self._lambda_,
            np.ones(T_post)/T_post
        ])
        time2weight = dict(zip(
            self._data.time.unique(),
            lambda_full
        ))
        
        self._data["sdid_weight"] = self._data.apply(
            lambda x: unit2weight[x["unit"]] * time2weight[x["time"]],
            axis=1
        )
        
    def _est_att(self):
        df_for_reg = self.data.loc[self.data["sdid_weight"] > 0]
        df_for_reg = df_for_reg.set_index(["unit","time"])

        reg_model = PanelOLS(
            df_for_reg["outcome"],
            df_for_reg["treated"],
            entity_effects = True, 
            time_effects = True,
            weights=df_for_reg["sdid_weight"],
        ).fit()

        return reg_model.params.iloc[0]
    
    @property
    def ATT(self):
        self._check_fitted()
        return self._att


    # ============================================================
    # Below are functions that can be called only after fitted
    # ============================================================

    @property
    def zeta_omega(self):
        self._check_fitted()
        return self._zeta_omega
    
    @property
    def zeta_lambda(self):
        self._check_fitted()
        return self._zeta_lambda
    
    @property
    def omega(self):
        self._check_fitted()
        return self._omega
    
    @property
    def lambda_(self):
        self._check_fitted()
        return self._lambda_


    @property
    def _clone(self):
        return SyntheticDID(
            zeta_omega=self._optimizer.zeta_omega_type,
            zeta_lambda=self._zeta_lambda,
            omega_type=self._optimizer.omega_type,
            negative_omega=self._optimizer.negative_omega,
            max_iter=self._optimizer.max_iter,
            tol=self._optimizer.tol,
            sparse_threshold = self._optimizer.sparse_threshold
        )

    @property
    def trajectories(self):
        self._check_fitted()
        
        wide_data_co = self.wide_data.loc[:, ~self.wide_data.columns.isin(self.treated_units)].copy()
        wide_data_tr = self.wide_data.loc[:, self.wide_data.columns.isin(self.treated_units)].copy()

        trajectories_dt = pd.DataFrame({
            "time": [],
            "control": [],
            "treated": []
        })

        trajectories_dt["time"] = list(self.wide_data.index)
        if self._optimizer.omega_type == "parallel":
            trajectories_dt["control"] = list(wide_data_co @ self.omega[1:])
        elif self._optimizer.omega_type == "match":
            trajectories_dt["control"] = list(wide_data_co @ self.omega)
        trajectories_dt["treated"] = list(wide_data_tr.mean(axis=1))

        return trajectories_dt

    def plot_trajectories(
            self, ax=None, show=True, time_weights=True,
            xlabel = "Time",
            ylabel = "Outcome",
            title = "Synthetic Difference-in-Differences: Trajectories",
            **kwargs
        ):

        self._plotter.trajectories(
            self, ax=ax, show=show, time_weights=time_weights,
            xlabel = xlabel,
            ylabel = ylabel,
            title = title,
            **kwargs
        )

    
    def infer(
            self,
            method: Literal["placebo", "bootstrap", "jackknife"] = "bootstrap",
            rep:int = 500
    ):
        self._check_fitted()

        # reproduce itself
        clone = self._clone

        # sending task
        if method == "bootstrap":
            return self._inferer.bootstrapping(
                raw_data=self._raw_data,
                ATT_twfe=self.ATT,
                ATT_diff=self.ATT_diff,
                covariates=self.covariates,
                model=clone,
                rep=rep
            )
        elif method == "jackknife":
            return self._inferer.jackknifing(
                data=self._data,
                ATT_twfe=self.ATT,
                ATT_diff=self.ATT_diff,
            )
        elif method == "placebo":
            pass
        else:
            raise ValueError("The indicated method is not supported (choose from [placebo, bootstrap, jackknife]).")




