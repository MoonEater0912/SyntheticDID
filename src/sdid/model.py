import pandas as pd
import numpy as np
from typing import List, Optional, Literal
import matplotlib.pyplot as plt
import math
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
            sparse_threshold=sparse_threshold,
            random_state=random_state
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
                raise ValueError("Data contains staggered setup, please use StaggeredSyntheticDID.")
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
    
    # ============================================================
    # Below are functions that can be called only after fitted
    # ============================================================

    @property
    def covariates(self):
        self._check_fitted()
        return self._covariates

    def _check_fitted(self):
        if self._is_fitted == False:
            raise ValueError("The model is not fitted yet, or the recent fitting failed.")

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
            figsize = (10, 6),
            xlabel = "Time",
            ylabel = "Outcome",
            title = "Synthetic Difference-in-Differences: Trajectories",
            **kwargs
        ):

        self._plotter.trajectories(
            self, ax=ax, show=show, time_weights=time_weights,
            figsize= figsize,
            xlabel = xlabel,
            ylabel = ylabel,
            title = title,
            **kwargs
        )


    def _infer2summary(self, infer_dict):
        print("Synthetic Difference-in-Differences Estimations")
        print("-" * 65)
        header = f"{'Parameter':<10} | {'Estimate':>15} | {'Std. Error':>12} | {'z-score':>10} | {'p > |z|':>10}"
        print(header)
        print("-" * 65)

        for param, metrics in infer_dict.items():
            est = metrics['estimate']
            stderr = metrics['Std.Err']
            z_val = metrics['z_value']
            p_val = metrics['p_value']
            ci_lower, ci_upper = metrics['ci_95']

            stars = ""
            if p_val < 0.01: stars = "***"
            elif p_val < 0.05: stars = "**"
            elif p_val < 0.1: stars = "*"

            print(f"{param:<10} | {est:>12.4f}{stars:<3} | {stderr:>12.4f} | {z_val:>10.3f} | {p_val:>10.4e}")
            print(f"{'':<10} | [{ci_lower:>5.3f}, {ci_upper:>5.3f}] (95% CI)")
            
        print("-" * 65)
        print("Significance codes:  *** p<0.01, ** p<0.05, * p<0.1")

    def infer(
            self,
            method: Literal["placebo", "bootstrap", "jackknife"] = "bootstrap",
            rep:int = 500,
            summary_show = True
    ):
        self._check_fitted()

        # reproduce itself
        clone = self._clone

        # sending task
        if method == "bootstrap":
            infer_dict = self._inferer.bootstrapping(
                raw_data=self._raw_data,
                ATT_twfe=self.ATT,
                ATT_diff=self.ATT_diff,
                covariates=self.covariates,
                model=clone,
                rep=rep
            )
        elif method == "jackknife":
            if len(self.treated_units) < 2:
                raise ValueError("Jackknife is supported only for data containing more than one treated unit. Please use method = bootstrap or placebo.")
            infer_dict = self._inferer.jackknifing(
                    ATT_twfe=self.ATT,
                    data=self.data
                )
        elif method == "placebo":
            if 2*len(self.treated_units) >= len(self.wide_data.columns):
                raise ValueError("There are more treated units than control units. Please use method = bootstrap or jackknife in this case.")
            infer_dict = self._inferer.placeboing(
                raw_data=self._raw_data,
                ATT_twfe=self.ATT,
                ATT_diff=self.ATT_diff,
                covariates=self.covariates,
                model=clone,
                rep=rep
            )
        else:
            raise ValueError("The indicated method is not supported (choose from [placebo, bootstrap]).")
        
        if summary_show:
            self._infer2summary(infer_dict=infer_dict)

        return infer_dict



class StaggeredSyntheticDID:
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
        # single adoption estimater
        self._single_adopter = SyntheticDID(
            zeta_omega, zeta_lambda, omega_type, negative_omega,
            random_state, max_iter, tol, sparse_threshold
        )

        # data
        self._data = None               # total panel
        self._adoption_times = None     # cohort periods
        self._unit_adoption_map = None  # {unit: cohort}
        self._cohorts = None            # list of sub-panels for each cohort 
        self._treated_unit_nums = None  # lits of numbers of treatment units}

        # estimater for each cohort
        self._estimaters = {}           # {cohort: estimater}

        # estimates
        self._att = None
        self._att_diff = None
        self._atts = None
        self._atts_diff = None


    def fit(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        treated_col: str, 
        covariate_cols: Optional[List[str]] = None,
    ):
        # flag
        self._is_fitted = False

        # check data and get adoption periods
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

        self._check_panel()
        self._split_panel()

        # loop for single adoption estimating
        for i in range(len(self._adoption_times)):
            try:
                cohort = self._adoption_times[i]
                panel = self._cohorts[i]

                estimater = self._single_adopter._clone
                self._estimaters[cohort] = estimater.fit(
                    panel, 
                    "outcome", "unit", "time", "treated",
                    covariate_cols
                )
            except Exception as e:
                raise ValueError(f"An error occurs when fitting cohort {cohort}. Detailed message: {e}")
        
        # estimate att and att_diff
        self._est_atts()
        
        self._is_fitted = True
        return self



        

    def _check_panel(self):
        if self.data.duplicated(subset=["unit", "time"]).any():
            raise ValueError("Data contains duplicated observations (unit, year).")

        period_counts = self.data.groupby("unit")["time"].nunique()
        expected_count = self.data["time"].nunique()

        if not (period_counts == expected_count).all():
            raise ValueError("Data is not a balanced panel. Please make sure all units have same periods of observation.")
        
        if set(self.data["treated"].to_list()) != {0, 1}:
            raise ValueError("Treatment status column contains values other than 0/1 or there's only one treatment status in the dataset.")

        sorted_data = self._data.sort_values(['unit', 'time'])
        treatment_times = (
            sorted_data[sorted_data['treated'] == 1]
            .groupby('unit')['time']
            .min()
        )
        
        unique_treatment_periods = sorted_data['time'].unique()
        all_periods = sorted(unique_treatment_periods)
        first_period = all_periods[0]
        actual_treatment_periods = set(treatment_times.unique())

        ## Below are checks in terms of the adoption times
        # if the earliest adoption time is the first year
        if first_period in actual_treatment_periods:
            raise ValueError(f"The earliest adoption time should not be the very first period ({first_period})。")

        # check if not staggered
        if len(actual_treatment_periods) <= 1:
            raise ValueError("The adoption is not staggered, please use SyntheticDID instead; or the dataset doesn't contain treated units.")

        # check if there is control unit
        total_units = self._data['unit'].nunique()
        if len(treatment_times) == total_units:
            raise ValueError("All units in the dataset are treated. The dataset should contain at least one control unit.")

        self._adoption_times = sorted(list(actual_treatment_periods))
        self._unit_adoption_map = treatment_times.to_dict()

    def _split_panel(self) -> List[pd.DataFrame]:
        all_units = set(self._data['unit'].unique())
        treated_units = set(self._unit_adoption_map.keys())
        never_treated_units = all_units - treated_units

        sub_panels = []
        treated_unit_nums = []

        for t_treat in self._adoption_times:
            current_treated_units = [
                unit for unit, adopt_t in self._unit_adoption_map.items() 
                if adopt_t == t_treat
            ]
            target_units = current_treated_units + list(never_treated_units)
            
            sub_panel = self._data[self._data['unit'].isin(target_units)].copy()
            sub_panels.append(sub_panel)
            treated_unit_nums.append(len(current_treated_units))
        
        if len(sub_panels) != len(self._adoption_times):
            raise ValueError("Something happens ...")
    
        self._cohorts = sub_panels
        self._treated_unit_nums = treated_unit_nums

    def _est_atts(self):
        self._atts = [
            estimater.ATT for estimater in list(self._estimaters.values())
        ]

        self._atts_diff = [
            estimater.ATT_diff for estimater in list(self._estimaters.values())
        ]

        self._att = np.average(self._atts, weights=self._treated_unit_nums)
        self._att_diff = np.average(self._atts_diff, weights=self._treated_unit_nums)

    @property
    def data(self):
        return self._data


    # ============================================================
    # Below are functions that can be called only after fitted
    # ============================================================

    def _check_fitted(self):
        if self._is_fitted == False:
            raise ValueError("The model is not fitted yet, or the recent fitting failed.")
        
    @property
    def single_fits(self):
        self._check_fitted()
        return self._estimaters

    @property
    def ATT(self):
        self._check_fitted()
        return self._att
    
    @property
    def ATT_diff(self):
        self._check_fitted()
        return self._att_diff

    @property
    def ATT_by_cohort(self):
        self._check_fitted()
        return self._atts
    
    @property
    def ATT_diff_by_cohort(self):
        self._check_fitted()
        return self._atts_diff


    def plot_all_trajectories(
        self, ax=None, time_weights=True,
        figsize = (6, 4),
        xlabel = "Time",
        ylabel = "Outcome",
        title = "Staggered Synthetic Difference-in-Differences: Trajectories",
        ncols:int = 3,
        **kwargs
    ):
        self._check_fitted()

        nrows = int(np.ceil(len(self._adoption_times) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
        axes_flat = axes.flatten()

        for i, (cohort, model) in enumerate(self._estimaters.items()):
            ax = axes_flat[i]
            model.plot_trajectories(
                ax=ax, show=False, time_weights=time_weights,
                figsize=figsize, 
                xlabel=xlabel, ylabel=ylabel, 
                title=f"Cohort {cohort}", 
                **kwargs
            )
        
        for j in range(len(self._adoption_times), len(axes_flat)):
            fig.delaxes(axes_flat[j])
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()




    