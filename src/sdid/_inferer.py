import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import scipy.stats as stats

class Inferer:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)

    def gen_boot_data(
            self,
            raw_data,
    ):
        while True:
            all_units = raw_data["unit"].unique()
            boot_units = self.rng.choice(
                all_units,
                size=len(all_units),
                replace=True
            )
            
            # constructing boot sample frame, allowing multiple appearance of one unit
            chunks = []
            for i, u in enumerate(boot_units):
                unit_data = raw_data[raw_data["unit"] == u].copy()
                unit_data["unit"] = f"{u}_boot_{i}"
                chunks.append(unit_data)
            res = pd.concat(chunks, ignore_index=True)

            if res["treated"].nunique() >= 2:
                break
            
        return res


    def gen_placebo_data(
            self,
            N_tr: int,
            co_data,
            co_units,
            treatment_year
    ):
        placed_units = self.rng.choice(
            co_units,
            size=N_tr,
            replace=False
        )

        placed_data = co_data.copy()
        placed_data = placed_data.assign(
            treated_new = lambda x: (x["unit"].isin(placed_units)) & (x["time"] >= treatment_year),
        )

        return placed_data


    def vcoc(self, estimate, se):
        z_critical_95 = stats.norm.ppf(1 - 0.05/2)
        ci_lower_95 = estimate - z_critical_95 * se
        ci_upper_95 = estimate + z_critical_95 * se

        z_critical_90 = stats.norm.ppf(1 - 0.10/2)
        ci_lower_90 = estimate - z_critical_90 * se
        ci_upper_90 = estimate + z_critical_90 * se

        z_score = estimate / se
        p = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "estimate": estimate,
            "Std.Err": se,
            "ci_90": (ci_lower_90, ci_upper_90),
            "ci_95": (ci_lower_95, ci_upper_95),
            "z_value": z_score,
            "p_value": p
        }


    def bootstrapping(
            self, ATT_twfe, ATT_diff,
            raw_data,
            covariates,
            model,
            rep: int,
    ):
        att_twfe, att_diff = [], []

        for i in tqdm(range(rep), desc="bootstrapping ..."):
            j = 0
            while True: # in case optimization failed
                try:
                    booted_data = self.gen_boot_data(raw_data=raw_data)
                    model.fit(
                        data = booted_data,
                        outcome_col = "outcome",
                        unit_col = "unit",
                        time_col = "time",
                        treated_col = "treated",
                        covariate_cols = covariates
                    )
                    att_twfe.append(model.ATT)
                    att_diff.append(model.ATT_diff)
                    break 
                except:
                    j = j + 1
                    if j >= 5:
                        raise ValueError("Optimization failed too many times, please change to another method of inference.")
                    continue # redo current loop

        se_twfe = np.std(att_twfe, ddof=0)
        se_diff = np.std(att_diff, ddof=0)

        infered_twfe = self.vcoc(ATT_twfe, se_twfe)
        infered_diff = self.vcoc(ATT_diff, se_diff)

        return {
            "ATT": infered_twfe,
            "ATT_diff": infered_diff
        }
    

    def placeboing(
            self, ATT_twfe, ATT_diff,
            raw_data,
            covariates,
            model,
            rep: int,
    ):
        att_twfe, att_diff = [], []

        N_tr = raw_data[raw_data["is_treated"] == 1].unit.nunique()
        co_data = raw_data[raw_data["is_treated"] == 0]
        co_data = co_data.drop(['treated', 'is_treated'], axis=1)
        co_units = co_data.unit.unique()
        treatment_year = raw_data[raw_data["treated"] == 1].time.min()

        for i in tqdm(range(rep), desc="placeboing ..."):
            j = 0 # count times of failed optimization
            while True:
                try:
                    placed_data = self.gen_placebo_data(N_tr, co_data, co_units, treatment_year)
                    model.fit(
                                data = placed_data,
                                outcome_col = "outcome",
                                unit_col = "unit",
                                time_col = "time",
                                treated_col = "treated_new",
                                covariate_cols = covariates
                            )
                    att_twfe.append(model.ATT)
                    att_diff.append(model.ATT_diff)
                    break
                except:
                    j = j + 1
                    if j >= 5:
                        raise ValueError("Optimization failed too many times, please change to another method of inference.")
                    continue
        
        se_twfe = np.std(att_twfe, ddof=0)
        se_diff = np.std(att_diff, ddof=0)

        infered_twfe = self.vcoc(ATT_twfe, se_twfe)
        infered_diff = self.vcoc(ATT_diff, se_diff)

        return {
            "ATT": infered_twfe,
            "ATT_diff": infered_diff
        }


    def jackknifing(
            self,
            data,
            ATT_twfe,
            ATT_diff,
    ):
        data = data.loc[data["sdid_weight"] > 0]
        all_units = data["unit"].unique()
        for u in all_units:
            jack_dt = data[data["unit"] != u]

