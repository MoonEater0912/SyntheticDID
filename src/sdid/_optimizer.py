import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Optimizer:
    def __init__(
            self,
            zeta_omega_type,
            omega_type: str,
            negative_omega: bool,
            random_state: int,
            max_iter: int,
            tol: float
    ):
        # hyper param
        self.zeta_omega_type = zeta_omega_type
        self.omega_type = omega_type
        self.negative_omega = negative_omega
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

        # check hyper param

        if not(
            self.zeta_omega_type in ["base"] or
            isinstance(self.zeta_omega_type, (int, float, complex))
        ):
            raise ValueError("Zeta type (scaling of regularization) not supported!")
        
        if self.omega_type not in ["match", "parallel"]:
            raise ValueError("Omega type (weights of units) not supported!")


    def est_zeta(self, 
                 wide_data,
                 treated_units, 
                 post_treatment_terms,
        ):

        if self.zeta_omega_type == "base":
            scaling = (len(treated_units) * len(post_treatment_terms)) ** (0.25)

            control_units = [u for u in wide_data.columns if u not in treated_units]
            pre_treatment_terms = [t for t in wide_data.index if t not in post_treatment_terms]

            wide_data_co_pre = wide_data.loc[pre_treatment_terms, control_units]
            delta_matrix = wide_data_co_pre.diff(axis=0).dropna()
            tau = delta_matrix.values.std(ddof=0)

            return scaling * tau

        elif isinstance(self.zeta_omega_type, (int, float, complex)):
            return self.zeta_omega_type
        
    def est_omega(
            self,
            wide_data,
            treated_units,
            post_treatment_terms,
            zeta
    ):
        wide_data_pre = wide_data.loc[~wide_data.index.isin(post_treatment_terms), :]
        N_tr = len(treated_units)
        N_co = wide_data.shape[1] - N_tr
        T_pre = wide_data_pre.shape[0]
        T_post = len(post_treatment_terms)

        # define tool functions (objective, constraints)
        def constraint_omega(x):
            if self.omega_type == "parallel":
                return np.sum(x[1:]) - 1
            elif self.omega_type == "match":
                return np.sum(x) - 1
        
        def objective_omega(x):
            if self.omega_type == "match":
                omega_0, omega_co, omega_tr = 0, x, np.ones(N_tr) / (-N_tr)
            elif self.omega_type == "parallel":
                omega_0, omega_co, omega_tr = x[0], x[1:], np.ones(N_tr) / (-N_tr)
            omega_full = np.concatenate([omega_co, omega_tr])
            
            # calculate loss
            loss = omega_0 + wide_data_pre @ omega_full
            loss = np.sum(loss**2)

            # calculate penalty
            penalty = (zeta**2) * T_pre * np.sum(omega_co**2)

            return loss + penalty

        # begin optimizing
        cons = {'type': 'eq', 'fun': constraint_omega}
        if self.omega_type == "parallel":
            bounds = [(None, None)] + [(0, None)] * N_co
            x0 = np.concatenate([[0], np.ones(N_co) / N_co])

        elif self.omega_type == "match":
            if self.negative_omega == True:
                bounds = [(None, None)] * N_co
            else:
                bounds = [(0, None)] * N_co
            x0 = np.ones(N_co) / N_co

        res = minimize(
            objective_omega,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            tol=self.tol,
            options={"maxiter":self.max_iter}
        )

        if not res.success:
            raise ValueError(f"Optimization of Omega (weights of units) failed: {res.message}")
        
        return res.x

    def est_lambda(
            self,
            wide_data,
            treated_units,
            post_treatment_terms,
            zeta
    ):
        wide_data = wide_data.T.copy() # row=unit, col=time
        wide_data_co = wide_data.loc[~wide_data.index.isin(treated_units), :]
        N_tr = len(treated_units)
        N_co = wide_data_co.shape[0]
        T_post = len(post_treatment_terms)
        T_pre = wide_data.shape[1] - T_post
        
        # define tool functions
        def constraints_lambda(x):
            return np.sum(x[1:]) - 1

        def objective_lambda(x):
            lambda_0, lambda_pre, lambda_post = x[0], x[1:], np.ones(T_post) / T_post
            lambda_full = np.concatenate([lambda_pre, lambda_post])

            # calculate loss
            loss = lambda_0 + wide_data_co @ lambda_full
            loss = np.sum(loss**2)

            # calculate penalty
            penalty = (zeta**2) * N_co * np.sum(lambda_pre**2)

            return loss + penalty

        # begin optimizing
        cons = {"type": "eq", "fun": constraints_lambda}
        bounds =  [(None, None)] + [(0, None)] * T_pre
        x0 = np.concatenate([[0], np.ones(T_pre)/T_pre])

        res = minimize(
            objective_lambda,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            tol=self.tol,
            options={"maxiter":self.max_iter}
        )

        if not res.success:
            raise ValueError(f"Optimization of Omega (weights of units) failed: {res.message}")
        
        return res.x
    
