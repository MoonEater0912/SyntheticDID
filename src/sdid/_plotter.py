import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def init(self):
        pass

    def trajectories(
            self, model, ax=None, show=True, time_weights=True,
            xlabel = "Time",
            ylabel = "Outcome",
            title = "Synthetic Difference-in-Differences: Trajectories",
            **kwargs
        ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        df = model.trajectories

        control_kwargs = {k.replace('co_', ''): v for k, v in kwargs.items() if k.startswith('co_')}
        treated_kwargs = {k.replace('tr_', ''): v for k, v in kwargs.items() if k.startswith('tr_')}
        common_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('co_', 'tr_'))}

        c_style = {**{'color': 'blue', 'ls': '--', 'label': 'Control'}, **common_kwargs, **control_kwargs}
        t_style = {**{'color': 'orange', 'ls': '-', 'label': 'Treated'}, **common_kwargs, **treated_kwargs}

        ax.plot(df['time'], df['control'], **c_style)
        ax.plot(df['time'], df['treated'], **t_style)


        if show:
            ax.axvline(
                x=np.min(list(model.post_treatment_terms)), 
                color='gray', 
                linestyle=':', 
                label='Treatment Start',
                alpha=0.8
            )

            if time_weights:
                ax_weight = ax.twinx()
                pre_treatment_periods = [t for t in model.wide_data.index if t not in model.post_treatment_terms]
                ax_weight.bar(pre_treatment_periods, model.lambda_[1:], alpha=0.3, color='gray', label='Time Weights')
                ax_weight.set_ylim(0, max(model.lambda_[1:]) * 8) 
                ax_weight.set_ylabel('Time Weights ($\lambda$)')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()