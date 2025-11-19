# advanced_visualizations.py
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sinusoid_features import build_features

# Rolling survival for each planet

def plot_rolling_survival_by_planet(df: pd.DataFrame,window: int = 30, save_path: str | None = None):
    planets = sorted(df["planet"].unique())

    plt.figure(figsize=(12, 6))

    for p in planets:
        sub = df[df["planet"] == p].sort_values("steps_taken")
        roll = sub["survived"].rolling(window=window, min_periods=1).mean()

        label = sub["planet_name"].iloc[0] if "planet_name" in sub.columns else f"Planet {p}"
        plt.plot(sub["steps_taken"], roll, label=label)

    plt.xlabel("Step")
    plt.ylabel(f"Rolling survival using window={window}")
    plt.title("Rolling survival probability by planet")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)

    plt.show()
    plt.close()

#Volatility
def plot_survival_volatility_by_planet(df: pd.DataFrame,window: int = 30,save_path: str | None = None,):
    plt.figure(figsize=(12, 6))
    planets = sorted(df["planet"].unique())

    for p in planets:
        sub = df[df["planet"] == p].sort_values("steps_taken")
        roll_std = sub["survived"].rolling(window=window, min_periods=1).std()

        label = sub["planet_name"].iloc[0] if "planet_name" in sub.columns else f"Planet {p}"
        plt.plot(sub["steps_taken"], roll_std, label=label)

    plt.xlabel("Step")
    plt.ylabel(f"Rolling std of survival (window={window})")
    plt.title("Survival volatility by planet")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)

    plt.show()
    plt.close()


# Spectre de fréquences (FFT)

def plot_fft_survival_by_planet(df: pd.DataFrame,save_path: str | None = None,):
    planets = sorted(df["planet"].unique())

    n_planets = len(planets)

    plt.figure()

    for i, p in enumerate(planets, start=1):
        sub = df[df["planet"] == p].copy()
        sub = sub.sort_values("steps_taken")

        step_mean = (sub.groupby("steps_taken")["survived"].mean().sort_index())

        idx = range(step_mean.index.min(), step_mean.index.max() + 1)
        series = step_mean.reindex(idx).ffill().fillna(step_mean.mean())

        y = series.values - series.values.mean()
        n = len(y)

        fft_vals = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, d=1)  
        amplitude = np.abs(fft_vals)

        plt.subplot(n_planets, 1, i)
        plt.plot(freqs, amplitude)
        label = sub["planet_name"].iloc[0] if "planet_name" in sub.columns else f"Planet {p}"
        plt.title(f"FFT of survival time series for  {label}")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


# Heatmap phase 

def plot_phase_survival_heatmaps(df: pd.DataFrame,periods: list[int] | None = None,save_path: str | None = None,):
    planets = sorted(df["planet"].unique())

    if periods is None:
        periods = [10, 20, 200]


    n_periods = len(periods)
    plt.figure(figsize=(14, 4 * n_periods))

    for i, T in enumerate(periods, start=1):
        mat = []
        row_labels = []

        for p in planets:
            sub = df[df["planet"] == p].copy()
            sub = sub.sort_values("steps_taken")
            sub["phase"] = sub["steps_taken"] % T

            phase_mean = (
                sub.groupby("phase")["survived"]
                .mean()
                .reindex(range(T))
            )

            mat.append(phase_mean.values)
            label = sub["planet_name"].iloc[0] if "planet_name" in sub.columns else f"Planet {p}"
            row_labels.append(label)

        mat = np.array(mat)

        plt.subplot(n_periods, 1, i)
        im = plt.imshow(mat, aspect="auto", origin="lower", vmin=0, vmax=1)
        plt.colorbar(im, label="Survival probability")
        plt.yticks(range(len(row_labels)), row_labels)
        plt.xticks(range(0, periods[i-1], max(1, T // 10)))
        plt.xlabel(f"Phase (step mod {T})")
        plt.title(f"Phase survival heatmap with period={T}")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def create_advanced_visualizations(df: pd.DataFrame,window, output_dir: str = "advanced_plots"):
    os.makedirs(output_dir, exist_ok=True)

    print("Creating advanced visualizations...")

    plot_rolling_survival_by_planet(df,window=window,save_path=os.path.join(output_dir, "rolling_survival_by_planet.png"))

    plot_survival_volatility_by_planet(df,window=window,save_path=os.path.join(output_dir, "survival_volatility_by_planet.png"))
    plot_fft_survival_by_planet(df, save_path=os.path.join(output_dir, "fft_survival_by_planet.png") )

    plot_phase_survival_heatmaps(df,periods=[10, 20, 200],save_path=os.path.join(output_dir, "phase_survival_heatmaps.png") )
    print(f"Advanced visualizations saved to {output_dir}/")



def plot_linucb_fit_vs_empirical(run_path: str = "late_run.csv",theta_prefix: str = "theta_planet_",window: int = 10):

    df_explo = pd.read_csv(run_path)
    def predict_prob(theta: np.ndarray, planet: int, step: int) -> float:
        theta = theta.reshape(-1, 1)
        x = build_features(planet, step, d=theta.shape[0]).reshape(-1, 1)
        mu = float(theta.T @ x)
        return max(0.0, min(1.0, mu))

    out_dir = "linucb_plots"
    os.makedirs(out_dir, exist_ok=True)

    for planet in [0, 1, 2]:
        print(f"\n=== Planet {planet} ===")
        theta_path = f"{theta_prefix}{planet}.npy"
        theta_p = np.load(theta_path)

        df_p = df_explo[df_explo["planet"] == planet].copy()
        if "steps_taken" in df_p.columns:
            time_col = "steps_taken"
        elif "step" in df_p.columns:
            time_col = "step"
        elif "trip_number" in df_p.columns:
            time_col = "trip_number"
        else:
            df_p = df_p.reset_index(drop=True)
            df_p["step_index"] = df_p.index
            time_col = "step_index"
        df_p = df_p.sort_values(time_col)

        # empirique lissé
        rolling_emp = df_p["survived"].rolling(window, center=True).mean()
        steps = df_p[time_col].values

        # prédictions lissées
        preds = np.array([predict_prob(theta_p, planet, s) for s in steps])
        rolling_preds = pd.Series(preds).rolling(window, center=True).mean()

        plt.figure(figsize=(11, 4))
        plt.plot(steps, rolling_emp, label=f"Empirique lissé (window={window})", linewidth=2, alpha=0.9)
        plt.plot(steps, rolling_preds, label=f"LinUCB lissé (window={window})", linestyle="--", linewidth=2)

        plt.title(f"Planet {planet} : survie réelle vs prédite (rolling={window})")
        plt.xlabel("Step")
        plt.ylabel("Proba de survie")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"linucb_fit_vs_empirical_planet_{planet}.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()
        print(f"Plot saved to {save_path}")