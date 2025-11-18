# generate_planet_function.py
import pandas as pd
from api_client import SphinxAPIClient
from data_collector import DataCollector

client = SphinxAPIClient()
collector = DataCollector(client)

all_runs = []

for run_id in range(3):   
    print(f"\n Exploration run {run_id} ...")
    df_run = collector.explore_all_planets(trips_per_planet=300, morty_count=1)
    df_run["run_id"] = run_id
    all_runs.append(df_run)

df_all = pd.concat(all_runs, ignore_index=True)
df_all.to_csv("exploration_data.csv", index=False)
print("Saved exploration_data.csv")





















# def plot_dynamic_survival(file_path="exploration_data.csv", window=10):
#     df = pd.read_csv(file_path)
    
#     df['step'] = df.groupby('planet')['survived'].cumcount()
#     df['rolling_survival'] = df.groupby('planet')['survived'].transform(
#         lambda x: x.rolling(window=window, min_periods=1).mean()
#     )
    
#     plt.figure(figsize=(10, 6))
#     for planet_id in df['planet'].unique():
#         subset = df[df['planet'] == planet_id]
#         plt.plot(subset['step'], subset['rolling_survival'], label=f'Planet {planet_id}')
    
#     plt.xlabel("Trip step")
#     plt.ylabel(f"Rolling Survival Rate (window={window})")
#     plt.title("Dynamic Survival Rate per Planet")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def plot_planet_dynamics_simple(df: pd.DataFrame, window: int = 50):
    """
    Pour chaque planète :
    - trace la courbe de rolling survival (0-1) en fonction du trip index local.
    """
    df = df.copy()
    df['step'] = df.groupby('planet')['survived'].cumcount()
    df['rolling_survival'] = df.groupby('planet')['survived'].transform(
        lambda x: x.astype(int).rolling(window=window, min_periods=1).mean()
    )

    plt.figure(figsize=(10, 6))

    for pid in sorted(df['planet'].unique()):
        sub = df[df['planet'] == pid].sort_values('step')
        name = sub['planet_name'].iloc[0]
        plt.plot(sub['step'], sub['rolling_survival'], label=f"{name} (planet {pid})")

    plt.xlabel("Trip index (per planet)")
    plt.ylabel(f"Rolling survival (window={window})")
    plt.title("Dynamic survival per planet (simple)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_planet_advanced(per_planet_ts: dict, summary_adv: pd.DataFrame,
                         planet_id: int, max_lag: int = 50):
    """
    Graphe avancé pour une planète :
      - 1er subplot : rolling_survival + points colorés par régime (low / medium / high)
      - 2e subplot : spectre FFT (power vs fréquence) -> périodicité
      - 3e subplot : autocorr(rolling_survival) vs lag -> structure temporelle
    """

    if planet_id not in per_planet_ts:
        print(f"No time series for planet {planet_id}")
        return

    ts = per_planet_ts[planet_id].copy().sort_values("step")
    steps = ts["step"].values
    rs = ts["rolling_survival"].values
    regimes = ts["regime"].values

    row = summary_adv[summary_adv["planet"] == planet_id].iloc[0]
    planet_name = row["planet_name"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # -------------------------------
    # 1) Rolling survival + régimes
    # -------------------------------
    ax1 = axes[0]
    ax1.plot(steps, rs, linewidth=2, label="rolling survival")

    color_map = {"low": "red", "medium": "orange", "high": "green"}
    for reg in ["low", "medium", "high"]:
        mask = regimes == reg
        if mask.any():
            ax1.scatter(steps[mask], rs[mask], s=20, c=color_map[reg], label=reg)

    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Rolling survival")
    ax1.set_title(
        f"Planet {planet_id} – {planet_name} (low={row['frac_low_regime']:.2f}, "
        f"mid={row['frac_medium_regime']:.2f}, high={row['frac_high_regime']:.2f})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # -------------------------------
    # 2) FFT spectrum
    # -------------------------------
    ax2 = axes[1]
    n = len(rs)
    if n >= 4:
        rs_centered = rs - np.nanmean(rs)
        rs_centered = np.nan_to_num(rs_centered)

        fft_vals = np.fft.rfft(rs_centered)
        freqs = np.fft.rfftfreq(len(rs_centered), d=1.0)
        power = np.abs(fft_vals) ** 2

        if len(power) > 1:
            freqs_no0 = freqs[1:]
            power_no0 = power[1:]
            ax2.stem(freqs_no0, power_no0, basefmt=" ")  # ⚠️ pas de use_line_collection
        else:
            ax2.stem(freqs, power, basefmt=" ")

    ax2.set_xlabel("Frequency (cycles / step)")
    ax2.set_ylabel("Power")
    ax2.set_title(
        f"FFT spectrum – main_freq={row['fft_dominant_freq']:.3f}, "
        f"score={row['fft_periodicity_score']:.2f}"
    )
    ax2.grid(True, alpha=0.3)

    # -------------------------------
    # 3) Autocorrelation
    # -------------------------------
    ax3 = axes[2]
    if n >= 2:
        rs2 = np.nan_to_num(rs)
        rs_centered = rs2 - rs2.mean()
        denom = rs_centered.std()

        if denom < 1e-8:
            ac = np.ones(1)
            lags = np.array([0])
        else:
            rs_norm = rs_centered / denom
            ac_full = np.correlate(rs_norm, rs_norm, mode="full")
            ac = ac_full[ac_full.size // 2 :] / n

            max_lag_eff = min(max_lag, n - 1)
            lags = np.arange(0, max_lag_eff + 1)
            ac = ac[: max_lag_eff + 1]

        ax3.stem(lags, ac, basefmt=" ")
        ax3.axvline(
            row["autocorr_peak_lag"],
            color="red",
            linestyle="--",
            label=f"peak lag={int(row['autocorr_peak_lag'])}",
        )

    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Autocorrelation")
    ax3.set_title(
        f"Autocorrelation – peak lag={int(row['autocorr_peak_lag'])}, "
        f"value={row['autocorr_peak_val']:.2f}"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

def plot_planet_risk_over_time(file_path="full_exploration.csv", window=20):
    df = pd.read_csv(file_path)

    df["trip_id"] = df.groupby("planet")["survived"].cumcount()

    df["rolling_survival"] = df.groupby("planet")["survived"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    plt.figure(figsize=(10, 6))
    for pid in df["planet"].unique():
        planet_df = df[df["planet"] == pid]
        plt.plot(planet_df["trip_id"], planet_df["rolling_survival"], label=f"Planet {pid}")
    
    plt.title("Rolling Survival Rate per Planet Over Time")
    plt.xlabel("Trip Number (per planet)")
    plt.ylabel(f"Survival Rate (rolling window={window})")
    plt.legend()
    plt.grid(True)
    plt.show()
