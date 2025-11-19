import pandas as pd
from api_client import SphinxAPIClient
from data_collector import DataCollector
from LinUCBStrategy import LinUCBStrategy
from visualizations import create_all_visualizations
from advanced_visualizations import create_advanced_visualizations, plot_linucb_fit_vs_empirical
import sys
def exploration_phase(client: SphinxAPIClient, df_path: str | None = None) -> pd.DataFrame:
    collector = DataCollector(client)

    if df_path is not None:
        print(f"\n Loading exploration data from {df_path}...")
        df = pd.read_csv(df_path)
    else:
        print("\n Starting new exploration phase...")
        df = collector.explore_all_planets(trips_per_planet=300, morty_count=1)
        df.to_csv("exploration_data.csv", index=False)
        print(" Exploration data saved to 'exploration_data.csv'")

    return df


def analyze_planets(client: SphinxAPIClient, df: pd.DataFrame) -> dict:
    collector = DataCollector(client)

    print("\n Analyzing run...")
    risk_analysis = collector.analyze_risk_changes(df)

    print("\n Risk Analysis:")
    for planet_name, data in risk_analysis.items():
        print(f"\n{planet_name}:")
        print(f"  Overall Survival Rate: {data['overall_survival_rate']:.2f}%")
        print(f"  Early Survival Rate:   {data['early_survival_rate']:.2f}%")
        print(f"  Late Survival Rate:    {data['late_survival_rate']:.2f}%")
        print(f"  Trend: {data['trend']} ({data['change']:+.2f}%)")

    print("\n Determining best planet...")
    best_planet, best_planet_name = collector.get_best_planet(df, consider_trend=True)
    print(f"Best planet: {best_planet_name} (index {best_planet})")

    create_all_visualizations(df, output_dir="plots")
    create_advanced_visualizations(df, window=30, output_dir="advanced_plots")
    return risk_analysis

def run_strategy(client: SphinxAPIClient):
    client.start_episode()
    strategy = LinUCBStrategy(client, d=5)
    strategy.execute_strategy()

def post_strategy_analysis(path="last_run.csv"):
    plot_linucb_fit_vs_empirical(run_path=path,theta_prefix="theta_planet_",window=10)
    create_advanced_visualizations(pd.read_csv(path),window=10,output_dir="post_strategy_advanced_plots")
    create_all_visualizations(pd.read_csv(path),output_dir="post_strategy_plots")


def main():
    try:
        client = SphinxAPIClient()
        print("✓ API client initialized successfully!")
    except ValueError as e:
        print(f"Error while initializing API client: {e}")
        return

    # OPTIONAL EXPLORATION PHASE
    if DO_EXPLORATION:
        print("\n=== Exploration Phase ===")
        df = exploration_phase(client, df_path="exploration_data.csv")

        print("\n=== Analysis of exploration data ===")
        analyze_planets(client, df)
    else:
        print("\nSkipping pure exploration phase ")
        df = pd.read_csv("exploration_data.csv") 

    # STRATEGY EXECUTION
    if NO_RUN:
        print("\nskipping strategy execution phase ")
    else:
        print("\n=== Strategy Execution Phase ===")
        run_strategy(client)

    # POST-STRATEGY
    if NO_POST_ANALYSIS:
        print("\nskipping post-strategy analysis ")
    else:
        print("\n=== Post Strategy Analysis ===")
        post_strategy_analysis(path="last_run.csv")


if __name__ == "__main__":
    DO_EXPLORATION = "--explore" in sys.argv
    NO_RUN = "--no-run" in sys.argv
    NO_POST_ANALYSIS = "--no-post" in sys.argv
    main()
