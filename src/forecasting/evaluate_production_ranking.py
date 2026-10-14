"""Evaluate next-day Top-10 depletion ranking for production and profile baseline.

Run after ``python -m src.forecasting.production_model`` on the real
historical dataset. Research-model prediction files are not used.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")
TOP_K = 10


def load_predictions(data_dir=DATA_DIR):
    frames = {}
    for target in ("departures", "arrivals"):
        path = data_dir / f"production_test_{target}_predictions.csv"
        frames[target] = pd.read_csv(
            path,
            parse_dates=["datetime"],
            dtype={"station_id": "string"},
        )

    departures, arrivals = frames["departures"], frames["arrivals"]
    if departures.duplicated(["station_id", "datetime"]).any():
        raise ValueError("Duplicate departure station-hour predictions")
    if arrivals.duplicated(["station_id", "datetime"]).any():
        raise ValueError("Duplicate arrival station-hour predictions")

    combined = departures.merge(
        arrivals, on=["station_id", "datetime"], validate="one_to_one"
    )
    if len(combined) != len(departures) or len(combined) != len(arrivals):
        raise ValueError("Departure and arrival predictions differ in coverage")
    return combined


def compute_daily_depletion(predictions):
    """Match rank_stations.py: maximum of zero and negative min cumulative net flow."""
    df = predictions.copy()
    df["date"] = df["datetime"].dt.date
    df = df.sort_values(["date", "station_id", "datetime"])

    definitions = {
        "actual": ("arrivals", "departures"),
        "production": ("predicted_arrivals", "predicted_departures"),
        "profile": ("baseline_arrivals", "baseline_departures"),
    }

    daily = None
    for name, (arrivals, departures) in definitions.items():
        net_flow = df[arrivals] - df[departures]
        grouped = [df["date"], df["station_id"]]
        cumulative_flow = net_flow.groupby(grouped, sort=False).cumsum()
        min_flow = cumulative_flow.groupby(grouped, sort=False).min()
        depletion = (-min_flow).clip(lower=0).rename(f"{name}_depletion")
        daily = depletion.to_frame() if daily is None else daily.join(depletion)

    return daily.reset_index()


def evaluate(daily, top_k=TOP_K):
    rows = []
    for date, frame in daily.groupby("date", sort=True):
        n_stations = len(frame)
        if n_stations < top_k:
            raise ValueError(f"Only {n_stations} stations for {date}; need {top_k}")

        # Stable ordering for exact-score ties, including zero-depletion stations.
        frame = frame.sort_values("station_id", kind="stable")
        actual_top = set(frame.nlargest(top_k, "actual_depletion")["station_id"])
        result = {"date": date, "n_stations": n_stations}
        for method in ("production", "profile"):
            chosen = set(frame.nlargest(top_k, f"{method}_depletion")["station_id"])
            overlap = len(actual_top & chosen)
            result[f"{method}_overlap"] = overlap
            result[f"{method}_hit_rate"] = overlap / top_k
        result["random_hit_rate"] = top_k / n_stations
        rows.append(result)

    if not rows:
        raise ValueError("No evaluation days found")
    return pd.DataFrame(rows)


def main():
    hourly = load_predictions()
    daily = compute_daily_depletion(hourly)
    scores = evaluate(daily)
    output = DATA_DIR / "production_ranking_daily_results.csv"
    scores.to_csv(output, index=False)

    print(f"\nTop-{TOP_K} ranking evaluation over {len(scores)} test days")
    for name in ("production", "profile"):
        print(
            f"{name.title():<12} overlap/day: {scores[f'{name}_overlap'].mean():.2f}; "
            f"mean hit rate: {scores[f'{name}_hit_rate'].mean():.1%}; "
            f"median hit rate: {scores[f'{name}_hit_rate'].median():.1%}"
        )
    print(f"Random expected Top-{TOP_K} hit rate: {scores['random_hit_rate'].mean():.1%}")
    print(f"Per-day results saved to: {output}")


if __name__ == "__main__":
    main()
