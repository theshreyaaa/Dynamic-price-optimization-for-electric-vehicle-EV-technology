"""
EV Charging Hourly Feature Engineering
-------------------------------------
Creates hourly occupancy, transformer-load, local-time, and TOU-price features
from the cleaned Caltech ACN charging-session CSV.

Usage:
    python src/hourly_feature_engineering.py \
        --input "caltech_acn_cleaned_merged (1).csv" \
        --output "ev_charging_hourly_features.csv"

Dependency:
    pandas
    numpy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# These capacity values are retained from the original notebook.
TOTAL_STALLS = 54
TRANSFORMER_CAPACITY_KW = 150
LOCAL_TIMEZONE = "America/Los_Angeles"


def get_base_price(hour: int, is_weekend: bool) -> float:
    """Return the notebook's base time-of-use price for a local hour."""
    if is_weekend:
        if 0 <= hour < 8:
            return 0.13
        elif 8 <= hour < 16:
            return 0.13
        elif 16 <= hour < 21:
            return 0.18
        else:
            return 0.13

    if 0 <= hour < 8:
        return 0.15
    elif 8 <= hour < 16:
        return 0.28
    elif 16 <= hour < 21:
        return 0.48
    else:
        return 0.28


def build_hourly_features(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Build the 9 hourly features and save them to a CSV file."""
    df = pd.read_csv(input_csv)

    required_input_columns = [
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "spaceID",
        "charging_rate_kw",
    ]
    missing_columns = [col for col in required_input_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    # Parse all session timestamps as timezone-aware UTC values.
    for column in ("connectionTime", "disconnectTime", "doneChargingTime"):
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    df["charging_rate_kw"] = pd.to_numeric(df["charging_rate_kw"], errors="coerce")

    # Preserve the original notebook's continuous 2019 UTC hourly timeline.
    hours = pd.date_range(
        start="2019-01-01 00:00:00",
        end="2019-12-31 23:00:00",
        freq="h",
        tz="UTC",
    )
    hourly_df = pd.DataFrame({"timestamp": hours})
    hourly_df["occupancy_rate"] = 0.0
    hourly_df["transformer_load_ratio"] = 0.0

    # For each UTC hour, count unique occupied stalls and integrate charging
    # power over the overlap between each charging session and the hour.
    for i, hour_start in enumerate(hourly_df["timestamp"]):
        hour_end = hour_start + pd.Timedelta(hours=1)

        connected = df[
            (df["connectionTime"] < hour_end)
            & (df["disconnectTime"] > hour_start)
        ]
        occupied_stalls = connected["spaceID"].nunique()
        hourly_df.loc[i, "occupancy_rate"] = min(
            occupied_stalls / TOTAL_STALLS, 1.0
        )

        charging = df[
            (df["connectionTime"] < hour_end)
            & (df["doneChargingTime"] > hour_start)
            & (df["charging_rate_kw"] > 0)
        ]

        total_kwh_in_hour = 0.0
        for _, session in charging.iterrows():
            active_start = max(hour_start, session["connectionTime"])
            active_end = min(hour_end, session["doneChargingTime"])
            active_hours = max(
                0.0, (active_end - active_start).total_seconds() / 3600
            )
            total_kwh_in_hour += session["charging_rate_kw"] * active_hours

        # A one-hour interval means kWh accumulated during the hour is
        # numerically equal to average kW over that hour.
        average_load_kw = total_kwh_in_hour
        hourly_df.loc[i, "transformer_load_ratio"] = (
            average_load_kw / TRANSFORMER_CAPACITY_KW
        )

    # Convert UTC timestamps to California local time. The timezone database
    # applies PST/PDT daylight-saving changes automatically.
    local_time = hourly_df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    hourly_df["local_hour"] = local_time.dt.hour
    hourly_df["local_dayofweek"] = local_time.dt.dayofweek
    hourly_df["is_weekend"] = (hourly_df["local_dayofweek"] >= 5).astype(int)

    hourly_df["hour_sin"] = np.sin(
        2 * np.pi * hourly_df["local_hour"] / 24
    )
    hourly_df["hour_cos"] = np.cos(
        2 * np.pi * hourly_df["local_hour"] / 24
    )

    hourly_df["base_tou_price"] = [
        get_base_price(hour, weekend == 1)
        for hour, weekend in zip(
            hourly_df["local_hour"], hourly_df["is_weekend"]
        )
    ]

    final_columns = [
        "timestamp",
        "occupancy_rate",
        "transformer_load_ratio",
        "local_hour",
        "local_dayofweek",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "base_tou_price",
    ]
    final_df = hourly_df[final_columns].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output shape: {final_df.shape}")
    print(f"Output saved to: {output_csv}")
    print("\nMissing values:")
    print(final_df.isna().sum())
    print("\nTOU price counts:")
    print(final_df["base_tou_price"].value_counts().sort_index())

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create hourly EV charging optimization features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("caltech_acn_cleaned_merged (1).csv"),
        help="Path to the cleaned charging-session CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ev_charging_hourly_features.csv"),
        help="Path for the generated hourly feature CSV.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {args.input}. "
            "Pass its path using --input."
        )

    build_hourly_features(args.input, args.output)


if __name__ == "__main__":
    main()
