"""
PHASE 5.2: Load response to the Phase 4.4 dynamic prices
Uses the demand formula and elasticities from Phase 5.1.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "hourly_pricing_optimized.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "phase5_2_load_response.csv"

ELASTICITY = {0: -1.2, 1: -0.8, 2: -0.4}   # same as Anshika's 5.1
STATE_NAMES = {0: "Surplus", 1: "Nominal", 2: "Peak Stress"}

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Demand after drivers react to the dynamic price
ratio = df["dynamic_price"] / df["base_tou_price"]
eps = df["state"].map(ELASTICITY)
df["new_kwh"] = df["hourly_kwh"] * ratio ** eps

# Load in kW (energy in 1 hour = average kW)
df["new_load_kw"] = df["new_kwh"]
df["base_load_kw"] = df["hourly_kwh"]

# Revenue per hour
df["rev_before"] = df["base_tou_price"] * df["hourly_kwh"]
df["rev_after"] = df["dynamic_price"] * df["new_kwh"]


def par(x):
    return x.max() / x.mean()


high = df["base_load_kw"].quantile(0.90)   # "high load" = top 10% of baseline hours

print("=== LOAD RESPONSE TO DYNAMIC PRICES ===")
print(f"Peak load (kW):        {df['base_load_kw'].max():.1f} -> {df['new_load_kw'].max():.1f}")
print(f"Peak-to-Average Ratio: {par(df['base_load_kw']):.3f} -> {par(df['new_load_kw']):.3f}")
print(f"Hours above {high:.1f} kW: {(df['base_load_kw'] > high).sum()} -> {(df['new_load_kw'] > high).sum()}")
print(f"Total energy (kWh):    {df['hourly_kwh'].sum():,.0f} -> {df['new_kwh'].sum():,.0f}")
print(f"Revenue ($):           {df['rev_before'].sum():,.2f} -> {df['rev_after'].sum():,.2f}"
      f"  ({(df['rev_after'].sum() / df['rev_before'].sum() - 1) * 100:+.2f}%)")
print(f"Avg price paid ($/kWh): {df['rev_before'].sum() / df['hourly_kwh'].sum():.4f}"
      f" -> {df['rev_after'].sum() / df['new_kwh'].sum():.4f}")

df["state_name"] = df["state"].map(STATE_NAMES)
summary = df.groupby("state_name").agg(
    hours=("state", "size"),
    energy_before=("hourly_kwh", "sum"),
    energy_after=("new_kwh", "sum"),
    avg_price_before=("base_tou_price", "mean"),
    avg_price_after=("dynamic_price", "mean"),
).round(3)
summary["energy_change_pct"] = ((summary["energy_after"] / summary["energy_before"] - 1) * 100).round(1)
print("\nBy regime:")
print(summary)

df.to_csv(OUTPUT_FILE, index=False)
print("\nSaved:", OUTPUT_FILE)