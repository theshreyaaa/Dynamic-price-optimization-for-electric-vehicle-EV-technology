"""Phase 3 Visualization Generator

Produces publication-grade diagnostic plots for project report and viva.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

df = pd.read_csv("hourly_features_with_states.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==============================================================================
# FIGURE 1: 24-Hour Diurnal Profile & State Probability
# ==============================================================================
print("Generating Figure 1: 24-Hour Diurnal Dynamics...")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
)

# Subplot 1: Occupancy vs Load Ratio
hourly_means = (
    df.groupby("local_hour")[["occupancy_rate", "transformer_load_ratio"]]
    .mean()
    .reset_index()
)
ax1.plot(
    hourly_means["local_hour"],
    hourly_means["occupancy_rate"] * 100,
    color="#1f77b4",
    linewidth=2.5,
    marker="o",
    label="Occupancy Rate (% of 54 stalls)",
)
ax1.plot(
    hourly_means["local_hour"],
    hourly_means["transformer_load_ratio"] * 100,
    color="#d62728",
    linewidth=2.5,
    marker="s",
    label="Transformer Load (% of 150 kW limit)",
)
ax1.set_ylabel("Utilization (%)", fontweight="bold")
ax1.set_title(
    "24-Hour Campus Diurnal EV Demand Profile (Caltech 2019)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left", frameon=True)

# Subplot 2: Probability of HMM Regimes by Hour
state_probs = pd.crosstab(
    df["local_hour"], df["state_name"], normalize="index"
)
colors = {"Surplus": "#2ca02c", "Nominal": "#ff7f0e", "Peak Stress": "#d62728"}
bottom = np.zeros(24)

for s_name in ["Surplus", "Nominal", "Peak Stress"]:
  if s_name in state_probs.columns:
    vals = state_probs[s_name].values
    ax2.bar(
        range(24),
        vals,
        bottom=bottom,
        label=s_name,
        color=colors[s_name],
        alpha=0.85,
        width=0.8,
    )
    bottom += vals

ax2.set_xlabel("Local Pacific Time (Hour of Day)", fontweight="bold")
ax2.set_ylabel("State Probability", fontweight="bold")
ax2.set_xticks(range(0, 24))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
ax2.set_ylim(0, 1.0)
ax2.grid(axis="y", linestyle="--", alpha=0.6)
ax2.legend(loc="upper right", ncol=3, frameon=True)

plt.tight_layout()
fig.savefig("figures/fig1_diurnal_regime_profile.png", dpi=300)
plt.close()
print("Saved: figures/fig1_diurnal_regime_profile.png")

# ==============================================================================
# FIGURE 2: Sorted Transition Probability Matrix Heatmap
# ==============================================================================
print("Generating Figure 2: Transition Probability Heatmap...")
states = df["state"].values
T_mat = np.zeros((3, 3))
for i in range(len(states) - 1):
  T_mat[states[i], states[i + 1]] += 1
P_emp = T_mat / T_mat.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
labels = ["State 0\n(Surplus)", "State 1\n(Nominal)", "State 2\n(Peak Stress)"]

sns.heatmap(
    P_emp,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    cbar=True,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 13, "weight": "bold"},
    ax=ax,
)

ax.set_title(
    "Empirical HMM Regime Transition Matrix",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Transition to State ($S_{t}$)", fontweight="bold", labelpad=8)
ax.set_ylabel("Current State ($S_{t-1}$)", fontweight="bold", labelpad=8)

plt.tight_layout()
fig.savefig("figures/fig2_transition_matrix_heatmap.png", dpi=300)
plt.close()
print("Saved: figures/fig2_transition_matrix_heatmap.png")

# ==============================================================================
# FIGURE 3: 7-Day Continuous Trajectory with Shaded Regimes
# ==============================================================================
print("Generating Figure 3: 7-Day Continuous Trajectory...")
# Sample one clean representative week: First full week of June 2019 (June 3 to June 9)
week_mask = (df["timestamp"] >= "2019-06-03 00:00:00+00:00") & (
    df["timestamp"] <= "2019-06-09 23:00:00+00:00"
)
week_df = df[week_mask].copy().reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 5.5))

# Plot active transformer load
ax.plot(
    week_df.index,
    week_df["transformer_load_ratio"] * 100,
    color="#111111",
    linewidth=2.0,
    label="Transformer Load Ratio (%)",
    zorder=3,
)
ax.axhline(
    y=100.0,
    color="darkred",
    linestyle=":",
    linewidth=1.5,
    label="Rated Transformer Capacity (150 kW)",
    zorder=2,
)

# Background regime shading
for i in range(len(week_df) - 1):
  st = week_df.loc[i, "state"]
  c = (
      "#d4edda"
      if st == 0
      else ("#fff3cd" if st == 1 else "#f8d7da")
  )  # soft green, yellow, red
  ax.axvspan(i, i + 1, color=c, alpha=0.85, zorder=1)

# Custom legend proxies for regimes
import matplotlib.patches as mpatches

patch_surplus = mpatches.Patch(color="#d4edda", label="State 0: Surplus")
patch_nominal = mpatches.Patch(color="#fff3cd", label="State 1: Nominal")
patch_peak = mpatches.Patch(color="#f8d7da", label="State 2: Peak Stress")

# Formatting x-ticks for the 7 days (24-hour steps)
day_ticks = np.arange(0, len(week_df), 24)
day_labels = [
    "Mon (Jun 3)",
    "Tue (Jun 4)",
    "Wed (Jun 5)",
    "Thu (Jun 6)",
    "Fri (Jun 7)",
    "Sat (Jun 8)",
    "Sun (Jun 9)",
]
ax.set_xticks(day_ticks)
ax.set_xticklabels(day_labels, fontweight="bold")
ax.set_xlim(0, len(week_df) - 1)
ax.set_ylim(0, max(week_df["transformer_load_ratio"] * 100) + 10)

ax.set_ylabel("Transformer Load (%)", fontweight="bold")
ax.set_title(
    "7-Day Continuous Demand Trajectory with Decoded HMM Regimes",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

handles, labels_list = ax.get_legend_handles_labels()
handles.extend([patch_surplus, patch_nominal, patch_peak])
ax.legend(handles=handles, loc="upper right", frameon=True, ncol=2)

plt.tight_layout()
fig.savefig("figures/fig3_regime_timeline_weekly.png", dpi=300)
plt.close()
print("Saved: figures/fig3_regime_timeline_weekly.png")
print("\nALL PHASE 3 VISUALIZATIONS GENERATED SUCCESSFULLY.")