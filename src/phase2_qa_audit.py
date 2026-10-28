"""Phase 3 QA & System Integration Audit Script

Verifies mathematical consistency and physical plausibility of HMM outputs.
"""

import numpy as np
import pandas as pd


def run_phase3_audit():
  print("=" * 60)
  print("RUNNING PHASE 3 SYSTEM INTEGRATION & QA AUDIT")
  print("=" * 60)

  # 1. Load the delivered state dataset
  csv_path = "hourly_features_with_states.csv"
  df = pd.read_csv(csv_path)
  print(f"[PASS] Successfully loaded '{csv_path}' with shape {df.shape}")

  # 2. Check row count and temporal continuity
  assert (
      len(df) == 8760
  ), f"Row count error: Expected 8760, found {len(df)}"  # 365 days * 24 hrs
  t = pd.to_datetime(df["timestamp"])
  assert (
      t.is_monotonic_increasing
  ), "Temporal error: Timestamps are not monotonically increasing."
  expected_diff = pd.Timedelta(hours=1)
  assert (
      t.diff().dropna() == expected_diff
  ).all(), "Continuity error: Gaps or duplicate hours found."
  print("[PASS] Exactly 8,760 continuous hours verified with zero gaps.")

  # 3. Check for missing values
  null_count = df.isna().sum().sum()
  assert null_count == 0, f"Null value error: Found {null_count} NaNs."
  print("[PASS] Zero missing values across all columns.")

  # 4. Check physical boundaries
  assert df["occupancy_rate"].between(0.0, 1.0).all(), (
      "Boundary error: Occupancy rate outside [0, 1]."
  )
  assert (df["transformer_load_ratio"] >= 0.0).all(), (
      "Boundary error: Negative transformer load detected."
  )
  print("[PASS] Physical bounds verified: OR in [0, 1], TLR >= 0.")

  # 5. Check Monotonic State Ordering (Label Switching Check)
  means = df.groupby("state")["transformer_load_ratio"].mean().to_dict()
  print("\nMean Transformer Load by State:")
  for s, m in means.items():
    print(f"  - State {s}: {m*100:.3f}%")

  assert means[0] < means[1] < means[2], (
      "LABEL SWITCHING DETECTED: States are not strictly sorted by load!"
  )
  print(
      "[PASS] Deterministic state ordering verified (0: Surplus < 1: Nominal <"
      " 2: Peak Stress)."
  )

  # 6. Check Empirical Transition Matrix Stochasticity & Inertia
  states = df["state"].values
  T_mat = np.zeros((3, 3))
  for i in range(len(states) - 1):
    T_mat[states[i], states[i + 1]] += 1
  P_emp = T_mat / T_mat.sum(axis=1, keepdims=True)

  row_sums = P_emp.sum(axis=1)
  np.testing.assert_allclose(
      row_sums, 1.0, atol=1e-5, err_msg="Matrix rows do not sum to 1."
  )
  print("\nEmpirical Transition Probability Matrix:")
  print(np.round(P_emp, 3))

  # Diagonal persistence check
  for k in range(3):
    assert P_emp[k, k] > 0.65, (
        f"Inertia error: State {k} persistence is too low ({P_emp[k, k]:.3f})."
    )
  print(
      f"[PASS] State persistence verified: Diagonal = {np.round(np.diag(P_emp), 3)}"
  )

  # Forbidden transitions check
  assert P_emp[2, 0] <= 0.02, (
      f"Physical anomaly: Direct transition from Peak to Surplus is too high ({P_emp[2, 0]:.3f})"
  )
  print(
      f"[PASS] Causal transition confirmed: P(Peak -> Surplus) = {P_emp[2, 0]:.3f}"
  )

  print("\n" + "=" * 60)
  print("PHASE 3 QA AUDIT COMPLETE: ALL ASSERTIONS PASSED.")
  print("DATASET CERTIFIED READY FOR PHASE 4 PRICING OPTIMIZATION.")
  print("=" * 60)


if __name__ == "__main__":
  run_phase3_audit()