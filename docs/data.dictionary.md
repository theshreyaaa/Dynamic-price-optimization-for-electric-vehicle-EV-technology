**session_duration_minutes:** Total plug-in duration (minutes).
**active_charging_minutes:** Time vehicle actually drew current (minutes).

## HMM Model

This document describes the input features, model parameters and operational states used in the Gaussian Hidden Markov Model (HMM).

## 1. Input Features

The HMM uses two input features to identify EV charging operating conditions.

| Column Name | Data Type | Description |
|---|---|---|
| `occupancy_rate` | Float | Represents the proportion of charging stations currently occupied. |
| `transformer_load_ratio` | Float | Represents the transformer load compared to its reference capacity. |

These two features are used together to identify the operating state of the charging infrastructure.

## 2. Hidden Operational States

The model identifies three operating conditions.

| State | State Name | Description |
|---|---|---|
| S1 | Renewable Surplus / Low Load | Low charging station occupancy and low transformer loading. |
| S2 | Nominal Operating Regime | Normal and stable charging conditions. |
| S3 | Peak Grid Stress | High charging activity and increased transformer loading. |

## 3. HMM Model Parameters

| Parameter | Description |
|---|---|
| `n_components` | Number of hidden states used in the model (3). |
| `covariance_type` | Type of covariance matrix used (`full`). |
| `n_iter` | Maximum number of training iterations (200). |
| `random_state` | Random seed used for reproducibility (42). |
| `startprob_` | Initial probability of each hidden state. |
| `transmat_` | Probability of transitioning between hidden states. |
| `means_` | Average input feature values for each hidden state. |
| `covars_` | Covariance matrices describing the variation of input features in each state. |

## 4. Model Evaluation Terms

| Term | Description |
|---|---|
| `log_likelihood` | Measures how well the model explains the observed data. |
| `AIC` | Model evaluation metric that considers fit and model complexity. |
| `BIC` | Model evaluation metric that considers fit, complexity and dataset size. |
| `n_components` | Number of hidden states tested (2, 3, 4 and 5). |

## 5. Generated Model Outputs

| Output | Description |
|---|---|
| Initial State Probabilities | Starting probability of each operating state. |
| Transition Probability Matrix | Probability of moving from one state to another. |
| Emission Means | Average occupancy and transformer load values for each state. |
| Emission Covariances | Variation and relationship between the two input features for each state. |
| Transition Counts | Number of observed transitions between states. |
| Transition Probabilities | Normalized probabilities calculated from transition counts. |

**Note:** The input features are numerical values, while the three operational states represent the operating conditions identified by the model.
