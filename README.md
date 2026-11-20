# Dynamic-price-optimization-for-electric-vehicle-EV-technology
This project aims to develop an intelligent dynamic pricing system for electric vehicle (EV) technology that optimizes prices based on market demand, battery costs, charging infrastructure, competitor pricing, customer preferences, and seasonal trends

# HMM-Based Operational State Modeling

### 4.1 Formal Mathematical Formulation

A Gaussian Hidden Markov Model (HMM) is used to identify the operating condition of EV charging infrastructure.

The model uses two input features:

- `occupancy_rate`: Indicates how much of the charging infrastructure is occupied.
- `transformer_load_ratio`: Indicates how heavily the electrical transformer is being used.

At every time step, the model identifies one of three hidden operating states:

| State | Operating Condition | Meaning |
|---|---|---|
| S1 | Renewable Surplus / Low Load | Low charging station occupancy and low transformer load. |
| S2 | Nominal Operating Regime | Normal and stable charging activity. |
| S3 | Peak Grid Stress | High charging activity and increased transformer load. |

**Three important HMM parameters:**

1. **Initial State Probability (π):** Describes the probability of which operating state occurs at the beginning.
2. **Transition Probability Matrix (A):** Describes the probability of moving from one operating state to another over time.
3. **Emission Probability (B):** Describes how likely the observed input values are under each operating state.

The model uses Gaussian distributions to represent the input values associated with each state.

### 4.2 Model Training Using Baum-Welch Algorithm

A Gaussian HMM with three hidden states was created using Python's `hmmlearn` library.

```python
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)
```

The model learns its parameters using the Baum-Welch algorithm, which follows the Expectation-Maximization (EM) approach.

- **E-Step:** Estimates the probability of each hidden state based on the observed data.
- **M-Step:** Updates the model parameters to improve the probability of observing the given data.
- **Convergence Monitoring:** Tracks changes in log-likelihood during training to check whether the model is learning and approaching convergence.

### 4.3 Model Selection and State Count Evaluation

To evaluate the number of hidden states, four Gaussian HMM configurations were compared using 2, 3, 4 and 5 states.

Two evaluation metrics were used:

- **AIC (Akaike Information Criterion):** Evaluates model fit while penalizing unnecessary complexity.
- **BIC (Bayesian Information Criterion):** Evaluates model fit while applying a penalty based on the number of parameters and observations.

Lower AIC and BIC values generally indicate a better balance between model fit and complexity.

Along with these metrics, domain interpretability was considered to determine whether the identified states represent meaningful EV charging conditions.

The three-state configuration represents Renewable Surplus, Nominal Operation and Peak Grid Stress.
