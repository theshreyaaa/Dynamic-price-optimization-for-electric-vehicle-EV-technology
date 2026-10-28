# Project Design Decisions & Technical Q&A

This document records architectural decisions, engineering edge cases, and clarifications resolved during development.

---

## Phase 1: Repository Initialisation and Appropriate Data Selection
### Q1: 
* **Raised by:** 
* **Resolution:** 

---

## Phase 2: Data Cleaning & Preprocessing
Raised by:Ankush

### Q1: What is the Caltech ACN dataset, and what physical facility does it represent?

* **Answer:** It records real-world EV charging sessions from a parking garage at the California Institute of Technology (Pasadena, California) across the entire year of 2019. It represents a **workplace charging facility** containing exactly **54 Level 2 AC charging stalls**, where employees and students plug in their vehicles for extended stays during work and class shifts.

---

### Q2: Why was raw data in two JSON files, and how did you verify no duplication occurred?

* **Answer:** The full 2019 dataset exceeded the Caltech API single-export size limit and was downloaded in two halves: File 1 (7,440 sessions) and File 2 (3,177 sessions), totaling 10,617 raw rows. After concatenating both files, we ran `df['sessionID'].duplicated().sum()`, which returned **0**, verifying that every session has a unique primary key.

---

### Q3: Why is strict chronological sorting (`connectionTime`) necessary?

* **Answer:** Machine learning models and hourly time-series resampling require time to move forward monotonically. Chronological sorting prevents **temporal data leakage** (using future charging events to predict past states) and guarantees causal consistency when aggregating data into hourly bins.

---

### Q4: Why convert timestamps to California local time, and why is Indian Standard Time (IST) completely invalid?

* **Answer:**
* **California Local Time (`America/Los_Angeles`):** Essential because commuter charging routines and Southern California Edison (SCE) Time-of-Use electricity tariffs strictly operate on local California wall-clock time.

* **Why Not IST:** India is 12.5 to 13.5 hours ahead of California. Converting to IST inverts day and night—turning an 8:30 AM California morning office arrival into a 10:00 PM late-night event, completely corrupting diurnal behavior patterns.

---

### Q5: What is the Dual-Timezone architecture, and why not use Pacific Time for everything?

* **Answer:**
* **The Problem:** California observes Daylight Saving Time (DST). If you build an annual timeline directly in Pacific Time, March skips an hour (8,759 hours) and November duplicates an hour (8,761 hours), breaking continuous time-series models.
* **The Solution:** We keep the continuous mathematical index in **UTC** (unaffected by DST, guaranteeing exactly 8,760 uninterrupted hours for 2019), and project to **Pacific Time** only when extracting behavioral features (`local_hour`, `is_weekend`) and SCE baseline tariffs.

---

### Q6: Why must total session duration be split into Active Charging and Idle Duration?

* **Answer:**
* **Session Duration:** Total time a car physically occupies a stall (`disconnectTime - connectionTime`). Measures parking congestion.


* **Active Duration:** Time electricity is flowing into the battery (`doneChargingTime - connectionTime`). Measures grid power demand.


* **Idle Duration:** Time a fully charged car sits plugged in (`disconnectTime - doneChargingTime`). Measures stall blocking.


* **Why Separate:** In workplace lots, cars finish charging in 2 to 3 hours but remain parked for 8 hours. Failing to separate them causes **phantom load errors**, where the model falsely assumes the vehicle is pulling electrical power for the entire 8-hour workday.

---

### Q7: Why clip active duration to $\ge 3.0$ minutes and cap charging power at $\le 25\text{ kW}$?

* **Answer:**
* **Clipping Duration:** Implied power is calculated as $\text{Energy} / \text{ActiveHours}$. An instant 10-second plug-in would cause division by near-zero, producing an impossible mathematical power spike ($> 300\text{ kW}$) on a standard $7\text{ kW}$ Level 2 charger.
* **Capping at $25\text{ kW}$:** Caltech exclusively uses J1772 Level 2 AC chargers, which have a physical hardware ceiling of $\sim 19.2\text{--}22\text{ kW}$. Any reading above $25\text{ kW}$ is sensor current drift or telemetry corruption.

---

### Q8: How did you handle missing or inverted `doneChargingTime` records?

* **Answer:** In records where `doneChargingTime` was missing ($NaN$) or corrupted (logged before `connectionTime` or after `disconnectTime`), it was imputed to equal `disconnectTime`. This conservatively assumes the vehicle drew trickle power until departure rather than discarding a valid physical session.

---

### Q9: Why are there 1,532 missing values in `userID`, and why is dropping them (`dropna()`) a mistake?

* **Answer:** Caltech allows anonymous guest charging for drivers who pay with a credit card without registering on the mobile app. While user profile fields are empty, the physical energy meters and connection timestamps are 100% complete. Because our project optimizes site-level transformer load rather than individual user recommendations, dropping these rows would ignore 14.5% of real grid electricity demand.

---

### Q10: Why filter out sessions delivering less than 0.5 kWh?

* **Answer:** Sessions under $0.5\text{ kWh}$ represent aborted communication handshakes, declined payment cards, or drivers who plugged in and immediately disconnected upon seeing a faulty stall. They represent telemetry noise rather than genuine battery charging cycles.

---

### Q11: Why drop sessions longer than 24 hours instead of capping them?

* **Answer:** The raw data contained vehicles parked for up to 9 days (214 hours) over university holidays. These are long-term vehicle storage events, not active workplace charging sessions. Leaving a 9-day car in the data marks a stall as 100% continuously occupied for over a week, severely distorting daily commuter occupancy profiles and Markov state transitions.

---

### Q12: Why does Z-Score fail on raw EV data, and why use IQR for outliers but Z-Score for standardization?

* **Answer:**
* **Why Z-Score Fails on Raw Data:** Z-scores assume a symmetric, normal bell curve. Raw EV charging data is heavily right-skewed with extreme long tails (e.g., 200-hour sessions). Extreme outliers inflate the mean ($\mu$) and standard deviation ($\sigma$), shrinking calculated Z-scores and allowing corrupt records to slip past detection.

* **The Two-Stage Solution:** We used non-parametric **IQR** (median and quartiles) for initial outlier removal because it is resistant to extreme tail distortion. Once extreme noise was removed, **Z-score transformation** was applied to standardize clean features onto a zero-mean, unit-variance scale for machine learning.

---

### Q13: What is the exact accounting of rows dropped during Phase 2?

* **Answer:** We ingested **10,617 raw rows** and dropped **78 invalid rows** (77 multi-day storage outliers $> 24$ hours, 1 sensor power spike $> 25\text{ kW}$, and handshake drops $< 0.5\text{ kWh}$), leaving exactly **10,539 clean rows** (a 99.26% data retention rate).

---

### Q14: What sanity checks prove that Phase 2 was completed correctly before Phase 3?

* **Answer:**
1. **Zero Duplicates:** `df['sessionID'].duplicated().sum() == 0`.
2. **Operational Completeness:** Zero null values across timestamps, durations, and energy columns.
3. **Temporal Validity:** 100% of rows satisfy $\text{connectionTime} \le \text{doneChargingTime} \le \text{disconnectTime}$.
4. **Duration Additivity:** $\text{Total Session Duration} = \text{Active Duration} + \text{Idle Duration}$ holds across all 10,539 rows.
5. **Physical Realism:** All sessions are under 24 hours, energy delivery is $\ge 0.5\text{ kWh}$, and maximum charging rate ($20.13\text{ kW}$) respects Level 2 charger limits.

---

### Q15: How does Phase 2 directly connect to Phase 3 and Phase 4?

* **Answer:** Phase 2 cleans session anomalies and separates active power from idle parking. This allows Phase 3 to resample sessions into an 8,760-hour continuous timeline to calculate **Stall Occupancy Rate ($OR_t$)** and **Transformer Load Ratio ($TLR_t$)**, which train the Gaussian HMM to classify grid stress regimes for Phase 4 dynamic pricing.

---

## Phase 3: Time-Series Resampling & HMM Modeling
# EV Charging Optimization — Project Q&A

### Q1: How can we convert charging session data into hourly data?
**Raised by:** Anshika Kumari
**Solution:** I created an hourly timeline to convert the charging session data into a format suitable for hourly analysis. This helps us understand charging activity hour by hour. For a complete non-leap year, the timeline contains 8,760 hours.

### Q2: How can we convert UTC timestamps into local time for charging demand analysis?
**Raised by:** Anshika Kumari
**Solution:** I converted the UTC timestamps into the local timezone (`America/Los_Angeles`). This helps identify the correct local hour, day of the week, and whether the day is a weekday or weekend, which are important for analysing charging demand.

### Q3: How can we calculate the transformer load ratio for each hour?
**Raised by:** Anshika Kumari
**Solution:** I estimated the charging load for each hour by calculating how long vehicles were actively charging during that hour and using their charging rates to estimate energy consumption. I then divided the estimated hourly load by the assumed transformer capacity of 150 kW to calculate the transformer load ratio.

### Q4: How can we assign time-of-use (TOU) prices to each hour?
**Raised by:** Anshika Kumari
**Solution:** I assigned electricity prices to each hourly record based on the local time and whether the day was a weekday or weekend. This helps represent how electricity prices may vary throughout the day. The price values used in the project are predefined assumptions and should be verified against the actual electricity tariff before being treated as official prices.

### Q5: What is the "Label Switching Problem", why is it dangerous for Phase 4, and how did you mathematically solve it?
**Raised by:** Palak Kumari
**Solution:** The Problem & Danger: Gaussian HMM is an unsupervised algorithm that assigns random numeric state IDs (0, 1, 2). Without sorting, State 0 might mean "Peak Stress" on one machine and "Off-Peak" on another, causing Phase 4 to mistakenly issue cheap discounts during severe afternoon transformer overloads.   
The Solution: I enforced a deterministic, rank-based sort indexed strictly on average transformer load ratio:   Pythonorder = np.argsort(model.means_[:, 1])  # Sort indices by load[cite: 5]
rank = np.empty_like(order)
rank[order] = np.arange(3)  # Invert mapping[cite: 5]
named_states = rank[raw_states]  # 0: Surplus, 1: Nominal, 2: Peak Stress[cite: 5]

---

### Q6: Why build an automated QA audit script (`assert`) instead of using Excel, and what 6 checks does it enforce?
**Raised by:** Palak Kumari
**Solution:** Manual inspection of 8,760 rows cannot catch subtle silent bugs like missing hours, negative loads, or floating-point drift. `src/phase3_qa_audit.py` acts as an automated programmatic barrier that halts execution unless all 6 conditions pass:
  1. **Row Count & Continuity:** Exactly 8,760 continuous hourly intervals with zero missing or duplicate timestamps.
  2. **Data Completeness:** Exactly zero $NaN$ values across all 12 columns.
  3. **Physical Bounds:** Occupancy Rate strictly in $[0.0, 1.0]$ and Transformer Load $\ge 0.0$.
  4. **Monotonic State Order:** Strict progression of mean loads ($\mu_0 < \mu_1 < \mu_2$), proving label switching is resolved[cite: 5].
  5. **Matrix Stochasticity:** Every row of the empirical transition matrix sums to $1.0 \pm 10^{-5}$.
  6. **Inertia & Causal Flow:** Diagonal persistence values exceed $0.65$[cite: 5], and forbidden transitions ($P_{20}$) are near $0.0$.

---

### Q7: Why must transition matrix rows sum to 1.0, what floating-point problem occurred, and how did you solve it?
**Raised by:** Palak Kumari
**Solution:**
  * **Mathematical Rule:** A station in state $i$ at hour $t$ must transition to *some* state $j \in \{0, 1, 2\}$ at hour $t+1$; total probability must equal certainty ($\sum_j P_{ij} = 1.0$).
  * **The Problem:** Standard computer floating-point math produced row sums like $0.9999$, causing strict equality checks (`sum == 1.0`) to fail.
  * **The Solution:** Used NumPy's tolerance assertion: `np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)` to verify stochasticity within standard IEEE floating-point precision ($0.00001$).

---

### Q8: What is Regime Persistence, what problem occurs if it is weak, and how did you verify it?
**Raised by:** Palak Kumari
**Solution:**
  * **The Concept & Risk:** Persistence is the probability of staying in the same state during the next hour ($P_{ii}$)[cite: 5]. Weak persistence causes erratic price flickering (e.g., jumping between Peak and Surplus every hour), frustrating drivers and destabilizing grid controls.
  * **The Solution:** Audited the diagonal elements of the transition matrix against a $> 0.65$ threshold[cite: 5]. The model achieved empirical diagonal values of **0.857**, **0.821**, and **0.907**, confirming high real-world grid inertia[cite: 5].

---

### Q9: What is a "Forbidden Transition", what physical error could occur, and how did you test it?
**Raised by:** Palak Kumari
**Solution:**
  * **The Physical Risk:** An unsupervised model could theoretically allow a direct transition from maximum grid overload (State 2) to an empty garage (State 0) in a single hour, which is physically impossible in workplace charging.
  * **The Solution:** Added an explicit assertion: `assert P_emp[2, 0] <= 0.02`. The audit proved $P_{20} = 0.000$, confirming the facility always steps down gradually through Nominal State 1 before returning to Surplus State 0[cite: 5].

---

### Q10: In Figure 1, why do Transformer Load and Stall Occupancy peak at different hours?
**Raised by:** Palak Kumari
**Solution:** Transformer load peaks at **10:00 AM** (morning arrivals plugging in depleted batteries and drawing high active power), while occupancy remains elevated until **3:00 PM**. This is not an error—it is the visual proof of **idle stall blocking**, confirming that fully charged cars remain parked while owners finish work shifts, which justifies separating active power from parking duration.

---

### Q11: What problem occurs when plotting all 8,760 hours, and how did Figure 3 solve it?
**Raised by:** Palak Kumari
**Solution:**
  * **The Problem:** Plotting 8,760 hours on one chart compresses the curve into an unreadable solid black smudge.
  * **The Solution:** Isolated a representative 168-hour continuous window (June 3–9, 2019). This provides the visual clarity needed to trace hourly power draws against the 150 kW capacity line while displaying background regime shading (`axvspan`) for Surplus (Green), Nominal (Yellow), and Peak Stress (Red)[cite: 5].

---

### Q12: What practical file-handling bug crashed the visualization script, and how did you solve it?
**Raised by:** Palak Kumari
**Solution:**
  * **The Bug:** Running `src/phase3_visualizations.py` on a freshly cloned repository triggered a `FileNotFoundError` because the `figures/` directory did not exist yet.
  * **The Solution:** Added defensive directory creation at the script entry point: `os.makedirs("figures", exist_ok=True)`. Python now creates the folder automatically if missing.

---

### Q13: How do your Phase 3 deliverables directly feed into Phase 4 Dynamic Pricing?
**Raised by:** Palak Kumari
**Solution:**
  1. **Bug-Free Dataset:** The QA audit certifies that `hourly_features_with_states.csv` is mathematically and physically validated[cite: 5].
  2. **Reliable Price Triggers:** Sorted states provide deterministic pricing actions (State 0 = surplus discount, State 1 = baseline SCE rate, State 2 = peak surcharge)[cite: 5].
  3. **Risk Adjustments:** The verified transition matrix provides transition probabilities ($P_{ij}$) used by Phase 4 optimization algorithms to anticipate upcoming grid stress.