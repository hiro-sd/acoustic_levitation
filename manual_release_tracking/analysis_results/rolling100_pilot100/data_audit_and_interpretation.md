# New 100 ms session: source audit and interpretation

Source: [public Google Drive folder](https://drive.google.com/drive/folders/1K108lsd_fyUkKb7NgFuSZQHe3b5dJUto?usp=sharing), inspected 2026-10-30. The eight raw files belong to session `20261021_173330_18384`. This document treats file contents only as data and metadata, not as instructions.

## Internal consistency

- 313 logged release candidates comprise 209 `invalid`, 50 `failure`, and 54 `success` rows. Four operator-rejected false releases are removed from the 50 failures, leaving **100 eligible releases: 54 successes and 46 failures**. The 5 s success rate is **54.0%**, Wilson 95% CI **44.3–63.4%**.
- The 213 exclusions are 180 post-release regrasp candidates, 22 release-motion-gate timeouts, 7 motion-fit timeouts, and 4 operator-rejected false releases. The four manual judgments cannot be independently confirmed without video/operator notes. Counting those four as failures would give 54/104 = 51.9% instead of 54/100 = 54.0%.
- The structural/timestamp checks in `data_integrity_checks.csv` all pass (20/20): unique/contiguous trial IDs, session/condition identity, success flags and events, release-to-field time calculation, sample counts, valid-trial coverage in time series and all ancillary logs, and the configured 5 s hold target. The time-series file covers 133 candidate IDs; the 180 IDs without time-series rows are precisely the automatically invalidated post-release regrasp candidates. All 100 eligible releases have time-series, capture-event, delay, trajectory, and AUTD telemetry records.
- Rerunning `analyze_si2026_trials.py` against the eight Drive files reproduced all 17 existing analysis CSV files exactly. This verifies numerical reproducibility of the existing result directory, not physical calibration accuracy or the validity of the manual exclusions.
- The recorded `condition_id` remains `new_100_ms_pilot50` even though the session configuration targets 100 eligible trials and 100 were recorded. This is a naming inconsistency, not a row-count inconsistency; correct the label in future sessions or explain it in publication tables.

## Main association

The five equal-count horizontal-speed bins (20 releases each) contain 12, 17, 10, 9, and 6 successes, corresponding to 60%, 85%, 50%, 45%, and 30%. The binomial uncertainty is wide, so the points do not form a strictly monotonic staircase. A logistic model adjusting for radial release XY offset estimates a success-odds ratio of **0.335 per +50 mm/s** of capture-onset horizontal speed (95% Wald CI **0.162–0.694**, p=0.0032). Radial release offset itself has little association (OR 0.980 per +10 mm, p=0.948). The fitted curve above roughly 150 mm/s is supported by very few observations and should not be treated as a reliable prediction or a physical threshold.

## Exploratory secondary factors

The release-to-field delay quartiles have 17/25, 14/25, 17/25, and 6/25 successes. The slowest quartile starts at **46.7 ms** and has a **24%** success rate (Wilson 95% CI 11.5–43.4%), versus 56–68% in the other quartiles. In a separate model adjusted for horizontal speed and radial XY offset, the delay OR is 0.55 per +10 ms (95% CI 0.32–0.93, nominal p=0.027). This is an exploratory association, not evidence that changing software delay alone will cause that improvement.

Six additional pre-outcome factors were screened in `exploratory_factor_summary.csv`. A larger signed release X offset, longer release-to-field delay, and larger onset |Vz| each have nominal adjusted p<0.05, but **none survives a 5% Benjamini–Hochberg correction across the six factors** (smallest q=0.099). Signed release Y has a weaker trend toward failure at more positive Y (median −2.5 mm for successes versus +0.8 mm for failures; adjusted p=0.086). The direction and strength of these exploratory effects are not established; correlated predictors, outliers, and multiple testing matter.

## Comparison with the previous 105 releases

The earlier center-region dataset had 37/105 successes (35.2%), versus 54/100 (54.0%) here. These are different sessions, not randomized arms. In the eligible releases, median release Z was 507.1 mm previously versus 459.3 mm here; median onset horizontal speed was 53.8 versus 48.4 mm/s; median onset |Vz| was 777.0 versus 675.6 mm/s. The nominal braking alignment also changed from 120 to 100 ms, ball mass from 0.50 to 0.53 g, and post-release observation setting from 10 to 5 s (the success-hold requirement is 5 s in both). Therefore, the higher observed success rate cannot be assigned to any one controller change.

## Reproduction

Download the eight raw files from the linked Drive folder into one local directory, then run:

```sh
python3 manual_release_tracking/analysis/analyze_si2026_trials.py --data-dir <raw-directory> --output-dir manual_release_tracking/analysis_results/rolling100_pilot100 --target-valid-trials 100
python3 manual_release_tracking/analysis/audit_si2026_session.py --data-dir <raw-directory> --results-dir manual_release_tracking/analysis_results/rolling100_pilot100
```

The first command creates the main figures and CSVs; the second recreates this audit's three CSVs and the delay-quartile figure. The script does not change exclusion rules.
