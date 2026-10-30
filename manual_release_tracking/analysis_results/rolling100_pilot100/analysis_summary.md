# 100 ms rolling-force capture: 100-trial analysis

## Outcome

- 313 recorded attempts contained 100 analysis-eligible releases: 54 successes and 46 failures.
- The 5 s capture-and-hold success rate was **54.0%** (Wilson 95% CI: **44.3% to 63.4%**).
- The 213 excluded attempts comprised 180 post-release regrasp exclusions, 22 release-motion-gate timeouts, 7 post-release motion-fit timeouts, and 4 manually rejected false releases.
- Failure categories among the 46 eligible failures were: lower/upper Z-workspace departure 28 (60.9%), regrasp/contact 16 (34.8%), and tracking/estimation loss 2 (4.3%).

For context, the earlier fixed-deadline 100 ms pilot yielded 7/50 (14.0%), whereas the earlier rolling-force 120 ms pilot yielded 36/50 (72.0%). The present 54/100 result is descriptively between those rates. These sessions were not randomized concurrent conditions and differed in release kinematics and other settings, so the difference cannot be attributed to the force law or braking duration alone.

## Capture transient

- Median release-to-field delay: **42.2 ms** (IQR 39.2 to 46.6 ms).
- Median physical-release-gate time: **37.7 ms** (IQR 34.7 to 41.3 ms).
- Median initial 3-D prediction error: **4.47 mm** (IQR 3.73 to 5.30 mm).
- Median Z velocity at field onset: **-675.6 mm/s**.
- In successful trials, the release-aligned median Z velocity reached **-638.2 mm/s at 86 ms**, crossed through zero near 190 to 200 ms, and reached an upward peak of **+121.0 mm/s at 244 ms**.
- The successful-trial median field onset was 41.6 ms after release. The median transition to `LOCAL_HOLD` was 209.9 ms after release.
- The requested 100 ms braking plan did not end control at 100 ms. Across trials with a recorded braking end, the median transition occurred **166.9 ms after field onset**, because the controller waited for its measured-velocity exit condition after the minimum 100 ms alignment time.

## Braking-plan accuracy

- Median planned stopping distance: **33.8 mm**.
- Median measured stopping distance: **46.0 mm**.
- Median excess stopping distance: **11.0 mm**.
- Median braking-end velocity: **+61.5 mm/s**, showing a modest upward rebound at the transition to hold.
- All 100 eligible initial plans were logged as feasible and not force-saturated. The median maximum commanded intensity was 0.954.
- During braking, the requested intensity was continuous over 0.5 to 1.0; 104 distinct quantized output ratios were observed, so the controller was not selecting only the six calibration knots.

## Timing validation

- For the first capture command, median measurement-to-send-complete time was **11.02 ms** and median prediction horizon was **11.01 ms**.
- Median prediction shortfall was only **0.11 ms** (IQR -0.39 to +0.50 ms), indicating that the timestamp-dependent horizon and initial-command latency estimator were well aligned in typical trials.
- Median initial enqueue-to-send-complete time was **3.57 ms**. During all braking updates it was 3.07 ms; during `LOCAL_HOLD` it was 2.12 ms.
- Rare timing outliers remain: the maximum initial prediction shortfall was 7.97 ms and the maximum initial release-to-send-complete delay was 93.35 ms.

## Factors associated with outcome

- Horizontal speed at field onset was the clearest measured separator. Median speed was 40.7 mm/s for successes versus 57.9 mm/s for failures (Cliff's delta -0.324).
- Equal-count success rates decreased to 30% in the highest horizontal-speed bin (median 110.5 mm/s), compared with 60% to 85% in the two lowest bins.
- A logistic association model estimated an odds ratio of 0.335 per +50 mm/s horizontal speed after adjustment for release XY offset (95% CI 0.162 to 0.694; Wald p=0.0032). This is an association, not proof that horizontal speed alone causes failure.
- Release XY offset itself showed almost no success/failure separation (median 9.57 versus 9.90 mm; Cliff's delta -0.004).
- The slowest release-to-field quartile (>46.6 ms) had 6/25 successes (24%), whereas the other quartiles had 56% to 68%. This supports inspecting rare release-gate or scheduling delays, but should not be interpreted causally without a prespecified delay experiment.

## Recommended use

The controller is suitable for continued demonstrations and for mechanistic data collection, but **54% should not yet be described as a high or robust capture probability in a paper**. For the next controlled comparison, keep the recovered rolling-force law fixed and randomize only 100 ms versus 120 ms braking time. Record at least 100 valid releases per condition and stratify or constrain field-onset horizontal speed. The largest immediate engineering opportunity is reducing high-horizontal-speed releases or increasing lateral capture tolerance; typical timestamp prediction is no longer the dominant error source.

The session metadata still labels this dataset `new_100_ms_pilot50` although it contains 100 valid trials. Rename the condition for future sessions (for example, `rolling_force_100ms_n100`) so paper tables cannot confuse it with the earlier 50-trial pilot.
