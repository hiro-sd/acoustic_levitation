# XY force characterization

This experiment measures the lateral acceleration produced by a circular STM
whose eight points have unequal dwell times. It does not use or modify the
automatic-release state machine.

Run from the repository root:

```bash
python tracking/experiments/xy_force_characterization.py
```

1. Place the sphere in the normal uniform circular field.
2. Press `ENTER` to start the ordinary PID hold.
3. Wait until the displayed velocity becomes small.
4. Press `0` for a uniform control pulse, or `1` to `4` to bias the `+X`,
   `-X`, `+Y`, or `-Y` side of the ring.
5. Press `B` between trials to cycle the bias level through 0.10, 0.20, and
   0.30.

The experiment requests the uniform STM after 80 ms and records the actual
sender completion, so the measured field duration also includes the AUTD send
latency. It requests an earlier stop if XY displacement exceeds
10 mm, Z drops more than 10 mm, `|vz|` exceeds 150 mm/s, or stereo measurement
is lost. During a pulse, XY feedback is not applied and the STM center is held
at its value immediately before the pulse. Z feedback remains active. After the
pulse, the original uniform STM and ordinary XYZ PID are restored for a 2 s
recovery interval.

The normal 8-point STM runs at 100 Hz (800 focus slots/s). During a weighted
64-slot pulse, its complete-cycle frequency is automatically reduced to
12.5 Hz so that the focus-slot update rate remains 800 slots/s. This keeps the
existing Silencer interpolation valid; the ordinary application and normal
unweighted STM remain at their existing frequency. Repeated presentations are
distributed across circular passes, so the zero-bias 64-slot control sequence
is the ordinary 8-point circle repeated eight times rather than eight long
consecutive holds at each point.

Outputs:

- `tracking/xy_force_frame_log.csv`: synchronized per-frame measurements
- `tracking/xy_force_trial_summary.csv`: fitted acceleration and force for each
  pulse

`BIAS_+X` means that the points on the `+X` side receive more STM dwell slots.
It does not assume that the actual sphere force is `+X`; determine the force
direction from the measured acceleration.
