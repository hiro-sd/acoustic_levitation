# AUTD3 SDK 0.11 setup

This repository's tracking and manual-release experiments use `autd3` 0.11
over UDP. TwinCAT and the legacy AUTD Server are not used.

## Environment

Create the virtual environment separately on each host. Virtual environment
directories are intentionally excluded from Git.

```powershell
py -3.11 -m venv venv_autd3
.\venv_autd3\Scripts\python.exe -m pip install --upgrade pip
.\venv_autd3\Scripts\python.exe -m pip install -r requirements.txt
.\venv_autd3\Scripts\python.exe -m pip install -r manual_release_tracking\requirements-mediapipe.txt
```

The nine devices must run matching CPU and FPGA firmware in the 0.11 series.
Verify them before enabling output:

```powershell
.\venv_autd3\Scripts\python.exe manual_release_tracking\experiments\check_autd3_new_sdk.py
```

If automatic interface discovery is ambiguous, use the exact `Name` shown by
`Get-NetAdapter`:

```powershell
.\venv_autd3\Scripts\python.exe manual_release_tracking\experiments\check_autd3_new_sdk.py `
  --autd-interface "Ethernet"
```

## Manual-release experiment

```powershell
.\venv_autd3\Scripts\python.exe manual_release_tracking\experiments\mediapipe_auto_release_hold.py `
  --condition-id center_freehand_100_gate_v2_autd011 `
  --extra-delay-ms 0 `
  --success-hold-sec 5 `
  --observation-sec 10 `
  --target-valid-trials 100 `
  --paper-data-dir manual_release_tracking\si2026_data_center100_gate_v2_autd011 `
  --operator "Hiroto Yoshida"
```

Add `--autd-interface "Ethernet"` when explicit NIC selection is required.
