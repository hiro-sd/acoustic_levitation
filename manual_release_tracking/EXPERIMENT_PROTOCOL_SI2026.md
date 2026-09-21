# SI2026 中心近傍の自由落下捕捉実験プロトコル

## 1. 目的

定位置治具は使用せず、指で把持した球体をAUTD中心付近で100回releaseする。
自然に生じるrelease位置と速度のばらつきを含め、次を評価する。

1. 捕捉成功時のrelease後0--10秒の球体軌道
2. 制動、整定、保持という制御機序と球体応答
3. 計画停止距離・停止位置と、制動終了時の実測値の対応
4. 実release 100試行の成功率と失敗挙動パターン
5. 成功時と失敗時のrelease位置、初期速度、遅延、予測誤差の比較

広いワークスペースの境界や、位置ごとの成功率は今回の主目的としない。

## 2. 試行の定義

目標は「解析対象となる実release 100試行」であり、全trial行を100行と数えない。

- 成功: release時刻から5秒以上、自動捕捉が停止せず保持された
- 失敗: 実際にreleaseしたが、5秒未満で計測ロスト、workspace離脱、再把持、
  または制御停止に至った
- 無効: 実releaseではない画像判定、運動ゲート不通過、Xキーによる誤判定取消、
  予備操作、または音場開始前の中断

成功判定後も10秒までは同じ場所で計測を続ける。`RETURN_TO_HOME`はrelease後
10秒より前には開始しない。成功と失敗の合計が100になるまで実施し、無効trialは
削除せず理由を残す。捕捉に失敗したこと自体を除外理由にしてはならない。

## 3. 誤release対策

MediaPipeのrelease候補だけでは音場を出さない。候補時のZを基準に、次の全条件が
成立した新規ステレオ推定が3回連続した場合だけ捕捉を開始する。

- `vz <= -100 mm/s`
- Zが2.5 mm以上下降
- 上記2条件が3推定連続

120 ms以内に通過しなければ`release_motion_gate_timeout`として自動的に無効になる。
音場開始後に誤判定と気付いた場合は直ちに `X` を押す。音場が消え、そのtrialは
`operator_rejected_false_release`、`status=invalid`として分母から外れる。
再把持判定などが先に働いて音場が消えた場合も、自動終了から5秒以内なら`X`で
直前の失敗trialが`trial_exclusions.csv`へ登録される。

ゲート通過後の実releaseで球体を再把持した場合は、誤判定ではなく捕捉失敗として
扱う。成功率を見てから除外基準を変更しない。

## 4. 実験前に固定する条件

- 同じ球体、AUTD配置、音場半径、intensity設定、制御ゲインを使用する
- カメラ位置、露光、照明、背景、calibrationを変更しない
- `--extra-delay-ms 0`
- `--success-hold-sec 5`
- `--observation-sec 10`
- 球体を投げず、意図的な水平速度を与えない
- 結果を見てrelease位置を成功しやすい方向へ修正しない

日をまたぐ場合はセッションを分け、日付、装置状態、calibration、Git commitを残す。
今回のCSV schemaは4なので、旧データとは別の新しいデータディレクトリを使う。

## 5. 実験実行

計画release XYZは指定しない。実測XYZが各trialに自動記録される。

PowerShellでは `\\` ではなくバッククォートで改行し、Windows側のPythonを使う。

```powershell
.\venv\Scripts\python.exe manual_release_tracking\experiments\mediapipe_auto_release_hold.py `
  --condition-id center_freehand_100_gate_v1 `
  --extra-delay-ms 0 `
  --success-hold-sec 5 `
  --observation-sec 10 `
  --target-valid-trials 100 `
  --paper-data-dir manual_release_tracking\si2026_data_center100_gate_v1 `
  --operator "Hiroto Yoshida"
```

macOS/Linuxの場合:

```bash
venv/bin/python manual_release_tracking/experiments/mediapipe_auto_release_hold.py \
  --condition-id center_freehand_100_gate_v1 \
  --extra-delay-ms 0 \
  --success-hold-sec 5 \
  --observation-sec 10 \
  --target-valid-trials 100 \
  --paper-data-dir manual_release_tracking/si2026_data_center100_gate_v1 \
  --operator "Hiroto Yoshida"
```

## 6. 1試行ごとの手順

1. 両カメラで球体が安定検出され、手状態が`GRASPED`になることを確認する。
2. 球体を目視でAUTD中心付近へ移す。厳密な定位置合わせは行わない。
3. 球体を押し出さないよう、指を左右へ開いてreleaseする。
4. 音場が出た後に球体がまだ指に保持されていると分かった場合だけ、直ちに`X`を押す。
5. 実release後は失敗終了またはrelease後10秒まで、球体や音場へ手を入れない。
6. 10秒を過ぎてから再把持する。次の試行は手状態と球検出が安定してから始める。
7. 10試行ごとに解析を実行し、`eligible_trials`だけを確認する。

危険時を除き、失敗しそうという理由で途中停止しない。`X`は目視で実releaseして
いないことが明らかな場合だけ使用する。

## 7. 試行数確認と解析

PowerShell:

```powershell
.\venv\Scripts\python.exe manual_release_tracking\analysis\analyze_si2026_trials.py `
  --data-dir manual_release_tracking\si2026_data_center100_gate_v1 `
  --output-dir manual_release_tracking\si2026_figures_center100_gate_v1 `
  --target-valid-trials 100
```

`outcome_summary.csv`の`scope=all`行で、`eligible_trials=100`かつ
`remaining_valid_trials=0`になった時点で終了する。`attempts`には無効trialも含まれる。

主な図:

- `successful_release_trajectory`: 代表成功試行の0--10秒のX/Y/Z・3D軌道
- `control_mechanism_validation`: Z、制動参照、速度、intensityと状態遷移
- `braking_distance_validation`: 計画停止距離と制動終了時実測距離
- `outcome_trajectory_comparison`: 成否別の相対軌道
- `behavior_pattern_counts`: 成功と失敗局面の件数
- `release_outcomes_3d`: 成否別release位置
- `success_failure_comparison`: release位置、速度、遅延、予測誤差
- `failure_reason_counts`: 5秒到達前の終了理由

主な表:

- `outcome_summary.csv`: 分母、成功数、成功率、95% Wilson信頼区間、残試行数
- `braking_validation.csv`: 計画・実測停止距離、停止位置誤差、制動終了速度
- `behavior_pattern_summary.csv`: 成功・失敗挙動パターン別件数
- `success_failure_comparison.csv`: 中央値、四分位範囲、Cliff's delta
- `exclusion_summary.csv`: 無効・手動除外trialと理由
- `trial_metrics_enriched.csv`: 1 trial 1行の全解析指標

## 8. 解釈

失敗原因は、終了理由、軌道、制動終了の有無、`LOCAL_HOLD`到達の有無を合わせて
判断する。終了理由は最後に検出された事象であり、必ずしも物理的な根本原因ではない。
成否差とCliff's deltaも関連を示すもので、単独では因果関係を示さない。
