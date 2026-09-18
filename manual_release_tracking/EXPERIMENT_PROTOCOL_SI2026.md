# SI2026 中心近傍の自由落下捕捉実験プロトコル

## 1. 目的

定位置治具は使用せず、指で把持した球体をAUTD中心付近で結果を見ながら調整せずに
50回releaseする。自然に生じるrelease位置と速度のばらつきを記録し、次を評価する。

1. 捕捉成功時のrelease後の球体軌道
2. 成功時と失敗時のrelease位置、初期速度、遅延、予測誤差の比較
3. 失敗時の軌道と、システムが記録した終了理由の集計

広いワークスペースの境界や、位置ごとの成功率は今回の目的としない。

## 2. 試行数と判定

目標は「解析対象と50試行」である。ソフトウェアが作成した全trial行を50行と数えない。

- 成功: release画像の撮影時刻から10秒以上、再接触や制御停止なしで保持できた
- 失敗: 実際にreleaseしたが、10秒以内に運動推定不成立、計測ロスト、
  workspace離脱、SETTLE timeout、または制御停止に到達した
- 除外: 球体をreleaseしていないのにreleaseと判定された、手が再接触した、
  予備操作であった、または有効なrelease前に中断した

成功と失敗の合計が50になるまで実施する。除外trialはログから削除せず、除外理由を
残す。単に捕捉に失敗したという理由で除外してはならない。

## 3. 自動除外と手動除外

解析コードは次を自動的に除外する。

- `status` が `invalid` または `aborted`
- `regrasp_detected...`
- `release_candidate_contact_returned`
- 手動releaseフォールバックと競合したtrial
- 前のtrial完了前に新しいrelease候補が作られたtrial

音場が出たが実際にはreleaseしていなかったなど、自動理由だけで判別できない場合は
手動除外に登録する。

```bash
venv/bin/python manual_release_tracking/analysis/manage_trial_exclusions.py \
  --data-dir manual_release_tracking/si2026_data_center50 \
  add --trial-id SESSION_T0003_123456 \
  --reason operator_confirmed_false_trigger
```

登録済みの除外を確認する。

```bash
venv/bin/python manual_release_tracking/analysis/manage_trial_exclusions.py \
  --data-dir manual_release_tracking/si2026_data_center50 list
```

間違って登録した場合:

```bash
venv/bin/python manual_release_tracking/analysis/manage_trial_exclusions.py \
  --data-dir manual_release_tracking/si2026_data_center50 \
  remove --trial-id SESSION_T0003_123456
```

`trial_exclusions.csv`にtrial IDと理由が保存される。除外基準は実験開始前に固定し、
50試行を見終わってから都合よく変更しない。

## 4. 実験前に固定する条件

- 同じ球体を使用する
- AUTD配置、音場半径、intensity設定、制御ゲインを変更しない
- カメラ位置、露光、照明、背景、calibrationを変更しない
- `--extra-delay-ms 0`
- `--success-hold-sec 10`
- 球体を投げず、意図的な水平速度を与えない
- データを見てrelease位置を成功しやすい方向へ変更しない

日をまぐ場合はセッションを分け、日付、装置状態、calibration、Git commitを記録する。

## 5. 実験実行

計画release XYZは指定しない。実測XYZが各trialに自動記録される。

```bash
venv/bin/python manual_release_tracking/experiments/mediapipe_auto_release_hold.py \
  --condition-id center_freehand_50 \
  --extra-delay-ms 0 \
  --success-hold-sec 10 \
  --paper-data-dir manual_release_tracking/si2026_data_center50 \
  --operator YOUR_NAME
```

AUTD中心XYZは実行時に装置配置から取得され、`trial_summary.csv`の
`autd_center_x_mm`, `autd_center_y_mm`, `autd_center_z_mm`に保存される。

## 6. 1試行ごとの手順

1. 両カメラで球体が検出されていることを確認する。
2. 球体を指で把持し、目視でAUTD中心付近へ移動する。正確な定位置は行わない。
3. 手状態が`GRASPED`になったことを確認する。
4. 球体を押し出さないよう、指を左右へ開いてreleaseする。
5. release後は球体と音場へ再接触せず、10秒成功判定または自動失敗終了まで待つ。
6. 明確な誤トリガがあった場合は時刻をメモし、`trial_summary.csv`の直近行から
   trial IDを確認する。
7. 次のtrialまで球体と手検出が安定するのを待つ。

10試行ごにログをバックアップする。明らかに危険な場合を除き、失敗しそうだからという
理由で途中停止しない。

## 7. 試行数の確認と解析

実験途中でも解析コマンドを実行できる。

```bash
MPLCONFIGDIR=/tmp/si2026-matplotlib \
venv/bin/python manual_release_tracking/analysis/analyze_si2026_trials.py \
  --data-dir manual_release_tracking/si2026_data_center50 \
  --output-dir manual_release_tracking/si2026_figures_center50
```

`si2026_figures_center50/outcome_summary.csv`の`scope=all`行にある`eligible_trials`が50に達したら終了する。
`attempts`は除外trialも含むため50より多くなる可能性がある。

生成される図:

- `successful_release_trajectory.png/.svg`: 代表成功試行のX/Y/Zと3D軌道
- `outcome_trajectory_comparison.png/.svg`: 成否別のδx/δy/δz軌道
- `release_outcomes_3d.png/.svg`: 成否別のrelease位置のXY、XZ、3D表示
- `success_failure_comparison.png/.svg`: release位置、速度、遅延、予測誤差の比較
- `failure_reason_counts.png/.svg`: 10秒到達前の失敗終了理由

生成される表:

- `outcome_summary.csv`: 解析対象数、成功数、失敗数、成功率、95% Wilson信頼区間
- `release_range_summary.csv`: 成否別のrelease XYZと中心からの距離範囲
- `success_failure_comparison.csv`: 中央値、四分位範囲、Cliff's delta、各群の`n`
- `failure_reason_summary.csv`: 個別の失敗理由と分類別件数
- `exclusion_summary.csv`: 自動および手動除外されたtrialと理由
- `trial_metrics_enriched.csv`: 除外フラグと追加指標を含む1trial 1行のデータ

## 8. 失敗原因の調べ方

1. `failure_reason_counts` で直接の終了トリガを確認する。
2. `release_outcomes_3d` で失敗がAUTD中心から離れた位置に偏っていないか確認する。
3. `success_failure_comparison` で位置、速度、遅延、予測誤差の効果量を確認する。
4. `outcome_trajectory_comparison` で失敗前の位置変化を確認する。
5. 必要なtrial IDの元時系列とイベントログを調べる。

終了理由はシステムが最後に検出したトリガであり、物理的な根本原因とは限らない。
成否間の差も関連を示すが、単独で因果関係を示さない。失敗数が少ない場合は記述的・
探索的解析として報告する。
