# Automatic Release Capture and Hold

手でつまんだ球体のreleaseをMediaPipeで検出し、ステレオ3D計測と
AUTD制御によって捕捉・保持する実験コードです。

## 準備

初回のみ、依存パッケージとMediaPipe公式モデルを準備します。

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python -m pip install -r manual_release_tracking/requirements-mediapipe.txt
python manual_release_tracking/experiments/download_mediapipe_hand_model.py
```

## 実行

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python manual_release_tracking/experiments/mediapipe_auto_release_hold.py
```

SI2026用の中心近傍100試行では、計画release座標を指定せず、各試行の実測
release XYZを記録します。追加無音時間の既定値は0 msで、従来の捕捉開始
タイミングを変更しません。

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

実験条件、試行数、成功定義、解析手順は
[`EXPERIMENT_PROTOCOL_SI2026.md`](./EXPERIMENT_PROTOCOL_SI2026.md)を参照してください。

この実験では次の順序で動作します。

1. 起動後は球検出、ステレオ3D推定、手認識だけを行い、音場は出しません。
2. 現在の自動保持実験ではXYカメラだけで親指と人差し指を判定し、
   MediaPipeへの投入上限を90 fpsとして、接触が継続すると `GRASPED` になります。
3. XYカメラで両指先が明確に離れた状態が継続すると
   15 msの確認後に`GRASPED -> RELEASED`へ遷移します。手ランドマークの一時的な
   検出ロストはreleaseとして扱いません。
4. 最初の明確な離反を `RELEASE_CANDIDATE` とした時点で、捕捉専用の
   3D運動履歴を作ります。ステレオ測定は常に撮影時刻付きで直近200 msを
   保存しており、非同期MediaPipe結果が届いた時点で、離反画像の撮影時刻
   以降の測定だけを捕捉用履歴へ再投入します。これによりrelease前の測定を
   混ぜず、MediaPipe処理中に取得済みのrelease後フレームを再利用します。
   この観測中は音場を出しません。接触へ戻った場合は候補と履歴を破棄します。
5. release確定に加えて、release候補より後の新規ステレオ測定が最低3フレーム、
   10 ms以上揃ってから初期捕捉を開始します。直近最大60 msのXYZ位置を
   時間に対して回帰し、最新時刻の位置と速度を求めます。Zは音場がない間の
   自由落下として既知の重力加速度を含むモデルで回帰するため、観測区間中央の
   平均速度ではなく最新時刻の速度を推定します。その後、最新測定値の経過時間と
   StaticとSTMの送信統合後に実測したAUTD送信遅延7.5 msを考慮した
   予測位置へ音場中心を合わせます。7.5 msはセッション最初のAUTD送信前だけに
   用いる初期値であり、送信実績が得られた後は、各測定のタイムスタンプから求めた
   測定経過時間と、そのセッションのAUTD命令投入から送信完了までの遅延の指数移動
   平均を加えた予測時間を使用します。制動中と保持中のPIDも同じ方法で予測時間を求めます。
   さらに `vz <= -100 mm/s`かつ候補時からZが2.5 mm以上下降した推定が
   3回連続した場合だけ音場を出します。120 ms以内に物理ゲートを通過しない
   場合は、把持中の誤release候補として無効終了します。
6. `CAPTURE_BRAKING` 中は現在のZ速度を120 msで停止させるための必要音響力を
   `F_req = m(g + u/T) - F_b`（`u=max(0,-v_z)`, `F_b`は浮力）で計算します。
   300試行のカメラ・ステップ応答実験（239有効試行）のゼロオフセット重み付き
   実測質量0.53 gに補正したモデル
   `F(A)=9.94437622 A-1.02485060 mN`を0.5から1.0まで区分線形補間し、
   intensityを逆算します。下降・静止中は既知の保持値0.6を下限とし、モデルの
   最大力約8.92 mNを超える場合は1.0で飽和を記録します。
   初回だけ自由落下モデルで音場位置を予測します。送信完了後は自由落下
   予測をやめ、初期位置・速度、最大測定力、作業空間から生成した
   一定減速の `z_ref(t), v_ref(t)` を追従します。XYは初回の複数フレーム
   予測位置を固定基準として保存します。最初の捕捉位置・強度は、そのsequenceの
   AUTD送信完了が確認されるまで後続更新で上書きしません。送信完了後は、保持時と
   同じタイムスタンプ依存のPIDでXYを制御し、Zだけを制動軌道で上書きします。
   最終的に送る音場中心は球から15 mm以内に制限し、音場の作用範囲外へ離れることを
   防ぎます。
   Zはその間も従来の制動軌道を継続します。
   上向き速度が20 mm/sを超えた場合はintensityを0.5へ下げ、
   音場中心のZを直前値より上へ動かしません。
7. release確定後、最低120 msの捕捉期間を経て、実測Z速度が-30 mm/s以上の
   状態が15 ms続いた時点で、計画終了時刻を待たずにZ制動を終えます。
   その時点のフィルタ済みZ位置を一時高さ基準とし、XYは固定捕捉基準として
   `LOCAL_HOLD`へ直接移ります。制動が1秒続いた場合も音場は停止せず、releaseが
   確定済みならその時点の位置で`LOCAL_HOLD`へ移ります。
8. 成功判定はrelease後5秒です。ただし論文用時系列はrelease後10秒まで継続し、
   この実験ランナーでは10秒より前に`RETURN_TO_HOME`へ移りません。その後、
   既存の速度制限付き復帰により基準位置をAUTD中心・Z=400 mmへ戻します。
   基準位置の移動速度はXY 20 mm/s、Z 15 mm/sです。復帰完了後も
   通常PIDでhomeを保持します。
9. 自動開始した捕捉・保持中にXYカメラの `GRASPED` が再成立した場合は、
   再把持または誤releaseと判断し、intensityを0にして音場を停止します。
   また、いずれの自動保持状態でもステレオ3D測定が100 ms以上更新されない
   場合は、安全のため音場を停止します。

Zカメラは引き続きステレオ3D位置推定に使用しますが、MediaPipeの手推論は
実行せず、GRASPED/RELEASED判定には影響しません。判定カメラは実験設定の
`cfg.mediapipe_auto_release_camera` で `xy`, `z`, `both` から選択できます。

制動中の0.005未満のintensity変化はデッドバンドで無視し、微小な速度ノイズに
よる再送を抑えます。上向き速度が20 mm/sを超えた場合の0.5出力も安全対策として
残します。`FOLLOW_AND_BRAKE`は無効です。
また、非同期MediaPipe結果が80 msより古い場合は自動トリガーを拒否します。
`X`を押すと、音場開始後に気付いた誤releaseを直ちに無音化し、そのtrialを
`invalid`として記録します。自動停止が先に起きても5秒以内なら、直前の失敗trialを
除外リストへ登録します。`r` は従来どおり手動フォールバックとして利用でき、
`ENTER`で保持を停止できます。
`r` で手動開始した保持はMediaPipeの再GRASPEDでは停止しません。

## ログ

ログは `manual_release_tracking/manual_release_log.csv` に出力されます。
自動保持実験のログは `manual_release_tracking/auto_release_log.csv` に出力されます。

自動release捕捉のイベントと遅延は、実行時に自動で
`manual_release_tracking/auto_release_capture_events.csv`へ追記されます。
このCSVには次が記録されます。

- 最初に指先が離れたMediaPipe画像のタイムスタンプ
- release候補からAUTDコマンド投入・実送信までの時間
- 送信時に使用したステレオ位置と速度
- 予測時間と実測遅延を考慮した予測位置
- 実際のtarget、intensity、AUTDコマンドsequence
- release確定、候補取消、`CAPTURE_BRAKING`/`LOCAL_HOLD`/
  `RETURN_TO_HOME`遷移、再把持停止の理由

実効遅延の測定専用ログは、実行時に自動で
`manual_release_tracking/auto_release_delay_measurements.csv`へ追記されます。
1つの初期捕捉コマンドにつき1行で、3D測定からコマンド投入、送信スレッド取得、
AUTD送信開始・完了までの時間を分解して記録します。
`prediction_shortfall_ms`は、現在の予測時間がAUTD送信完了までに何ms不足したかを
表します。現在は初期sequenceを厳密に待つため、正常動作では
`command_superseded=0`になります。

制動軌道と実測球状態の比較は、
`manual_release_tracking/auto_release_trajectory_log.csv`へ自動記録されます。
各新規3D測定に対し、計画停止時間・停止位置・必要力、`z_ref/v_ref`、
実測`z/vz`、誤差補正、target、intensityを記録します。

SI2026用実験では、上記ログに加えて
`--paper-data-dir`で指定したディレクトリへ次を自動記録します。

- `trial_timeseries.csv`: release候補から10秒観測終了または失敗までの連続時系列
- `trial_summary.csv`: releaseゲート、計画・実測制動、保持到達、成否を1試行1行で集計
- `autd_update_telemetry.csv`: 全音場更新のsequence、量子化強度、送信完了時刻、
  measurement age、推定遅延、prediction horizonと実測不足時間
- `session_*.json`: 実験条件、全設定、calibrationハッシュ、Git commit

`--target-valid-trials`で指定したsuccess/failure試行数に到達すると、実験プログラムは
自動終了します。`invalid`と`aborted`はこの回数に含めません。

論文用グラフと条件別集計表は次のコマンドで生成できます。

```bash
MPLCONFIGDIR=/tmp/si2026-matplotlib \
venv/bin/python manual_release_tracking/analysis/analyze_si2026_trials.py \
  --data-dir manual_release_tracking/si2026_data_center100_gate_v1 \
  --output-dir manual_release_tracking/si2026_figures_center100_gate_v1 \
  --target-valid-trials 100
```

誤トリガなどの手動除外は
`manual_release_tracking/analysis/manage_trial_exclusions.py`でtrial IDと理由を記録します。
