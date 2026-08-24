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

この実験では次の順序で動作します。

1. 起動後は球検出、ステレオ3D推定、手認識だけを行い、音場は出しません。
2. 現在の自動保持実験ではXYカメラだけで親指と人差し指を判定し、
   MediaPipeへの投入上限を90 fpsとして、接触が継続すると `GRASPED` になります。
3. XYカメラで両指先が明確に離れた状態が継続すると
   25 msの確認後に`GRASPED -> RELEASED`へ遷移します。手ランドマークの一時的な
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
   予測位置へ音場中心を合わせます。
   100 ms以内に有効な複数フレーム推定を作れない場合は、安全のため音場を
   出さずにそのrelease試行を終了します。
6. `CAPTURE_ALIGN` 中は現在のZ速度を120 msで停止させるための必要力を
   `F = m(g + u/T)`で計算し、補正したロードセルモデル
   `(0.5, 0.6, 0.7, 0.8) = (4, 5, 6, 7) mN`を区分線形補間して
   intensityを逆算します。例えば5.5 mNなら0.65です。下降・静止中は
   既知の保持値0.6を下限とし、7 mNを超える場合は0.8で飽和を記録します。
   初回だけ自由落下モデルで音場位置を予測します。送信完了後は自由落下
   予測をやめ、初期位置・速度、最大測定力、作業空間から生成した
   一定減速の `z_ref(t), v_ref(t)` を追従します。XYは初回の複数フレーム
   予測位置を固定基準として保存します。最初の音場が実際に送信された後は、
   固定基準からの位置偏差と、複数フレーム推定したXY速度を使う専用PD制御へ
   直ちに切り替えます。捕捉中は積分項を使わないため、出力制限中の積分飽和を
   引き継ぎません。`CAPTURE_ALIGN`と`CAPTURE_SETTLE`では、最終的に送る
   音場中心を球から15 mm以内に制限し、固定基準へ戻す方向の力を維持しながら
   音場の作用範囲外へ離れることを防ぎます。制限へ初めて到達した場合は
   `capture_xy_pd_saturated`をイベントログへ記録します。
   Zはその間も従来の制動軌道を継続します。
   上向き速度が20 mm/sを超えた場合はintensityを0.5へ下げ、
   音場中心のZを直前値より上へ動かしません。
7. release確定後、最低120 msの捕捉期間を経て、実測Z速度が-30 mm/s以上の
   状態が15 ms続いた時点で、計画終了時刻を待たずにZ制動を終えます。
   その時点のフィルタ済みZ位置を一時高さ基準とし、XYは固定捕捉基準の
   専用PDを継続したまま`CAPTURE_SETTLE`へ移ります。
   `CAPTURE_ALIGN`が1秒続いた場合も音場は停止せず、その時点の位置で
   `CAPTURE_SETTLE`へ強制移行します。
8. `CAPTURE_SETTLE`でZ速度、XY速度、一時基準からのXY距離とZ誤差が
   設定範囲内になり、通常intensity 0.6の状態が25 ms続くと
   `LOCAL_HOLD`へ移ります。移行条件は`|vz| <= 60 mm/s`、
   XY速度60 mm/s以下、XY距離15 mm以下、Z誤差10 mm以下です。
9. `LOCAL_HOLD`で離した位置を3秒間保持した後、既存の速度制限付き
   `RETURN_TO_HOME`へ移り、基準位置をAUTD中心・Z=400 mmへ徐々に戻します。
   基準位置の移動速度はXY 20 mm/s、Z 15 mm/sです。復帰完了後も
   通常PIDでhomeを保持します。また、端の音場などで振動が残り
   `LOCAL_HOLD`の厳しい条件を満たさない場合も、`CAPTURE_SETTLE`が10秒間
   連続し、球体が緩和した速度・音場距離・Zワークスペース条件内にあれば、
   現在のフィルタ済み球体位置から`RETURN_TO_HOME`へ直接移行します。
10. 自動開始した捕捉・保持中にXYカメラの `GRASPED` が再成立した場合は、
   再把持または誤releaseと判断し、intensityを0にして音場を停止します。
   また、いずれの自動保持状態でもステレオ3D測定が100 ms以上更新されない
   場合は、安全のため音場を停止します。

Zカメラは引き続きステレオ3D位置推定に使用しますが、MediaPipeの手推論は
実行せず、GRASPED/RELEASED判定には影響しません。判定カメラは実験設定の
`cfg.mediapipe_auto_release_camera` で `xy`, `z`, `both` から選択できます。

`CAPTURE_SETTLE`でも同じ必要力ベースの連続補間を使います。0.005未満の
intensity変化はデッドバンドで無視し、微小な速度ノイズによる再送を抑えます。
上向き制動は安全対策として残し、`vz > 60 mm/s`で0.5にして
`vz < 20 mm/s`まで維持します。`FOLLOW_AND_BRAKE`は無効です。
また、非同期MediaPipe結果が80 msより古い場合は自動トリガーを拒否します。
`r` は従来どおり手動フォールバックとして利用でき、`ENTER`で保持を停止できます。
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
- release確定、候補取消、`CAPTURE_SETTLE`/`LOCAL_HOLD`/
  `RETURN_TO_HOME`遷移、再把持停止の理由

実効遅延の測定専用ログは、実行時に自動で
`manual_release_tracking/auto_release_delay_measurements.csv`へ追記されます。
1つの初期捕捉コマンドにつき1行で、3D測定からコマンド投入、送信スレッド取得、
AUTD送信開始・完了までの時間を分解して記録します。
`prediction_shortfall_ms`は、現在の予測時間がAUTD送信完了までに何ms不足したかを
表します。`command_superseded=1`の行は、初期コマンドが後続更新に置き換えられた
試行なので、遅延の代表値を求める際は分けて扱います。

制動軌道と実測球状態の比較は、
`manual_release_tracking/auto_release_trajectory_log.csv`へ自動記録されます。
各新規3D測定に対し、計画停止時間・停止位置・必要力、`z_ref/v_ref`、
実測`z/vz`、誤差補正、target、intensityを記録します。
