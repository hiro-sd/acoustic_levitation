# Manual Release Handoff Experiment

手でつまんだ球体を `r` キーでその場保持へ受け渡すための実験用コードです。

## 実行

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python3 manual_release_tracking/experiments/manual_release_hold.py
```

## Release判定preview

音場制御に接続せず、指を離した判定だけを可視化する場合:

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python3 manual_release_tracking/experiments/release_detection_preview.py
```

previewでは、2台のカメラ映像に以下を表示します。

- 球体検出円
- 選択カメラの球近傍にある肌色候補輪郭
- 球半径で正規化した `gap/r`
- 球表面の接触リングに占める肌色の割合 `contact`
- `GRASPED` 時に自動保存した接触基準と現在値の比
- grasp/release候補の直近フレーム投票数
- `WAITING_FOR_GRASP` / `GRASPED` / `RELEASED`
- 推定3D位置、速度、10ms後予測位置

現在のデフォルトでは、奥行き方向の画像上の重なりを接触と誤認しにくくするため、
`both` でXY・Z両カメラの判定一致を要求します。
`manual_release_tracking/experiments/release_detection_preview.py` 末尾の
`cfg.release_preview_camera` を変更すると、`xy`, `z`, `either`, `both` を切り替えられます。

判定は固定ピクセル値だけではなく、球半径で正規化した接触量と、
`GRASPED` になった直後の基準値からの相対変化を使用します。
一時的な肌色抽出失敗や短時間の球検出ロストだけでは `RELEASED` にしません。
`RELEASED` は終端状態ではなく、previewで遷移を確認できるよう
100フレーム（約190 fps時に約0.53秒）表示した後に自動で
`WAITING_FOR_GRASP`へ戻り、次の把持を判定できます。

`SPACE` で判定状態をリセットし、`ESC` で終了します。

## MediaPipe Hand Landmarker preview

現在の肌色previewとは別に、MediaPipe Tasks版Hand Landmarkerで親指先と
人差し指先を追跡するpreviewを実行できます。AUTDには接続しません。

初回のみ、依存パッケージと公式モデルを準備します。

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python -m pip install -r manual_release_tracking/requirements-mediapipe.txt
python manual_release_tracking/experiments/download_mediapipe_hand_model.py
```

previewを実行します。

```bash
python manual_release_tracking/experiments/mediapipe_release_detection_preview.py
```

MediaPipeは各カメラに1つずつ、`num_hands=1`、`LIVE_STREAM`で非同期実行します。
カメラ・球検出ループを不要に重くしないよう、デフォルトでは各カメラ最大60 fpsで
MediaPipeへ投入し、処理中に古くなった結果は制御ループで待ちません。
古い推論結果で現在の球位置を評価しないよう、各結果はMediaPipeへ渡した画像の
撮影時刻と、その画像で検出した球中心・半径に紐づけています。

画面には以下を表示します。

- 親指先（オレンジ）と人差し指先（緑）
- 各指先から球表面までの正規化距離 `gap/r`
- 球中心から見た2指の角度
- XY、Z、両カメラ一致によるpreview状態
- MediaPipe結果FPS
- 撮影から結果取得までの遅延 `Result latency`
- 最新結果が現在から何ms前かを示す `Result age`
- 単眼・両眼の手動正解ラベル区間における誤判定率

比較用キー:

- `G`: 正解をGRASPEDにする
- `N`: 正解をNOT_GRASPED / RELEASEDにする
- `U`: 正解不明として誤判定率の集計から除外する
- `C`: 誤判定率をリセットする
- `SPACE`: preview状態をリセットする
- `ESC`: 終了する

両眼のpreview判定は、GRASPEDにはXY・Z両方の接触候補を要求します。
一方、GRASPED後はいずれかのカメラで指先が離れた状態が継続すると
RELEASEDになります。これは把持の誤認を抑えつつ、releaseを遅らせすぎないための
preview用の非対称判定です。

## 操作

- 起動直後は、球体検出とステレオ3D位置推定だけを行います。AUTDの音場は出ません。
- 球体を人差し指と親指などでつまみ、2台のカメラから検出できる状態にします。
- `r` を押すと、その時点の3D位置を一時基準にして音場を出し、`LOCAL_HOLD` に入ります。
- 画面・ターミナルで armed 表示を確認し、約0.3秒後に手を離します。
- その後は、元の home へ戻さず、その場でPID保持します。
- `ENTER` は、rトリガー後の一時停止用です。
- `ESC` で終了します。

## MediaPipe releaseから自動保持

MediaPipe previewと手動保持の確認後、`GRASPED -> RELEASED` を自動的に
LOCAL_HOLDへ接続する独立実験を実行できます。

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python manual_release_tracking/experiments/mediapipe_auto_release_hold.py
```

この実験では次の順序で動作します。

1. 起動後は球検出、ステレオ3D推定、手認識だけを行い、音場は出しません。
2. 現在の自動保持実験ではXYカメラだけで親指と人差し指を判定し、
   接触が継続すると `GRASPED` になります。
3. XYカメラで両指先が明確に離れた状態が継続すると
   `GRASPED -> RELEASED` へ遷移します。手ランドマークの一時的な
   検出ロストはreleaseとして扱いません。
4. 最初の明確な離反を `RELEASE_CANDIDATE` とし、35 msのrelease確定を
   待たずに予測位置へ音場を出します。接触へ戻った場合は候補を取り消して
   intensityを0にします。
5. 球を持っている間からステレオ3D位置・速度を常時推定し、最新測定値の
   経過時間と実測したAUTD送信遅延13 msを考慮した予測位置へ
   音場中心を合わせます。
6. `CAPTURE_ALIGN` 中は現在のZ速度を120 msで停止させるための必要力を
   `F = m(g + u/T)`で計算し、補正したロードセルモデル
   `(0.5, 0.6, 0.7, 0.8) = (4, 5, 6, 7) mN`を区分線形補間して
   intensityを逆算します。例えば5.5 mNなら0.65です。下降・静止中は
   既知の保持値0.6を下限とし、7 mNを超える場合は0.8で飽和を記録します。
   初回だけ自由落下モデルで音場位置を予測します。送信完了後は自由落下
   予測をやめ、初期位置・速度、最大測定力、作業空間から生成した
   一定減速の `z_ref(t), v_ref(t)` を追従します。XYは初回予測位置に固定します。
   上向き速度が20 mm/sを超えた場合はintensityを0.5へ下げ、
   音場中心のZを直前値より上へ動かしません。
7. release確定後、実測Z速度が-30 mm/s以上の状態が15 ms続いた時点で、
   計画終了時刻を待たずに制動を終えます。その時点のフィルタ済み3D位置を
   一時基準として、通常PIDを使う `CAPTURE_SETTLE`へ移ります。
   `CAPTURE_ALIGN`が1秒続いた場合も音場は停止せず、その時点の位置で
   `CAPTURE_SETTLE`へ強制移行します。
8. `CAPTURE_SETTLE`でZ速度、XY速度、一時基準からのXY距離とZ誤差が
   設定範囲内になり、通常intensity 0.6の状態が25 ms続くと
   `LOCAL_HOLD`へ移ります。移行条件は`|vz| <= 60 mm/s`、
   XY速度60 mm/s以下、XY距離15 mm以下、Z誤差10 mm以下です。
9. `LOCAL_HOLD`で離した位置を3秒間保持した後、既存の速度制限付き
   `RETURN_TO_HOME`へ移り、基準位置をAUTD中心・Z=400 mmへ徐々に戻します。
   基準位置の移動速度はXY 20 mm/s、Z 15 mm/sです。復帰完了後も
   通常PIDでhomeを保持します。
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
- release確定、候補取消、`CAPTURE_SETTLE`/`LOCAL_HOLD`遷移、再把持停止の理由

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
