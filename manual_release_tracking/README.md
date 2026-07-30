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
4. 新しい `just_released` イベントを1回だけ受理し、その時点で利用できる
   最新ステレオ3D位置を一時基準にします。
5. 手動実験と同じ通常PID、円軌道半径19 mm、intensity 0.6で
   `LOCAL_HOLD`を開始します。
6. 自動開始した `LOCAL_HOLD` 中にXYカメラの `GRASPED` が再成立した場合は、
   再把持または誤releaseと判断し、intensityを0にして音場を停止します。

Zカメラは引き続きステレオ3D位置推定に使用しますが、MediaPipeの手推論は
実行せず、GRASPED/RELEASED判定には影響しません。判定カメラは実験設定の
`cfg.mediapipe_auto_release_camera` で `xy`, `z`, `both` から選択できます。

最初の比較実験で原因を分離できるよう、落下速度によるintensity変更、
必要放射圧計算、`FOLLOW_AND_BRAKE`、homeへの自動復帰は無効です。
また、非同期MediaPipe結果が80 msより古い場合は自動トリガーを拒否します。
`r` は従来どおり手動フォールバックとして利用でき、`ENTER`で保持を停止できます。
`r` で手動開始した保持はMediaPipeの再GRASPEDでは停止しません。

## ログ

ログは `manual_release_tracking/manual_release_log.csv` に出力されます。
自動保持実験のログは `manual_release_tracking/auto_release_log.csv` に出力されます。
