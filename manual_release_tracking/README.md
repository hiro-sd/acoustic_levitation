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

## 操作

- 起動直後は、球体検出とステレオ3D位置推定だけを行います。AUTDの音場は出ません。
- 球体を人差し指と親指などでつまみ、2台のカメラから検出できる状態にします。
- `r` を押すと、その時点の3D位置を一時基準にして音場を出し、`LOCAL_HOLD` に入ります。
- 画面・ターミナルで armed 表示を確認し、約0.3秒後に手を離します。
- その後は、元の home へ戻さず、その場でPID保持します。
- `ENTER` は、rトリガー後の一時停止用です。
- `ESC` で終了します。

## ログ

ログは `manual_release_tracking/manual_release_log.csv` に出力されます。
