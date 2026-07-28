# Manual Release Handoff Experiment

手でつまんだ球体を `r` キーでその場保持へ受け渡すための実験用コードです。

## 実行

```bash
cd /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation
python3 manual_release_tracking/experiments/manual_release_hold.py
```

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
