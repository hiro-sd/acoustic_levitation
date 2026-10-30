# 8点同時多焦点の予備実験

円状8点を単一焦点で周回する従来STMと、同じ半径19 mmの8点に同時に
焦点を作る静的多焦点を比較する。通常のPID・捕捉入口は引き続きSTMが既定。
この予備実験では、利用者の希望により**多焦点専用の力推定は行わず、
従来STMで測定した鉛直力モデルをそのまま使用する**。したがってログ内の
推定力、力飽和判定、停止距離の予測は多焦点に対して未校正である。

AUTDとカメラを接続した実機で、球の下に受け皿を置いて試すこと。
この変更自体は実機で検証されていない。

## 1. PID保持を先に確認

```powershell
.\venv\Scripts\python.exe tracking\experiments\stereo_feedback_static_multi.py
```

EnterでPID開始、Lでログ記録、Escで終了。多焦点のログは
`tracking/stability_log_stereo_feedback_static_multi.csv` に分離した。
従来STMとの比較には `tracking/experiments/stereo_feedback.py` を使用する。
保持が不安定なら、落下捕捉には進まない。

## 2. 少数の落下捕捉試行

```powershell
.\venv\Scripts\python.exe manual_release_tracking\experiments\mediapipe_auto_release_hold.py --static-multi-focus --target-valid-trials 5
```

多焦点の既定condition IDは `static_multi_focus_circle`、試行データの
既定出力先は `manual_release_tracking/si2026_data_static_multi_focus`。
従来STMのデータと混在させない。session設定とnotesにも「未校正のSTM力
モデルを流用」と記録し、起動時にも警告を表示する。まずは少数試行で
落下量、捕捉後の振動、指令遅延、成功／失敗を観察する。

## 元に戻す

`--static-multi-focus` を付けずに通常の捕捉入口を実行すれば、従来STMに
戻る。PIDも通常の `tracking/experiments/stereo_feedback.py` を使えばよい。
PID本体、捕捉状態機械、AUTD送信本体は変更していない。予備実験に固有の
変更は多焦点PIDのログ先と捕捉入口のオプションだけであり、後で削除できる。
