# 月のかけら救出ミッション

自由落下球体の捕捉実験と同時に表示する、子ども向けの独立画面です。制御ループに処理を追加せず、`auto_release_capture_events.csv` に新しく追記されたイベントだけを監視します。

Windows PowerShell で、捕捉実験とは別のターミナルから実行します。

```powershell
.\venv_autd3\Scripts\python.exe manual_release_tracking\demo_open_campus\moon_fragment_mission.py
```

操作は次の通りです。

- `Space`: タイトル → 準備画面 → 3秒カウントダウン。成功または失敗画面から次の体験へ戻るときにも使います。
- `Esc`: 画面を終了。
- `Enter`: AUTDなしのリハーサル用に解放を疑似。
- `S` / `F`: 成功・失敗画面の表示テスト。
- `R`: 準備画面へ戻る。

実験側が `release_candidate_command_enqueued` を記録すると「超音波ネット展開中」に切り替わります。`capture_braking_to_local_hold` 到達後、解放から2秒間停止イベントがなければ成功と表示します。2秒以内に `automatic_field_stopped` が記録された場合や、4秒以内に保持段階へ到達しない場合は、ロボット回収の失敗画面へ移行します。この2秒判定は展示用であり、論文用ログの5秒成功判定は変更しません。
