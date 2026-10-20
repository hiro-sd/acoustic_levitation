# 円軌道STM音場のステップ応答による力測定

## 目的

通常のPID制御で静止させた球体に短時間の固定音場を与え、2台の高速
カメラで得た3次元軌道から加速度と音響力を推定する。

ここで指定する`intensity`はAUTDの定振幅変調に与える振幅比であり、
音響インテンシティそのものではない。SDKへは0～255の整数に量子化して
送られるため、要求値、変調バイト、実際の量子化後比率をすべて保存する。

カメラのセンサ解像度は各648×488 pxである。ただし、画素数だけから
mm/pxや奥行き分解能は決まらない。この実験では次の値を記録する。

- 各画像における球体中心と球体半径 [px]
- ステレオ校正後の球体中心 [mm]
- ステップ前の安定保持区間におけるXYZ標準偏差とrobust sigma
- 2個の単一フレーム位置の差を識別する保守的な95%変位量
- 二次軌道フィットの残差、決定係数、加速度標準誤差

画像上の概算スケールは、球体直径を画像上の球体直径で割った

```text
approximate_scale = sphere_diameter_mm / (2 * detected_radius_px)
```

として記録する。この値は球体付近の面内方向の目安であり、ステレオの
奥行き分解能そのものではない。制御に使うXYZの実効分解能は、安定保持
区間の再構成XYZのばらつきから評価する。

## 測定原理

ステップ中はXYZすべてのPID更新を停止し、ステップ開始時の球体位置を
基準に焦点群中心と強度を固定する。AUTD送信完了から10 ms（STM 1周）を
除外した軌道に

```text
p(t) = p0 + v0*t + 0.5*a*t^2
```

を最小二乗法で当てはめる。鉛直上向きを正とした音響力は

```text
F_ac,z = m * (a_z + g) - F_b
F_b    = rho_air * sphere_volume * g
```

で求める。数値上、`kg * mm/s^2 = mN`である。測定開始時の速度を小さく
制限しているため、初期区間の空気抗力はまず無視する。必要なら後の解析で
抗力モデルを追加する。この二次式から得られる値は解析区間内の実効平均
加速度であるため、ログには解析区間の平均球体位置も保存する。

`offset_x/y/z_mm`は「試験音場の中心－ステップ開始時の球体中心」である。
すべて0なら、円軌道の中心を観測球体中心に一致させる。

したがって、`offset_z_mm > 0`では円軌道中心を球体中心より上に、
`offset_z_mm < 0`では下に配置する。ログの`actual_relative_z_mm`と
`fit_mean_relative_z_mm`は逆向きの「球体中心－試験音場中心」なので、
ステップ開始時には概ね`actual_relative_z_mm = -offset_z_mm`となる。

振幅だけの依存性を調べる場合にも、再現性のため
`--z-offsets-mm 0`を明示する。捕捉時の力を校正する場合は相対位置によって
力が変化するため、捕捉中に使用し得る範囲を複数のz offsetで測定する。
各試行の代表位置には、指令したoffsetだけでなく、解析区間中の実位置である
`fit_mean_relative_z_mm`も使用する。

## 最初の安全確認

最初は既知の範囲である0.5～0.8、中心オフセット0、各3回程度で確認する。
リポジトリ直下からPowerShellで実行する。
既定値は質量0.53 g、直径38 mmなので、実験前に同じ球体を測定し、異なる
場合は`--sphere-mass-g`と`--sphere-diameter-mm`で実測値を指定する。

```powershell
.\venv_autd3\Scripts\python.exe tracking\experiments\acoustic_force_step_response.py `
  --intensities "0.5,0.6,0.7,0.8" `
  --z-offsets-mm 0 `
  --repeats 3 `
  --pulse-ms 60 `
  --sphere-mass-g 0.530 `
  --sphere-diameter-mm 38.0 `
  --operator "Hiroto Yoshida"
```

ウィンドウが開いたら次の順に操作する。

1. 通常の円軌道音場に球体を入れる。
2. `ENTER`でPID保持を開始する。
3. 球体が安定したら`S`でランダム化済みの測定列を開始する。
4. 異常時は`ESC`を押す。AUTD接続終了時にもsilent stopが実行される。

PID保持中に`ENTER`を押してPIDを停止すると、固定音場の中心は初期位置
（既定ではAUTD中心の`z=400 mm`）へ、強度は`--baseline-intensity`へ戻る。
パルス後に球体が落下した場合は、この状態で球体を再配置し、再度`ENTER`で
PID保持を開始してから`S`で残りの試行を続行できる。

条件順は反復ごとのブロックランダム化で生成する。1ブロックには指定した
振幅とoffsetの全組合せが1回ずつ含まれ、ブロック内だけをランダム化する。
したがって、ブロック境界で中断した場合も各条件の反復数が揃う。

60 msの予定時間に達する前でも、XY変位15 mm、絶対Z変位15 mm、速度
500 mm/s、ワークスペース逸脱またはステレオ計測消失で回復音場へ切り替える。
この試行は`invalid`として残る。

## 0.9、1.0の確認

低い強度で動作と自動回復を確認してから、短いパルス・少数回で追加する。

```powershell
.\venv_autd3\Scripts\python.exe tracking\experiments\acoustic_force_step_response.py `
  --intensities "0.8,0.9,1.0" `
  --z-offsets-mm 0 `
  --repeats 3 `
  --pulse-ms 50 `
  --operator "Hiroto Yoshida"
```

問題がなければ各条件10～20回へ増やす。位置依存性は次のように測る。
負数を含む引数は`=`形式にする。

```powershell
.\venv_autd3\Scripts\python.exe tracking\experiments\acoustic_force_step_response.py `
  --intensities "0.5,0.6,0.7,0.8,0.9,1.0" `
  --z-offsets-mm=-10,-5,0,5,10 `
  --repeats 10 `
  --operator "Hiroto Yoshida"
```

## 出力

実行ごとに次のディレクトリが作られる。

```text
tracking/acoustic_force_data/session_YYYYMMDD_HHMMSS_PID/
```

- `session.json`: 実験条件、実カメラ画像サイズ、校正ファイル、条件順
- `step_samples.csv`: 安定保持区間とステップ中の画素・XYZ時系列
- `trial_summary.csv`: 各試行の加速度、音響力、標準誤差、位置分解能

## 解析

```powershell
.\venv_autd3\Scripts\python.exe tracking\analysis\analyze_acoustic_force_steps.py `
  --session-dir tracking\acoustic_force_data\session_YYYYMMDD_HHMMSS_PID
```

解析結果はセッション内の`analysis`に作られる。

- `force_by_condition.csv`: 条件別平均音響力と試行間95%信頼区間
- `adjacent_intensity_differences.csv`: 隣接強度の力差を識別できたか
- `force_vs_intensity.png`: 強度―鉛直力曲線

ステップ応答法が利用可能かは、少なくとも次を確認する。

- 有効試行が各条件で5回以上ある。
- 二次フィットのZ残差が安定保持時の位置ばらつきと同程度である。
- 強度を上げると平均上向き力が概ね増加する。
- 隣接する強度間の力差が試行間95%信頼区間より大きい。
- 0.6付近の力が、静止球に必要な`mg - F_b`と大きく矛盾しない。

最後の条件は校正の独立な整合性確認になる。質量0.53 g、直径38 mm、
空気密度1.204 kg/m³なら、浮力は約0.34 mN、静止に必要な上向き音響力は
約4.86 mNである。
