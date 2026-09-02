# Acoustic Levitation Research

音響浮揚による軽量物体の浮揚・計測・フィードバック制御に関する修士研究用リポジトリです。AUTD3による音場生成、2台のXIMEAカメラを用いたステレオ3D追跡、カメラ校正、安定性解析、回転計測、手放し時の自動捕捉実験をまとめています。

> [!CAUTION]
> 実機用コードには、AUTD3、TwinCAT、XIMEAカメラ、および実験装置固有の校正値・シリアル番号が含まれます。実行前に装置構成と `tracking/core/config.py` を確認し、安全に音場を停止できる状態で使用してください。

## ディレクトリ構成

| パス | 内容 |
| --- | --- |
| `tracking/` | ステレオ3D追跡、フィードバック制御、AUTD送信、校正、解析、テスト |
| `tracking/core/` | 追跡・制御アプリケーションから共用する実装 |
| `tracking/experiments/` | ステレオフィードバック、多焦点音場、XY力推定などの実験入口 |
| `tracking/calibration/` | ChArUcoによる内部・ステレオ校正とカメラ座標系からAUTD座標系への変換 |
| `tracking/analysis/` | 安定性、FFT、PSD、区間別統計の解析コードと結果 |
| `manual_release_tracking/` | MediaPipeによる手放し検出と、落下する球の自動捕捉・保持実験 |
| `balloon_levitation/` | 半球型紙風船の水平・鉛直移動と回転の実験 |
| `styrol_levitation/` | 発泡スチロール球向けの周回・往復音場実験 |
| `rotation_measurement/` | 撮影画像からのマーカー検出、角速度・回転軸の推定 |
| `graph/` | AUTD配置やSTM軌道など、論文・発表用図の生成 |
| `others/` | 多焦点・傾斜音場などの探索的な実験コード |

## 環境構築

リポジトリ直下をカレントディレクトリにして実行します。多くの設定パスがリポジトリ直下からの相対パスになっているためです。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windowsでは仮想環境の有効化に次を使用します。

```powershell
.venv\Scripts\Activate.ps1
```

実機で追跡アプリケーションを動かす場合は、上記に加えて次の準備が必要です。

- AUTD3とTwinCATを利用できる環境
- XIMEA SDKおよびSDK付属のPython API
- キー入力検出用の `keyboard` パッケージ（`python -m pip install keyboard`）
- 使用するカメラに対応した校正ファイル

XIMEA Python APIの場所、カメラのシリアル番号、露光時間などは `tracking/core/config.py` の `AppConfig` で設定します。

手放し検出実験では、追加でMediaPipeと公式モデルを準備します。

```bash
python -m pip install -r manual_release_tracking/requirements-mediapipe.txt
python manual_release_tracking/experiments/download_mediapipe_hand_model.py
```

## カメラ校正

現在の追跡処理では、主に次のファイルを使用します。

- `tracking/calibration/intrinsic_charuco_cam1.npz`
- `tracking/calibration/intrinsic_charuco_cam2.npz`
- `tracking/calibration/stereo_charuco_calibration_result.npz`
- `tracking/calibration/stereo_camera_to_autd.npz`

再校正に使用する主なスクリプトは以下です。

```bash
python tracking/calibration/charuco/create_charuco.py
python tracking/calibration/camera/capture_stereo_charuco.py
python tracking/calibration/camera/fit_stereo_charuco.py
python tracking/calibration/camera/validate_stereo_charuco.py
python tracking/calibration/camera/fit_camera_to_autd.py
```

カメラの向き、ChArUcoボードの寸法、カメラのシリアル番号を変えた場合は、既存の校正結果をそのまま使用しないでください。カメラ座標系からAUTD座標系への変換には `tracking/calibration/camera_to_autd_points.csv` の対応点を使用します。

## 主な実験の実行

### ステレオ追跡とフィードバック制御

円周上を走査する単焦点STMを用いる主要な実験です。

```bash
python tracking/experiments/stereo_feedback.py
```

円周上の複数焦点を同時生成する試験は次の入口を使用します。

```bash
python tracking/experiments/stereo_feedback_static_multi.py
```

主なキー操作は以下です。

- `Enter`: フィードバック制御の開始・一時停止
- `Esc`: 終了
- `P`: 遅延補償の有効・無効を切り替え
- `D`: 自動デモ軌道の開始・停止（有効な設定の場合）
- `L`: 安定性ログの記録開始
- 矢印キー、`Page Up`、`Page Down`: 基準位置の移動（有効な設定の場合）

### XY方向の力推定

```bash
python tracking/experiments/xy_force_characterization.py
```

実験手順と出力CSVの詳細は [`tracking/experiments/README_xy_force.md`](tracking/experiments/README_xy_force.md) を参照してください。

### 手放し検出と自動捕捉

```bash
python manual_release_tracking/experiments/mediapipe_auto_release_hold.py
```

状態遷移、安全停止条件、ログの詳細は [`manual_release_tracking/README.md`](manual_release_tracking/README.md) を参照してください。

### その他の実験・解析

- 紙風船: `balloon_levitation/` 内の各スクリプト
- 発泡スチロール球: `styrol_levitation/` 内の各スクリプト
- 回転計測: `rotation_measurement/marker_detection.py`、`angular_velocity.py`、`rotation_axis.py`
- 安定性解析: `tracking/analysis/analyze_stability.py`
- 周波数解析: `tracking/analysis/analyze_fft.py`、`analyze_psd.py`
- 図の生成: `graph/` 内の各スクリプト

これらの一部には入力ファイルや実験条件がコード内の既定値として記述されています。実行前に対象スクリプトのパスとパラメータを確認してください。CLI対応スクリプトは `--help` で引数を確認できます。

## テスト

ハードウェアを接続せずに、追跡・制御ロジックと手放し検出ロジックの単体テストを実行できます。

```bash
cd tracking
python -m unittest discover -s tests -p 'test_*.py'
cd ..
python -m unittest discover -s manual_release_tracking/tests -p 'test_*.py'
```

## 実験データの管理方針

再現性とリポジトリ容量の両方を保つため、次の区分を推奨します。

- Gitで管理する: ソースコード、設定、校正結果、小容量の代表CSV・図、実験条件を説明するREADME
- Gitで管理しない: 生動画、連番の校正画像、学習データセット、繰り返し生成される大容量ログ
- 各結果には実験日、条件、元データの保存場所、対応するコミットIDを記録する

大容量データはリポジトリ外の研究データ領域へ保存し、READMEまたは実験メモから場所を参照できるようにすると、コードと測定結果の対応を失わずに管理できます。
