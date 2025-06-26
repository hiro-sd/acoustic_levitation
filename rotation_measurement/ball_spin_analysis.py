import cv2
import numpy as np
import pandas as pd
from math import atan2, degrees

# 球の位置と回転軸を抽出し、CSVに保存するファイル
def process_video(video_path: str, # 動画ファイルのパス
                  target_fps: int = 240, # 抽出対象のフレームレート
                  debug_preview: bool = False, # True の場合はデバッグ用のプレビューを表示
                  csv_out: str = "results.csv", # 出力するCSVファイルのパス
                  output_video_path: str = None): # 解析結果付き動画の出力パス
    cap = cv2.VideoCapture(video_path) # 動画を読み込む
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) # 元のフレームレートを取得
    if src_fps <= 0:
        raise ValueError("Source FPS could not be read.")

    step = max(1, round(src_fps / target_fps)) # 何フレームごとに処理するか
    frame_idx = 0
    records = []

    # VideoWriterの初期化
    writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step: # 抽出対象でなければスキップ
            frame_idx += 1
            continue

        # --- 前処理 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # グレースケールに変換
        gray = cv2.GaussianBlur(gray, (7, 7), 0) # ガウシアンブラーで平滑化

        # --- 球の検出（Hough変換） ---
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=50, param1=100, param2=30,
                                   minRadius=10, maxRadius=0) # 円を検出

        if circles is not None: # 円が検出された場合
            circles = np.uint16(np.around(circles[0])) # 円の座標と半径を取得
            # 半径が最大の円を「球」とみなす
            cx, cy, rad = max(circles, key=lambda c: c[2]) # 半径が最大の円を取得

            # --- 主軸推定 ---
            mask = np.zeros_like(gray) # マスクを作成
            cv2.circle(mask, (cx, cy), rad, 255, -1)     # 円形マスク
            roi = cv2.bitwise_and(gray, gray, mask=mask) # 円形マスクを適用
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE) # 輪郭を検出
            if cnts: # 輪郭が検出された場合
                largest = max(cnts, key=cv2.contourArea) # 面積が最大の輪郭を取得
                # 輪郭点の PCA で主成分を求める
                pts = largest.reshape(-1, 2).astype(np.float32)
                mean, eigvecs = cv2.PCACompute(pts, mean=np.empty((0))) # 主成分分析
                # 第 1 主成分の方向ベクトル
                v1 = eigvecs[0]
                theta = degrees(atan2(v1[1], v1[0])) # ラジアン→度(deg)

                # データ保存
                time_sec = frame_idx / src_fps # 元フレーム時刻
                records.append((time_sec, cx, cy, theta))

                # 可視化（プレビューまたは動画保存用）
                if debug_preview or writer is not None:
                    cv2.circle(frame, (cx, cy), rad, (0, 255, 0), 2)
                    p1 = (int(cx), int(cy)) # 円の中心
                    p2 = (int(cx + v1[0]*rad), int(cy + v1[1]*rad)) # 主軸の端点
                    cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.2)
                    # 角度を文字で表示
                    cv2.putText(frame, f"{theta:.1f} deg", (cx+10, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    if debug_preview:
                        cv2.imshow("preview", frame)
                        if cv2.waitKey(1) & 0xFF == 27: # Esc で中断
                            break
        # 動画として保存
        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # --- CSV 出力 ---
    df = pd.DataFrame(records, columns=["time[s]", "cx", "cy", "theta[deg]"])
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} rows to {csv_out}")
    if writer is not None:
        print(f"Saved video to {output_video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ball centroid & axis extractor")
    parser.add_argument("video", help="input video path")
    parser.add_argument("--fps", type=int, default=240,
                        help="target extraction fps (default 240)")
    parser.add_argument("--preview", action="store_true",
                        help="show debug preview")
    parser.add_argument("--csv", default="results.csv",
                        help="output CSV filename")
    parser.add_argument("--outvideo", default=None,
                        help="output video filename (with overlay)")
    args = parser.parse_args()
    process_video(args.video, args.fps, args.preview, args.csv, args.outvideo)
