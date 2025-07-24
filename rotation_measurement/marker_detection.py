from __future__ import annotations
import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 円検出（橙ライン）
# ---------------------------------------------------------------------------

def detect_circle_from_orange(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 30:
        return None
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return int(x), int(y), int(r)

# ---------------------------------------------------------------------------
# カラー設定
# ---------------------------------------------------------------------------

HSV_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray] | List[Tuple[np.ndarray, np.ndarray]]] = {
    # 新画像セット向け: 暗め赤・黒に近い青
    "red": [
        (np.array([0, 25, 40]),  np.array([10, 255, 255])),
        (np.array([150, 25, 40]), np.array([180, 255, 255])),
    ],
    "blue": (
        np.array([90, 30, 10]),
        np.array([130, 255, 120]),
    ),
}

COLOR_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
}

# ---------------------------------------------------------------------------
# RGB 比率ヘルパ
# ---------------------------------------------------------------------------

def rgb_ratio_mask(img_bgr: np.ndarray, color: str) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    if color == "red":
        # 暗めの赤も拾う
        mask = (r > g + 5) & (r > b + 5) & (r > 30)
    else:  # blue（黒マーカーも含める）
        mask = (r < 60) & (g < 60) & (b < 60)
    return mask.astype(np.uint8) * 255

# ---------------------------------------------------------------------------
# マーカー検出
# ---------------------------------------------------------------------------

def detect_color_marker(hsv: np.ndarray, mask_circle: np.ndarray, color: str):
    ranges = HSV_RANGES[color]
    if isinstance(ranges, list):
        mask1 = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
        mask2 = cv2.inRange(hsv, ranges[1][0], ranges[1][1])
        mask_color = cv2.bitwise_or(mask1, mask2)
    else:
        lo, hi = ranges
        mask_color = cv2.inRange(hsv, lo, hi)

    # ハイライト除去 (彩度<10)
    if color == "red":
        low_sat = cv2.inRange(hsv[:, :, 1], 0, 10)
        mask_color = cv2.bitwise_and(mask_color, cv2.bitwise_not(low_sat))

    # 円内マスク適用
    mask = cv2.bitwise_and(mask_color, mask_circle)

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.dilate(mask, kernel, 1)

    # 最大輪郭
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5:        # 旧値 10 → 5 に緩和
        return None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def process_frames(input_dir: Path, output_dir: Path, output_csv: Path):
    ensure_dir(output_dir)
    files = sorted(
        glob.glob(str(input_dir / "*.jpg")) + glob.glob(str(input_dir / "*.png")),
        key=natural_sort_key,
    )
    if not files:
        raise FileNotFoundError("入力ディレクトリに画像がありません")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cx", "cy", "radius", "red_x", "red_y", "blue_x", "blue_y"])

        for idx, file in enumerate(files):
            img = cv2.imread(file)
            if img is None:
                continue
            circle = detect_circle_from_orange(img)
            if circle is None:
                continue
            cx, cy, r = circle

            mask_circle = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(mask_circle, (cx, cy), r - 2, 255, -1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            coords = {c: detect_color_marker(hsv, mask_circle, c) for c in ("red", "blue")}
            writer.writerow([
                idx,
                cx,
                cy,
                r,
                *(coords["red"] or (-1, -1)),
                *(coords["blue"] or (-1, -1)),
            ])

            # デバッグ描画
            dbg = img.copy()
            cv2.circle(dbg, (cx, cy), r, (0, 165, 255), 2)
            for col, p in coords.items():
                if p is not None:
                    cv2.circle(dbg, p, 6, COLOR_BGR[col], -1)
            cv2.imwrite(str(output_dir / f"frame{idx:04d}.jpg"), dbg)

            if idx % 100 == 0:
                print(f"progress {idx}/{len(files)}")

    print("done →", output_csv)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="赤=HSV∧RGB, 青=HSV 専用 マーカー検出")
    ap.add_argument("--input_dir", type=str, default="/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ouhuku/detected_circles_ouhuku")
    ap.add_argument("--output_dir", type=str, default="/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ouhuku/detected_markers_ouhuku")
    ap.add_argument("--output_csv", type=str, default="/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ouhuku/markers_ouhuku.csv")
    args = ap.parse_args()

    process_frames(Path(args.input_dir), Path(args.output_dir), Path(args.output_csv))


if __name__ == "__main__":
    main()