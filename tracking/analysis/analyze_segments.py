#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30秒ごとのセグメントに分割して、各セグメント内のdx,dy,dzの標準偏差を計算。
FIXEDとPID各5回（合計30個のセグメント）の標準偏差を出力する。
"""

import os
import sys
import pandas as pd
import numpy as np

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "stability_log.csv")


def main():
    # ログファイルを読み込む
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] ファイルが見つかりません: {LOG_FILE}")
        sys.exit(1)
    
    df = pd.read_csv(LOG_FILE)
    print(f"[INFO] ログファイルを読み込みました: {len(df)} 行")
    
    # 中心位置を取得（最初の行から）
    center_x = df.iloc[0]["center_x_mm"]
    center_y = df.iloc[0]["center_y_mm"]
    center_z = df.iloc[0]["center_z_mm"]
    print(f"[INFO] 中心位置: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    
    # dx, dy, dz を計算
    df["dx"] = df["x_mm"] - center_x
    df["dy"] = df["y_mm"] - center_y
    df["dz"] = df["z_mm"] - center_z
    
    # タイムスタンプでソート
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 30秒ごとのセグメントに分割
    segments = []
    segment_count = 0
    segment_start_idx = 0
    
    for idx in range(1, len(df)):
        time_diff = df.iloc[idx]["timestamp"] - df.iloc[segment_start_idx]["timestamp"]
        
        # 30秒以上経過したらセグメントを分割
        if time_diff >= 30.0:
            segment_data = df.iloc[segment_start_idx:idx]
            if len(segment_data) > 0:
                segment_count += 1
                mode = segment_data.iloc[0]["mode"]
                
                # dx, dy, dz の標準偏差を計算
                sigma_x = segment_data["dx"].std()
                sigma_y = segment_data["dy"].std()
                sigma_z = segment_data["dz"].std()
                
                segments.append({
                    "segment_id": segment_count,
                    "mode": mode,
                    "n_samples": len(segment_data),
                    "duration": time_diff,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "sigma_z": sigma_z,
                })
                
                print(f"Segment {segment_count:02d}: {mode:5s} - "
                      f"samples={len(segment_data):3d}, duration={time_diff:6.2f}s, "
                      f"sigma_x={sigma_x:7.4f}, sigma_y={sigma_y:7.4f}, sigma_z={sigma_z:7.4f}")
            
            segment_start_idx = idx
    
    # 最後のセグメント（残り）
    if segment_start_idx < len(df):
        segment_data = df.iloc[segment_start_idx:]
        if len(segment_data) > 0:
            segment_count += 1
            mode = segment_data.iloc[0]["mode"]
            time_diff = segment_data.iloc[-1]["timestamp"] - segment_data.iloc[0]["timestamp"]
            
            sigma_x = segment_data["dx"].std()
            sigma_y = segment_data["dy"].std()
            sigma_z = segment_data["dz"].std()
            
            segments.append({
                "segment_id": segment_count,
                "mode": mode,
                "n_samples": len(segment_data),
                "duration": time_diff,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "sigma_z": sigma_z,
            })
            
            print(f"Segment {segment_count:02d}: {mode:5s} - "
                  f"samples={len(segment_data):3d}, duration={time_diff:6.2f}s, "
                  f"sigma_x={sigma_x:7.4f}, sigma_y={sigma_y:7.4f}, sigma_z={sigma_z:7.4f}")
    
    # 結果をDataFrameに変換して保存
    segments_df = pd.DataFrame(segments)
    out_csv = os.path.join(LOG_DIR, "segments_sigma.csv")
    segments_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print()
    print("=" * 80)
    print(f"[INFO] セグメント別標準偏差を保存しました: {out_csv}")
    print(f"[INFO] 合計 {segment_count} セグメント (FIXED: {len(segments_df[segments_df['mode'] == 'FIXED'])}, "
          f"PID: {len(segments_df[segments_df['mode'] == 'PD'])})")
    print("=" * 80)
    
    # 統計情報を表示
    print()
    print("[INFO] 統計サマリー")
    print("=" * 80)
    for mode in ["FIXED", "PD"]:
        sub = segments_df[segments_df["mode"] == mode]
        if len(sub) > 0:
            label = "固定音場 (FIXED)" if mode == "FIXED" else "制御あり (PID)"
            print(f"\n--- {label} ---")
            print(f"  セグメント数: {len(sub)}")
            print(f"  sigma_x: mean={sub['sigma_x'].mean():.4f}, std={sub['sigma_x'].std():.4f}")
            print(f"  sigma_y: mean={sub['sigma_y'].mean():.4f}, std={sub['sigma_y'].std():.4f}")
            print(f"  sigma_z: mean={sub['sigma_z'].mean():.4f}, std={sub['sigma_z'].std():.4f}")


if __name__ == "__main__":
    main()
