import time
import sys
from pathlib import Path
import cv2
import numpy as np

# 2台のカメラで同時撮影するためのスクリプト

# XIMEA設定
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[ERROR] ximea モジュールが見つかりません。")
    sys.exit(1)

CAM1_SN = "43430551"  # 真上カメラ (Left)
CAM2_SN = "43435351"  # 斜めカメラ (Right)

# 保存先ディレクトリ
SAVE_DIR_L = Path("./tracking/calibration/stereo_images/cam1")
SAVE_DIR_R = Path("./tracking/calibration/stereo_images/cam2")
SAVE_DIR_L.mkdir(parents=True, exist_ok=True)
SAVE_DIR_R.mkdir(parents=True, exist_ok=True)

EXPOSURE_US = 10000    # 露出時間
INTERVAL_SEC = 7.5     # 撮影間隔 (秒)
MAX_PAIRS = 20         # 合計撮影枚数

def main():
    cam1 = xiapi.Camera()
    cam2 = xiapi.Camera()

    try:
        print("[INFO] Opening cameras...")
        cam1.open_device_by_SN(CAM1_SN)
        cam2.open_device_by_SN(CAM2_SN)

        for cam in [cam1, cam2]:
            cam.set_imgdataformat('XI_RGB24')
            try:
                cam.enable_auto_wb()
            except AttributeError:
                pass
            cam.set_exposure(EXPOSURE_US)
            cam.set_trigger_source('XI_TRG_SOFTWARE')

        img1 = xiapi.Image()
        img2 = xiapi.Image()

        cam1.start_acquisition()
        cam2.start_acquisition()

        print("\n" + "="*40)
        print("  ステレオ画像 自動収集ツール")
        print(f"  設定: {INTERVAL_SEC}秒おきに計{MAX_PAIRS}枚を自動撮影します")
        print("  [Esc/Q] : 途中で終了")
        print("="*40 + "\n")

        pair_count = 0
        last_save_time = time.time()
        
        window_name = "Stereo Auto Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while pair_count < MAX_PAIRS:
            # ソフトウェアトリガーを発火
            cam1.set_trigger_software(1)
            cam2.set_trigger_software(1)

            # 画像取得
            cam1.get_image(img1, timeout=1000)
            cam2.get_image(img2, timeout=1000)

            # NumPy配列に変換
            frame_l = img1.get_image_data_numpy()
            frame_r = img2.get_image_data_numpy()
            
            # カラー変換 (RGB -> BGR)
            frame_l_bgr = frame_l.copy() # cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
            frame_r_bgr = frame_r.copy() # cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

            # プレビュー表示の作成
            h, w = frame_l_bgr.shape[:2]
            preview_scale = 0.5
            combined = np.hstack((frame_l_bgr, frame_r_bgr))
            display_img = cv2.resize(combined, (int(w * 2 * preview_scale), int(h * preview_scale)))
            
            # 次の撮影までのカウントダウンを表示
            elapsed = time.time() - last_save_time
            remaining = max(0, INTERVAL_SEC - elapsed)
            cv2.putText(display_img, f"Next: {remaining:.1f}s | Progress: {pair_count}/{MAX_PAIRS}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_img)

            # 自動保存ロジック
            if elapsed >= INTERVAL_SEC:
                fname_l = SAVE_DIR_L / f"img_{pair_count:02d}.png"
                fname_r = SAVE_DIR_R / f"img_{pair_count:02d}.png"
                
                cv2.imwrite(str(fname_l), frame_l_bgr)
                cv2.imwrite(str(fname_r), frame_r_bgr)
                
                print(f"[AUTO SAVE] {pair_count+1}/{MAX_PAIRS} を保存しました。")
                pair_count += 1
                last_save_time = time.time() # タイマーリセット

                # 保存時のフラッシュエフェクト
                flash = np.full_like(display_img, 255)
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)

            # 途中終了の受付
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] ユーザーによって中断されました。")
                break

        print(f"\n[FINISH] 全{pair_count}枚の撮影が完了しました。")

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cam1.stop_acquisition()
        cam2.stop_acquisition()
        cam1.close_device()
        cam2.close_device()
        cv2.destroyAllWindows()
        print("[INFO] カメラを終了しました。")

if __name__ == "__main__":
    main()