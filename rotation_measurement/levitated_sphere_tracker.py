import cv2
import numpy as np
import os
import glob

# ball_spinフォルダのパス
folder_path = "/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ball_spin"

# 出力フォルダを作成
output_folder = os.path.join(os.path.dirname(folder_path), "detected_circles")
os.makedirs(output_folder, exist_ok=True)

# サポートする画像形式
image_extensions = ['*.jpg']

# フォルダ内の全ての画像ファイルを取得
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

print(f"Found {len(image_files)} images in the folder.")

for i, image_path in enumerate(image_files):
    print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
    
    # 画像を読み込み
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"  Error: Could not load {image_path}")
        continue
    
    # 画像サイズを取得
    h, w = img.shape[:2]
    
    # 中央部をクロップ（元画像の50%の領域）
    crop_ratio = 0.5
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # 中央部のみを切り出し
    cropped_img = img[start_y:end_y, start_x:end_x]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # 前処理でノイズ除去と円検出の向上
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 複数のパラメータ設定で円検出を試行
    circles = None
    param_sets = [
        # (param1, param2, minRadius, maxRadius)
        (100, 60, 0, 0),      # 元の設定
        (80, 50, 0, 0),       # より緩い設定
        (120, 70, 10, 200),   # より厳しい設定 + サイズ制限
        (60, 40, 0, 0),       # 最も緩い設定
    ]
    
    for param1, param2, minR, maxR in param_sets:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                  param1=param1, param2=param2, 
                                  minRadius=minR, maxRadius=maxR)
        if circles is not None:
            print(f"  Found circles with params: param1={param1}, param2={param2}")
            break
    
    if circles is None:
        print(f"  Warning: No circles detected in {os.path.basename(image_path)}")
        # 検出できなかった画像もコピーして確認用に保存
        output_filename = f"no_detection_{os.path.basename(image_path)}"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, img)
        continue
    
    circles_found = 0
    circles = np.uint16(np.around(circles))
    circles_found = len(circles[0])
    
    for circle in circles[0, :]:
        # クロップした座標を元画像の座標に変換
        original_x = circle[0] + start_x
        original_y = circle[1] + start_y
        
        # 元画像に円周を描画する
        cv2.circle(img, (original_x, original_y), circle[2], (0, 165, 255), 2)
        # 元画像に中心点を描画する
        # cv2.circle(img, (original_x, original_y), 2, (0, 0, 255), 3)
    
    print(f"  Detected {circles_found} circles")
    
    # 結果を保存
    output_filename = f"detected_{os.path.basename(image_path)}"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img)

print(f"Processing complete. Results saved to: {output_folder}")
