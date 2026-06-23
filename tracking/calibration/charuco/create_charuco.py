import cv2
from pathlib import Path

# 設定
# ボードのマス数（チェス盤のマス数）
SQUARES_X = 7
SQUARES_Y = 5

# 実寸 [mm]
SQUARE_LENGTH_MM = 25.0
MARKER_LENGTH_MM = 18.0

# OpenCV の辞書
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# 出力画像の解像度 [px]
# 印刷時に十分きれいに出るように少し大きめ
OUT_WIDTH_PX = 2000
OUT_HEIGHT_PX = 1400

# 余白 [px]
MARGIN_PX = 40

# 出力先
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "charuco_board_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "charuco_board_7x5.png"
OUTPUT_INFO = OUTPUT_DIR / "charuco_board_7x5_info.txt"


def main():
    # ArUco辞書
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # ChArUcoボード生成
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            dictionary,
        )
    else:
        # 古いOpenCV向けの互換パス
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            dictionary,
        )

    # 画像生成
    if hasattr(board, "generateImage"):
        board_img = board.generateImage(
            (OUT_WIDTH_PX, OUT_HEIGHT_PX),
            marginSize=MARGIN_PX,
            borderBits=1,
        )
    else:
        board_img = board.draw(
            (OUT_WIDTH_PX, OUT_HEIGHT_PX),
            marginSize=MARGIN_PX,
            borderBits=1,
        )

    # 保存
    cv2.imwrite(str(OUTPUT_PNG), board_img)

    # 設定情報も保存
    info_text = f"""ChArUco Board Information
========================
squaresX         : {SQUARES_X}
squaresY         : {SQUARES_Y}
squareLength_mm  : {SQUARE_LENGTH_MM}
markerLength_mm  : {MARKER_LENGTH_MM}
aruco_dictionary : DICT_4X4_50
output_width_px  : {OUT_WIDTH_PX}
output_height_px : {OUT_HEIGHT_PX}
margin_px        : {MARGIN_PX}

[IMPORTANT]
- 印刷時は「実際のサイズ」で印刷してください
- 「ページに合わせる」「拡大縮小」はOFFにしてください
- 印刷後に定規で1マスの実寸を測ってください
- キャリブレーションコードでは、その実測値を squareLength として使ってください
"""
    OUTPUT_INFO.write_text(info_text, encoding="utf-8")

    print(f"[SUCCESS] ChArUcoボード画像を保存しました: {OUTPUT_PNG}")
    print(f"[SUCCESS] 設定情報を保存しました: {OUTPUT_INFO}")


if __name__ == "__main__":
    main()
