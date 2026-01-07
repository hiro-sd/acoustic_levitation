from ultralytics import YOLO

def main():
    # 1. モデルのロード
    # 'yolo11n.pt' は最も軽量(nano)なモデルです。
    # 精度を上げたい場合は 'yolo11s.pt' (small) や 'yolo11m.pt' (medium) に変更します。
    model = YOLO('yolo11n.pt')

    # 2. 学習の実行
    # data: data.yamlへのパス
    # epochs: 学習回数（最初は30~50回で様子見、本番は100回以上）
    # imgsz: 画像サイズ（基本は640）
    results = model.train(
        data=' /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/tracking/datasets/data.yaml', # ← 解凍したフォルダ内のdata.yamlのパスを指定
        epochs=50,
        imgsz=640,
        device='mps'  # MacのGPU(Metal)を使う指定。エラーが出る場合は 'cpu' に変更
    )

if __name__ == '__main__':
    main()