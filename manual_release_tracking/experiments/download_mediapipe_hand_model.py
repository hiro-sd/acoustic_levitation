from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "manual_release_tracking/models/hand_landmarker.task"


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"[INFO] Model already exists: {MODEL_PATH}")
        return
    print(f"[INFO] Downloading MediaPipe hand model from:\n{MODEL_URL}")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
