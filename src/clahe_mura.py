import os
import cv2
import logging
from tqdm import tqdm

# -------- CONFIG --------
INPUT_ROOT = r"C:\Users\preeti\Desktop\x-rayData\muramskxrays\MURA-v1.1\MURA-v1.1\train\XR_ELBOW"
OUTPUT_ROOT = r"C:\Users\preeti\Desktop\x-rayData\muramskxrays\MURA-v1.1\MURA-v1.1\clahe_train\XR_ELBOW"
LOG_FILE = "logs/mura_processing.log"

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
# ------------------------


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s"
    )


def process_dataset():
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

    all_files = []

    for root, _, files in os.walk(INPUT_ROOT):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    print(f"Found {len(all_files)} images")

    for img_path in tqdm(all_files, desc="Processing MURA dataset"):
        try:
            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            output_path = os.path.join(OUTPUT_ROOT, relative_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                logging.error(f"Failed to read {img_path}")
                continue

            enhanced = clahe.apply(img)

            cv2.imwrite(output_path, enhanced)

        except Exception as e:
            logging.exception(f"Error processing {img_path}: {e}")

    print("✅ CLAHE applied to entire dataset")

if __name__ == "__main__":
    setup_logging()
    process_dataset()