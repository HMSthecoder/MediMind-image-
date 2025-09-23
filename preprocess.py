import os
import cv2
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================================
# Helper Functions
# ================================

def apply_CLAHE(img):
    """Apply CLAHE to enhance contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def preprocess_and_save(image_path, save_path, size=(224,224), use_clahe=False):
    """Resize, optionally apply CLAHE, and save image as jpg."""
    img = cv2.imread(image_path)
    if img is None:
        return False  # skip corrupted image

    if use_clahe:
        img = apply_CLAHE(img)

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return True


def split_and_save_dataset(df, image_dir, output_dir, use_clahe=False):
    """Split dataset into train/val/test and preprocess images based on CSV labels."""

    X = df["id_code"].values
    y = df["diagnosis"].values  # labels: 0-4

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    for split, (X_split, y_split) in splits.items():
        print(f"\nProcessing {split} set ({len(X_split)} images)...")

        for img_id, label in tqdm(zip(X_split, y_split), total=len(X_split)):
            src_path = os.path.join(image_dir, f"{img_id}.png")  # APTOS uses .png
            if not os.path.exists(src_path):
                src_path = os.path.join(image_dir, f"{img_id}.jpg")  # fallback for .jpg
                if not os.path.exists(src_path):
                    continue  # skip missing files

            # Save to correct folder
            save_folder = os.path.join(output_dir, split, str(label))
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{img_id}.jpg")

            preprocess_and_save(src_path, save_path, size=(224,224), use_clahe=use_clahe)


# ================================
# Main Script
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images and organize by class labels from CSV.")
    parser.add_argument("--csv", type=str, default="data/train.csv",
                        help="Path to train.csv (APTOS/DDR). Must contain 'id_code' and 'diagnosis' (default: 'data/train.csv')")
    parser.add_argument("--image_dir", type=str, default="data",
                        help="Directory containing raw images (default: 'data')")
    parser.add_argument("--output_dir", type=str, default="data_processed",
                        help="Directory to save processed dataset (default: 'data_processed')")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE for contrast enhancement")
    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found. Please make sure the train.csv file exists.")
        exit()

    # Load CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples from {args.csv}")
    
    # Check if required columns exist
    if 'id_code' not in df.columns or 'diagnosis' not in df.columns:
        print("Error: CSV file must contain 'id_code' and 'diagnosis' columns.")
        exit()

    # Display class distribution
    print("\nClass distribution:")
    print(df['diagnosis'].value_counts().sort_index())

    # Process dataset
    split_and_save_dataset(df, args.image_dir, args.output_dir, use_clahe=args.clahe)

    print("\n Preprocessing complete! Dataset ready in:", args.output_dir)
