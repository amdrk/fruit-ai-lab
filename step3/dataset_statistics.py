from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# مسیر اصلی پروژه
BASE_DIR = Path(__file__).resolve().parent

# مسیر دیتاست
DATA_DIR = BASE_DIR / "data" / "processed"

# مسیر ذخیره نتایج
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULT_DIR / "dataset_statistics.csv"

# مجموعه‌های داده
SPLITS = ["train", "validation", "test"]

# پسوندهای مجاز تصاویر
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_accumulator():
    """ساخت متغیرهای موردنیاز برای محاسبه آمار"""
    return {
        "image_count": 0,
        "pixel_count": 0,
        "sum": np.zeros(3, dtype=np.float64),
        "sum_sq": np.zeros(3, dtype=np.float64)
    }


def add_image(stats, image):
    """اضافه کردن اطلاعات یک تصویر به آمار"""

    pixels = np.asarray(image, dtype=np.float64)

    stats["image_count"] += 1
    stats["pixel_count"] += pixels.shape[0] * pixels.shape[1]

    stats["sum"] += pixels.sum(axis=(0, 1))
    stats["sum_sq"] += np.square(pixels).sum(axis=(0, 1))


def calculate_statistics(stats, split, class_name):
    """محاسبه میانگین و واریانس"""

    count = stats["pixel_count"]

    if count == 0:
        return None

    mean = stats["sum"] / count

    variance = (stats["sum_sq"] / count) - np.square(mean)

    # جلوگیری از خطای احتمالی اعداد اعشاری
    variance = np.maximum(variance, 0)

    return {
        "Split": split,
        "Class": class_name,
        "Image_Count": stats["image_count"],
        "Pixel_Count": count,
        "Mean_R": round(mean[0], 3),
        "Mean_G": round(mean[1], 3),
        "Mean_B": round(mean[2], 3),
        "Variance_R": round(variance[0], 3),
        "Variance_G": round(variance[1], 3),
        "Variance_B": round(variance[2], 3)
    }


def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    results = []

    total_stats = create_accumulator()

    print("\nDataset Statistics")
    print("========================")

    for split in SPLITS:

        split_dir = DATA_DIR / split

        if not split_dir.exists():
            print(f"Warning: {split} folder not found.")
            continue

        split_stats = create_accumulator()

        print(f"\nProcessing: {split}")

        class_dirs = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir()
        )

        for class_dir in class_dirs:

            class_stats = create_accumulator()

            image_paths = [
                path for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            for image_path in image_paths:

                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        add_image(class_stats, img)

                except Exception as error:
                    print(f"Skipped: {image_path.name} ({error})")

            if class_stats["image_count"] > 0:

                results.append(
                    calculate_statistics(
                        class_stats,
                        split,
                        class_dir.name
                    )
                )

                # جمع آمار کلاس‌ها در آمار کل مجموعه
                for key in ["image_count", "pixel_count"]:
                    split_stats[key] += class_stats[key]
                    total_stats[key] += class_stats[key]

                for key in ["sum", "sum_sq"]:
                    split_stats[key] += class_stats[key]
                    total_stats[key] += class_stats[key]

                print(
                    f"{class_dir.name}: "
                    f"{class_stats['image_count']} images"
                )

        if split_stats["image_count"] > 0:

            results.append(
                calculate_statistics(
                    split_stats,
                    split,
                    "ALL_CLASSES"
                )
            )

    # آمار کل دیتاست
    if total_stats["image_count"] > 0:

        results.append(
            calculate_statistics(
                total_stats,
                "all",
                "ALL_CLASSES"
            )
        )

    if not results:
        raise ValueError("No valid images were found in the dataset.")

    # ذخیره نتایج در فایل CSV
    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========================")
    print("Statistics completed!")
    print(f"Total images: {total_stats['image_count']}")
    print(f"Results saved to: {OUTPUT_CSV}")

    print("\nStatistics Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()