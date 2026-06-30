"""
Полностью очищает датасет: удаляет все файлы в папках images/ и lidar/,
и стирает содержимое labels.csv (оставляет только заголовок).

Использование:
    python scripts/clean_dataset.py
    python scripts/clean_dataset.py --dry-run   # только показать что будет удалено
"""
import os
import glob
import argparse
import pandas as pd

DATA_DIR   = os.path.join("data", "train")
CSV_PATH   = os.path.join(DATA_DIR, "labels.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LIDAR_DIR  = os.path.join(DATA_DIR, "lidar")

def count_files(directory, pattern="*"):
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, pattern)))

def delete_all_files(directory):
    if not os.path.exists(directory):
        return 0
    files = glob.glob(os.path.join(directory, "*"))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    return len(files)

def main(dry_run=False):
    img_count = count_files(IMAGES_DIR)
    lid_count = count_files(LIDAR_DIR)

    csv_rows = 0
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        csv_rows = len(df)

    print(f"Найдено: {csv_rows} строк в CSV, {img_count} файлов в images/, {lid_count} файлов в lidar/")

    if dry_run:
        print("--- DRY RUN --- (файлы не удалены)")
        return

    deleted_img = delete_all_files(IMAGES_DIR)
    deleted_lid = delete_all_files(LIDAR_DIR)

    # Очищаем CSV — оставляем только заголовок
    if os.path.exists(CSV_PATH):
        df_empty = df.iloc[0:0]
        df_empty.to_csv(CSV_PATH, index=False)
        print(f"CSV очищен (заголовок сохранён)")
    else:
        print("labels.csv не найден, пропускаем")

    print(f"Удалено: {deleted_img} файлов из images/, {deleted_lid} файлов из lidar/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать что будет удалено, не удалять')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
