"""Ekstraksi patch 48x48 untuk pipeline segmentasi retina (jurnal CAS-UNet).

Training : balanced random sampling, patch pembuluh dan background diimbangi.
Testing  : sliding window + stitching menjadi mask segmentasi utuh.

Citra input berasal dari hasil preprocessing batch:
    data/processed/variants/{train,test}/{variant}/<id>.png

Label dan mask FOV tetap diambil dari data mentah DRIVE:
    data/raw/DRIVE/training/1st_manual/<id>_manual1.gif
    data/raw/DRIVE/training/mask/<id>_training_mask.gif
    data/raw/DRIVE/test/mask/<id>_test_mask.gif
"""

from pathlib import Path
import argparse
import csv
import shutil

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "raw" / "DRIVE"
VARIANT_DIR = BASE_DIR / "data" / "processed" / "variants"
PROCESSED_ROOT = BASE_DIR / "data" / "processed" / "patches"

VARIANTS = ("gray", "gray_clahe", "gray_clahe_gamma", "gray_clahe_frangi")

PATCH_SIZE = 48
# Jurnal: 40.000 patch dari 40 citra latih = 1.000 patch per citra.
# Project ini hanya punya 20 citra latih DRIVE, jadi defaultnya 20.000 patch.
PATCHES_PER_IMAGE = 1000
VAL_RATIO = 0.2
SEED = 42
STRIDE = 24  # sliding window 50% overlap untuk inference

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Citra gagal disimpan: {path}")


def _variant_path(split, variant, stem):
    path = VARIANT_DIR / split / variant / f"{stem}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Hasil preprocessing tidak ditemukan: {path}. "
            "Jalankan preprocessing_pipeline.py terlebih dahulu."
        )
    return path


def _window_sums(integral, ys, xs, size):
    """Jumlah nilai pada jendela size×size yang pojok kiri-atasnya (ys, xs)."""
    return (
        integral[ys + size, xs + size]
        - integral[ys, xs + size]
        - integral[ys + size, xs]
        + integral[ys, xs]
    )


def get_training_samples():
    """Pasangkan citra latih dengan manual (ground truth) dan mask FOV."""
    image_dir = DATASET_DIR / "training" / "images"
    manual_dir = DATASET_DIR / "training" / "1st_manual"
    mask_dir = DATASET_DIR / "training" / "mask"

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder citra latih tidak ditemukan: {image_dir}")

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = image_path.stem.replace("_training", "")
        manual_path = manual_dir / f"{image_id}_manual1.gif"
        mask_path = mask_dir / f"{image_id}_training_mask.gif"

        for path in (manual_path, mask_path):
            if not path.exists():
                raise FileNotFoundError(f"File pendamping tidak ditemukan: {path}")

        samples.append((image_id, image_path.stem, image_path, manual_path, mask_path))

    if not samples:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di: {image_dir}")

    return samples


def get_test_samples():
    """Pasangkan citra uji dengan mask FOV (tanpa label pembuluh)."""
    image_dir = DATASET_DIR / "test" / "images"
    mask_dir = DATASET_DIR / "test" / "mask"

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder citra uji tidak ditemukan: {image_dir}")

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = image_path.stem.replace("_test", "")
        mask_path = mask_dir / f"{image_id}_test_mask.gif"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask FOV tidak ditemukan: {mask_path}")

        samples.append((image_path.stem, image_path, mask_path))

    if not samples:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di: {image_dir}")

    return samples


def valid_positions(mask, patch_size=PATCH_SIZE):
    """Posisi kiri-atas patch yang seluruh pikselnya berada di dalam FOV."""
    height, width = mask.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Ukuran citra {height}x{width} lebih kecil dari patch {patch_size}x{patch_size}"
        )

    outside = (mask == 0).astype(np.uint8)
    integral = cv2.integral(outside, cv2.CV_32S)

    ys, xs = np.meshgrid(
        np.arange(height - patch_size + 1),
        np.arange(width - patch_size + 1),
        indexing="ij",
    )
    outside_counts = _window_sums(integral, ys, xs, patch_size)

    valid = outside_counts == 0
    return ys[valid], xs[valid]


def vessel_counts(manual, ys, xs, patch_size=PATCH_SIZE):
    """Banyak piksel pembuluh di setiap kandidat patch."""
    vessel = (manual > 0).astype(np.uint8)
    integral = cv2.integral(vessel, cv2.CV_32S)
    return _window_sums(integral, ys, xs, patch_size)


def balanced_positions(manual, mask, n_patches, rng, patch_size=PATCH_SIZE):
    """Ambil patch berpusat di pembuluh dan di background secara seimbang.

    Catatan: pada DRIVE hampir tidak ada patch 48x48 yang seluruhnya background,
    karena jaringan pembuluh sangat rapat. Keseimbangan oleh karena itu diukur
    dari piksel pusat patch, bukan dari ketiadaan piksel pembuluh di seluruh patch.

    Mengembalikan list (y, x, vessel_ratio).
    """
    ys, xs = valid_positions(mask, patch_size)
    if len(ys) == 0:
        raise ValueError("Tidak ada posisi patch yang seluruhnya berada di dalam FOV")

    counts = vessel_counts(manual, ys, xs, patch_size)

    center_offset = patch_size // 2
    center_is_vessel = manual[ys + center_offset, xs + center_offset] > 0
    vessel_idx = np.flatnonzero(center_is_vessel)
    background_idx = np.flatnonzero(~center_is_vessel)

    if len(vessel_idx) == 0:
        raise ValueError("Tidak ditemukan patch yang pusatnya berada di pembuluh")
    if len(background_idx) == 0:
        raise ValueError("Tidak ditemukan patch yang pusatnya berada di background")

    n_vessel = n_patches // 2
    n_background = n_patches - n_vessel

    def pick(indices, n):
        replace = n > len(indices)
        return rng.choice(indices, size=n, replace=replace)

    selected = np.concatenate(
        [pick(vessel_idx, n_vessel), pick(background_idx, n_background)]
    )
    rng.shuffle(selected)

    area = float(patch_size * patch_size)
    return [
        (int(ys[i]), int(xs[i]), float(counts[i]) / area)
        for i in selected
    ]


def sliding_window_positions(shape, patch_size=PATCH_SIZE, stride=STRIDE):
    """Posisi kiri-atas sliding window yang menutup seluruh citra uji."""
    height, width = shape[:2]
    y_last = height - patch_size
    x_last = width - patch_size

    ys = list(range(0, y_last + 1, stride))
    xs = list(range(0, x_last + 1, stride))
    if ys[-1] != y_last:
        ys.append(y_last)
    if xs[-1] != x_last:
        xs.append(x_last)

    return [(y, x) for y in ys for x in xs]


def stitch_patches(predictions, positions, output_shape, patch_size=PATCH_SIZE):
    """Gabungkan prediksi patch yang tumpang tindih dengan rata-rata."""
    height, width = output_shape[:2]
    accumulator = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for prediction, (y, x) in zip(predictions, positions):
        patch = np.asarray(prediction, dtype=np.float32)
        if patch.ndim == 3 and patch.shape[-1] == 1:
            patch = patch[..., 0]
        if patch.shape != (patch_size, patch_size):
            raise ValueError(
                f"Ukuran prediksi {patch.shape} tidak sesuai {patch_size}x{patch_size}"
            )

        accumulator[y:y + patch_size, x:x + patch_size] += patch
        weights[y:y + patch_size, x:x + patch_size] += 1.0

    return accumulator / np.maximum(weights, 1e-6)


def _reset_split_dirs(root, splits):
    for split in splits:
        split_dir = root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "masks").mkdir(parents=True)


def extract_training_patches(
    patches_per_image=PATCHES_PER_IMAGE,
    val_ratio=VAL_RATIO,
    seed=SEED,
    variant="gray_clahe",
):
    """Buat patch latih/validasi beserta mask labelnya dari suatu varian."""
    if variant not in VARIANTS:
        raise ValueError(f"Varian tidak dikenal: {variant}")

    patch_root = PROCESSED_ROOT / variant
    samples = get_training_samples()
    rng = np.random.default_rng(seed)
    _reset_split_dirs(patch_root, ("train", "val"))

    manifest_path = patch_root / "train_val_manifest.csv"
    counts = {"train": 0, "val": 0}

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "patch", "y", "x", "vessel_ratio"])

        for image_id, variant_stem, _, manual_path, mask_path in samples:
            variant_path = _variant_path("train", variant, variant_stem)
            image = cv2.imread(str(variant_path), cv2.IMREAD_GRAYSCALE)
            manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
            fov_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or manual is None or fov_mask is None:
                raise ValueError(f"Gagal membaca data varian '{variant}' untuk {image_id}")

            positions = balanced_positions(
                manual, fov_mask, patches_per_image, rng
            )

            for index, (y, x, ratio) in enumerate(positions):
                split = "val" if rng.random() < val_ratio else "train"
                patch_name = f"{image_id}_p{index:04d}.png"

                patch_image = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                patch_mask = (manual[y:y + PATCH_SIZE, x:x + PATCH_SIZE] > 0).astype(
                    np.uint8
                ) * 255

                _save_image(patch_root / split / "images" / patch_name, patch_image)
                _save_image(patch_root / split / "masks" / patch_name, patch_mask)

                writer.writerow([split, image_id, patch_name, y, x, f"{ratio:.4f}"])
                counts[split] += 1

            print(f"[{variant}] {image_id}: {len(positions)} patch")

    print(
        f"[{variant}] Ekstraksi training selesai: "
        f"{counts['train']} patch train, {counts['val']} patch val"
    )
    print(f"[{variant}] Manifest: {manifest_path}")


def extract_test_patches(variant="gray_clahe"):
    """Buat patch sliding window inferensi dari suatu varian (tanpa label)."""
    if variant not in VARIANTS:
        raise ValueError(f"Varian tidak dikenal: {variant}")

    patch_root = PROCESSED_ROOT / variant
    samples = get_test_samples()
    split_dir = patch_root / "test"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    (split_dir / "images").mkdir(parents=True)

    manifest_path = patch_root / "test_positions.csv"
    total = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "patch", "y", "x"])

        for image_id, _, mask_path in samples:
            variant_path = _variant_path("test", variant, image_id)
            image = cv2.imread(str(variant_path), cv2.IMREAD_GRAYSCALE)
            fov_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or fov_mask is None:
                raise ValueError(f"Gagal membaca data varian '{variant}' untuk {image_id}")

            # Patch di luar FOV tetap disimpan supaya urutan stitching utuh;
            # prediksi pada area tersebut nanti bisa dinolkan dengan mask FOV.
            positions = sliding_window_positions(image.shape)

            for index, (y, x) in enumerate(positions):
                patch_name = f"{image_id}_p{index:04d}.png"
                patch_image = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                _save_image(split_dir / "images" / patch_name, patch_image)
                writer.writerow([image_id, patch_name, y, x])
                total += 1

            print(f"[{variant}] {image_id}: {len(positions)} patch sliding window")

    print(f"[{variant}] Ekstraksi testing selesai: {total} patch")
    print(f"[{variant}] Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ekstraksi patch 48x48 dari hasil preprocessing DRIVE"
    )
    parser.add_argument(
        "--mode",
        choices=("all", "train", "test"),
        default="all",
        help="bagian yang dibuat (default: keduanya)",
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + VARIANTS,
        default="all",
        help="varian preprocessing yang dipakai (default: semua)",
    )
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=PATCHES_PER_IMAGE,
        help="jumlah patch per citra latih (default: 1000)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="proporsi patch untuk validasi (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="seed random (default: 42)",
    )

    args = parser.parse_args()

    if args.variant == "all":
        if PROCESSED_ROOT.exists():
            shutil.rmtree(PROCESSED_ROOT)
        variants = VARIANTS
    else:
        variants = (args.variant,)

    for variant in variants:
        print(f"=== Varian: {variant} ===")
        if args.mode in ("all", "train"):
            extract_training_patches(
                args.patches_per_image, args.val_ratio, args.seed, variant
            )
        if args.mode in ("all", "test"):
            extract_test_patches(variant)
