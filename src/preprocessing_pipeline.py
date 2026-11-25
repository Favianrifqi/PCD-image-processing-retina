"""Batch preprocessing untuk pipeline segmentasi retina.

Variants:
    gray               : RGB -> grayscale (0,299R + 0,587G + 0,114B)
    gray_clahe         : grayscale -> CLAHE
    gray_clahe_gamma   : grayscale -> CLAHE -> gamma correction (gamma = 0,8)
    gray_clahe_frangi  : grayscale -> CLAHE -> median -> Frangi vesselness

Varian keempat mengikuti pendekatan Wiharto & Palgunadi (2019),
"Blood Vessels Segmentation in Retinal Fundus Image using Hybrid Method of
Frangi Filter, Otsu Thresholding and Morphology", IJACSA 10(6).

Output:
    data/processed/variants/{train,test}/{variant}/<id>.png
"""

from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np

from histogram_enhancement import gamma_correction, GAMMA


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "raw" / "DRIVE"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "variants"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
MEDIAN_KERNEL_SIZE = 3
FRANGI_SCALES = (1, 2, 3, 4, 5)
FRANGI_BETA = 0.5   # sensitivitas terhadap bentuk blob
FRANGI_GAMMA = 0.5  # sensitivitas terhadap kekuatan respons

VARIANTS = ("gray", "gray_clahe", "gray_clahe_gamma", "gray_clahe_frangi")
RAW_SPLIT = {"train": "training", "test": "test"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Citra gagal disimpan: {path}")


def _frangi_single_scale(inverted, sigma, beta, gamma):
    """Respons Frangi pada satu skala sigma."""
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    scale = float(sigma * sigma)

    fxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * scale
    fyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * scale
    fxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * scale

    trace = fxx + fyy
    determinant = fxx * fyy - fxy * fxy
    discriminant = np.sqrt(np.maximum(trace * trace - 4.0 * determinant, 0.0))

    lambda1 = 0.5 * (trace + discriminant)  # eigenvalue terbesar
    lambda2 = 0.5 * (trace - discriminant)  # eigenvalue terkecil

    abs_lambda2 = np.abs(lambda2)
    blob_ratio = np.divide(
        np.abs(lambda1),
        abs_lambda2,
        out=np.full_like(abs_lambda2, np.inf),
        where=abs_lambda2 > 1e-6,
    )
    shape = np.exp(-(blob_ratio * blob_ratio) / (2.0 * beta * beta))

    strength_arg = lambda1 * lambda1 + lambda2 * lambda2
    strength = 1.0 - np.exp(-strength_arg / (2.0 * gamma * gamma))

    response = shape * strength
    # Untuk struktur tubular terang, lambda2 harus negatif.
    response[lambda2 >= 0] = 0.0

    return response


def frangi_vesselness(
    image,
    scales=FRANGI_SCALES,
    beta=FRANGI_BETA,
    gamma=FRANGI_GAMMA,
):
    """Peta vesselness Frangi untuk pembuluh gelap pada citra fundus.

    Citra di-invers dulu supaya pembuluh yang gelap menjadi ridge terang,
    sesuai asumsi filter Frangi.
    """
    inverted = (255 - image.astype(np.float32)) / 255.0

    response = np.zeros_like(inverted, dtype=np.float32)
    for sigma in scales:
        response = np.maximum(
            response, _frangi_single_scale(inverted, sigma, beta, gamma)
        )

    min_value = float(response.min())
    max_value = float(response.max())
    if max_value > min_value:
        normalized = (response - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(response)

    return (normalized * 255.0).astype(np.uint8)


def get_image_paths(split):
    raw_split = RAW_SPLIT.get(split, split)
    image_dir = DATASET_DIR / raw_split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder citra tidak ditemukan: {image_dir}")

    paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di: {image_dir}")

    return paths


def preprocess_variants(image_bgr):
    """Kembalikan keempat varian preprocessing dari satu citra BGR."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    gray_clahe = clahe.apply(gray)

    gray_clahe_gamma = gamma_correction(gray_clahe, GAMMA)

    # Varian tambahan: CLAHE -> median filter -> Frangi vesselness.
    # Median filter mengikuti pipeline Wiharto & Palgunadi (2019).
    gray_clahe_median = cv2.medianBlur(gray_clahe, MEDIAN_KERNEL_SIZE)
    gray_clahe_frangi = frangi_vesselness(gray_clahe_median)

    return {
        "gray": gray,
        "gray_clahe": gray_clahe,
        "gray_clahe_gamma": gray_clahe_gamma,
        "gray_clahe_frangi": gray_clahe_frangi,
    }


def process_split(split):
    image_paths = get_image_paths(split)
    split_dir = OUTPUT_DIR / split
    if split_dir.exists():
        shutil.rmtree(split_dir)

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Gagal membaca citra: {image_path}")

        variants = preprocess_variants(image)
        for name, result in variants.items():
            _save_image(split_dir / name / f"{image_path.stem}.png", result)

    print(f"Split '{split}': {len(image_paths)} citra x {len(VARIANTS)} varian")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch preprocessing grayscale / CLAHE / CLAHE+gamma"
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
        help="split yang diproses (default: semua)",
    )
    args = parser.parse_args()

    splits = ("train", "test")
    if args.split != "all":
        splits = (args.split,)

    for split in splits:
        process_split(split)

    print(f"Hasil preprocessing tersimpan di: {OUTPUT_DIR}")
