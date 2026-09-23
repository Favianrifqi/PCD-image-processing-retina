from pathlib import Path

import cv2
import numpy as np


# ==========================================
# FOLDER INPUT DAN OUTPUT
# ==========================================

INPUT_FOLDER = Path("data/raw/DRIVE/training/images")
OUTPUT_FOLDER = Path("results/histogram")

# Citra training yang digunakan: 21 sampai 40
IMAGE_NAMES = [
    f"{number}_training.tif"
    for number in range(21, 41)
]

# Parameter enhancement
GAMMA = 0.7
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


# ==========================================
# FUNGSI PENYIMPANAN
# ==========================================

def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(
            f"Citra gagal disimpan: {output_path}"
        )


# ==========================================
# HISTOGRAM EQUALIZATION
# ==========================================

def histogram_equalization(image):
    """
    Histogram Equalization diterapkan pada
    kanal luminance (L) dalam ruang warna LAB.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l_equalized = cv2.equalizeHist(l)

    result = cv2.merge(
        [l_equalized, a, b]
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    return result


# ==========================================
# CLAHE
# ==========================================

def clahe_enhancement(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram
    Equalization) diterapkan pada kanal luminance.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )

    l_clahe = clahe.apply(l)

    result = cv2.merge(
        [l_clahe, a, b]
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    return result


# ==========================================
# GAMMA CORRECTION
# ==========================================

def gamma_correction(image, gamma=GAMMA):
    """
    Gamma Correction diterapkan pada kanal
    luminance.

    gamma < 1 menghasilkan citra yang lebih terang.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    normalized = l.astype(np.float32) / 255.0

    corrected = np.power(
        normalized,
        gamma
    ) * 255.0

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    result = cv2.merge(
        [corrected, a, b]
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    return result


# ==========================================
# CONTRAST STRETCHING
# ==========================================

def contrast_stretching(image):
    """
    Contrast Stretching menggunakan percentile
    2% dan 98% pada kanal luminance.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    low = np.percentile(l, 2)
    high = np.percentile(l, 98)

    if high <= low:
        stretched = l.copy()

    else:
        stretched = (
            (l.astype(np.float32) - low)
            * 255.0
            / (high - low)
        )

        stretched = np.clip(
            stretched,
            0,
            255
        ).astype(np.uint8)

    result = cv2.merge(
        [stretched, a, b]
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    return result


# ==========================================
# PROSES SEMUA CITRA
# ==========================================

def main():

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: "
            f"{INPUT_FOLDER}"
        )

    print(
        f"Folder input : {INPUT_FOLDER}"
    )

    print(
        f"Folder output: {OUTPUT_FOLDER}"
    )

    print(
        f"Jumlah citra : {len(IMAGE_NAMES)}"
    )

    print()

    processed = 0

    for image_name in IMAGE_NAMES:

        image_path = INPUT_FOLDER / image_name

        if not image_path.exists():
            print(
                f"[SKIP] Citra tidak ditemukan: "
                f"{image_path}"
            )
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[SKIP] Gagal membaca: "
                f"{image_path}"
            )
            continue

        print(
            f"Memproses: {image_name}"
        )

        # HE
        he = histogram_equalization(image)

        save_image(
            he,
            OUTPUT_FOLDER
            / "he"
            / f"he_{image_name.replace('.tif', '.png')}"
        )

        # CLAHE
        clahe = clahe_enhancement(image)

        save_image(
            clahe,
            OUTPUT_FOLDER
            / "clahe"
            / f"clahe_{image_name.replace('.tif', '.png')}"
        )

        # Gamma Correction
        gamma = gamma_correction(image)

        save_image(
            gamma,
            OUTPUT_FOLDER
            / "gamma"
            / f"gamma_{image_name.replace('.tif', '.png')}"
        )

        # Contrast Stretching
        stretching = contrast_stretching(image)

        save_image(
            stretching,
            OUTPUT_FOLDER
            / "contrast_stretching"
            / f"contrast_stretching_{image_name.replace('.tif', '.png')}"
        )

        processed += 1

    print()
    print("==========================================")
    print("Histogram enhancement selesai.")
    print(f"Citra berhasil diproses: {processed}")
    print("Metode: HE, CLAHE, Gamma, Contrast Stretching")
    print("==========================================")


if __name__ == "__main__":
    main()