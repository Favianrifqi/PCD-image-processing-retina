import os

import cv2
import numpy as np


# Lokasi citra input
IMAGE_A_PATH = (
    "data/raw/chest_xray/train/NORMAL/IM-0115-0001.jpeg"
)
IMAGE_B_PATH = (
    "data/raw/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg"
)

# Ukuran canvas bersama (height, width)
TARGET_SIZE = (1152, 1152)

# Lokasi penyimpanan hasil
OUTPUT_DIR = "results/arithmetic"
ADDITION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "addition.png")
SUBTRACTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "subtraction.png")


def load_grayscale(image_path):
    """Membaca citra dalam mode grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Citra tidak ditemukan atau gagal dibaca: {image_path}"
        )

    return image


def resize_and_pad(image, target_size):
    """
    Resize citra dengan mempertahankan aspect ratio,
    kemudian memberikan padding agar sesuai dengan target_size.
    """
    target_height, target_width = target_size

    image_height, image_width = image.shape

    scale = min(
        target_width / image_width,
        target_height / image_height,
    )

    new_width = round(image_width * scale)
    new_height = round(image_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded


def add_images(image_a, image_b):
    """
    Operasi penjumlahan dua citra.

    C(x,y) = A(x,y) + B(x,y)

    Nilai pixel dibatasi pada rentang 0-255.
    """
    image_a_int = image_a.astype(np.int16)
    image_b_int = image_b.astype(np.int16)

    addition = image_a_int + image_b_int
    addition = np.clip(addition, 0, 255)

    return addition.astype(np.uint8)


def subtract_images(image_a, image_b):
    """
    Operasi pengurangan dua citra sesuai implementasi
    pada materi P3.

    C(x,y) = A(x,y) - B(x,y)

    Nilai 0 tetap 0, sedangkan nilai selain 0
    menjadi 255.
    """
    image_a_int = image_a.astype(np.int16)
    image_b_int = image_b.astype(np.int16)

    subtraction = image_a_int - image_b_int

    subtraction = np.where(
        subtraction != 0,
        255,
        0,
    )

    return subtraction.astype(np.uint8)


def main():
    # Membaca kedua citra sebagai grayscale
    image_a = load_grayscale(IMAGE_A_PATH)
    image_b = load_grayscale(IMAGE_B_PATH)

    # Menyesuaikan ukuran kedua citra tanpa mengubah aspect ratio
    image_a_processed = resize_and_pad(image_a, TARGET_SIZE)
    image_b_processed = resize_and_pad(image_b, TARGET_SIZE)

    # Memastikan ukuran kedua citra sama
    if image_a_processed.shape != image_b_processed.shape:
        raise ValueError(
            "Ukuran citra setelah preprocessing tidak sama: "
            f"A={image_a_processed.shape}, "
            f"B={image_b_processed.shape}"
        )

    # Operasi arithmetic
    addition = add_images(
        image_a_processed,
        image_b_processed,
    )

    subtraction = subtract_images(
        image_a_processed,
        image_b_processed,
    )

    # Memastikan folder output tersedia
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Menyimpan hasil
    if not cv2.imwrite(ADDITION_OUTPUT_PATH, addition):
        raise IOError(
            f"Gagal menyimpan hasil addition: "
            f"{ADDITION_OUTPUT_PATH}"
        )

    if not cv2.imwrite(SUBTRACTION_OUTPUT_PATH, subtraction):
        raise IOError(
            f"Gagal menyimpan hasil subtraction: "
            f"{SUBTRACTION_OUTPUT_PATH}"
        )

    # Informasi proses
    print("Operasi arithmetic berhasil.")
    print(f"Citra A asli : {image_a.shape}")
    print(f"Citra B asli : {image_b.shape}")
    print(f"Citra A akhir: {image_a_processed.shape}")
    print(f"Citra B akhir: {image_b_processed.shape}")
    print(f"Addition     : {ADDITION_OUTPUT_PATH}")
    print(f"Subtraction  : {SUBTRACTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
