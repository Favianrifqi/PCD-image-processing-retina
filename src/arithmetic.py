import cv2
import numpy as np


# ==========================================
# DAFTAR PASANGAN CITRA
# ==========================================

image_pairs = [
    ("21_training.tif", "22_training.tif"),
    ("23_training.tif", "24_training.tif"),
    ("25_training.tif", "26_training.tif"),
    ("27_training.tif", "28_training.tif"),
    ("29_training.tif", "30_training.tif"),
    ("31_training.tif", "32_training.tif"),
    ("33_training.tif", "34_training.tif"),
    ("35_training.tif", "36_training.tif"),
    ("37_training.tif", "38_training.tif"),
    ("39_training.tif", "40_training.tif"),
]


# ==========================================
# PROSES SETIAP PASANGAN
# ==========================================

for image_a_name, image_b_name in image_pairs:

    image_a_path = f"data/raw/DRIVE/training/images/{image_a_name}"
    image_b_path = f"data/raw/DRIVE/training/images/{image_b_name}"

    # Membaca dua citra berwarna
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)

    # Memastikan citra berhasil dibaca
    if image_a is None:
        raise FileNotFoundError(
            f"Citra A tidak ditemukan: {image_a_path}"
        )

    if image_b is None:
        raise FileNotFoundError(
            f"Citra B tidak ditemukan: {image_b_path}"
        )

    # ==========================================
    # CEK KOMPATIBILITAS CITRA
    # ==========================================

    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Ukuran/channel citra tidak sama: "
            f"A={image_a.shape}, B={image_b.shape}"
        )

    print(f"\nMemproses: {image_a_name} + {image_b_name}")
    print(f"Ukuran citra: {image_a.shape}")

    # Mengubah ke integer sementara
    image_a_int = image_a.astype(np.int16)
    image_b_int = image_b.astype(np.int16)

    # ==========================================
    # ADDITION
    # ==========================================

    addition = image_a_int + image_b_int

    # Membatasi nilai pixel ke 0-255
    addition = np.clip(addition, 0, 255)

    # Mengembalikan ke uint8
    addition = addition.astype(np.uint8)

    # Nama output
    image_a_id = image_a_name.replace("_training.tif", "")
    image_b_id = image_b_name.replace("_training.tif", "")

    addition_output_path = (
        f"results/arithmetic/addition_{image_a_id}_{image_b_id}.png"
    )

    cv2.imwrite(addition_output_path, addition)

    print(f"Addition berhasil: {addition_output_path}")

    # ==========================================
    # SUBTRACTION
    # ==========================================

    subtraction = image_a_int - image_b_int

    # Nilai negatif menjadi 0
    subtraction = np.clip(subtraction, 0, 255)

    # Mengembalikan ke uint8
    subtraction = subtraction.astype(np.uint8)

    subtraction_output_path = (
        f"results/arithmetic/subtraction_{image_a_id}_{image_b_id}.png"
    )

    cv2.imwrite(subtraction_output_path, subtraction)

    print(f"Subtraction berhasil: {subtraction_output_path}")


# ==========================================
# SELESAI
# ==========================================

print("\nSemua pasangan citra berhasil diproses.")