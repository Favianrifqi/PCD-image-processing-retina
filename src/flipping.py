import cv2
import os


# ==========================================
# FOLDER INPUT DAN OUTPUT
# ==========================================

input_folder = "data/raw/DRIVE/training/images"
output_folder = "results/geometry"


# Membuat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)


# ==========================================
# DAFTAR CITRA
# ==========================================

image_names = [
    f"{number}_training.tif"
    for number in range(21, 41)
]


# ==========================================
# PROSES SETIAP CITRA
# ==========================================

for image_name in image_names:

    image_path = os.path.join(input_folder, image_name)

    # Membaca citra berwarna
    image = cv2.imread(image_path)

    # Memastikan citra berhasil dibaca
    if image is None:
        raise FileNotFoundError(
            f"Citra tidak ditemukan: {image_path}"
        )

    # Mendapatkan nomor citra
    image_id = image_name.replace("_training.tif", "")

    print(f"\nMemproses: {image_name}")
    print(f"Ukuran citra: {image.shape}")


    # ==========================================
    # HORIZONTAL FLIPPING
    # ==========================================

    horizontal_flip = cv2.flip(image, 1)

    horizontal_output_path = (
        f"{output_folder}/flipping_horizontal_{image_id}.png"
    )

    cv2.imwrite(horizontal_output_path, horizontal_flip)

    print(
        f"Horizontal flipping berhasil: "
        f"{horizontal_output_path}"
    )


    # ==========================================
    # VERTICAL FLIPPING
    # ==========================================

    vertical_flip = cv2.flip(image, 0)

    vertical_output_path = (
        f"{output_folder}/flipping_vertical_{image_id}.png"
    )

    cv2.imwrite(vertical_output_path, vertical_flip)

    print(
        f"Vertical flipping berhasil: "
        f"{vertical_output_path}"
    )


# ==========================================
# SELESAI
# ==========================================

print("\nSemua citra berhasil diproses.")
print("Total citra input : 20")
print("Total hasil       : 40")