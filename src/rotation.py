import numpy as np

from preprocessing import (
    load_image,
    save_image,
    get_images,
    RESULTS_DIR
)

def rotation90_ccw(image):
    """
    Rotasi citra 90 derajat berlawanan arah jarum jam (CCW).

    A berukuran M x N
    B berukuran N x M

    B[N - 1 - j][i] = A[i][j]
    """

    M, N = image.shape[:2]

    if image.ndim == 3:
        result = np.zeros(
            (N, M, image.shape[2]),
            dtype=image.dtype
        )
    else:
        result = np.zeros(
            (N, M),
            dtype=image.dtype
        )

    for i in range(M):
        for j in range(N):
            result[N - 1 - j, i] = image[i, j]

    return result


def rotation90_cw(image):
    """
    Rotasi citra 90 derajat searah jarum jam (CW).

    A berukuran M x N
    B berukuran N x M

    B[j][M - 1 - i] = A[i][j]
    """

    M, N = image.shape[:2]

    if image.ndim == 3:
        result = np.zeros(
            (N, M, image.shape[2]),
            dtype=image.dtype
        )
    else:
        result = np.zeros(
            (N, M),
            dtype=image.dtype
        )

    for i in range(M):
        for j in range(N):
            result[j, M - 1 - i] = image[i, j]

    return result

def rotation180(image):

    result = rotation90_ccw(image)
    result = rotation90_ccw(result)

    return result


def process_images(total_data):

    if total_data <= 0:
        raise ValueError(
            "Jumlah data harus lebih besar dari 0."
        )

    # Menentukan jumlah masing-masing kelas
    normal_count = total_data // 2
    pneumonia_count = total_data - normal_count

    # Mengambil daftar citra
    normal_images = get_images(
        "train",
        "NORMAL"
    )

    pneumonia_images = get_images(
        "train",
        "PNEUMONIA"
    )

    if normal_count > len(normal_images):
        raise ValueError(
            f"Data NORMAL tidak mencukupi. "
            f"Dibutuhkan {normal_count}, "
            f"tersedia {len(normal_images)}."
        )

    if pneumonia_count > len(pneumonia_images):
        raise ValueError(
            f"Data PNEUMONIA tidak mencukupi. "
            f"Dibutuhkan {pneumonia_count}, "
            f"tersedia {len(pneumonia_images)}."
        )

    selected_normal = normal_images[:normal_count]
    selected_pneumonia = pneumonia_images[:pneumonia_count]

    selected_images = [
        ("NORMAL", image)
        for image in selected_normal
    ]

    selected_images += [
        ("PNEUMONIA", image)
        for image in selected_pneumonia
    ]

    # Memproses setiap citra
    for category, image_path in selected_images:

        print(f"Memproses: {category} / {image_path.name}")

        image = load_image(image_path)

        # Rotasi
        image_ccw = rotation90_ccw(image)
        image_cw = rotation90_cw(image)
        image_180 = rotation180(image)

        # Folder output
        output_dir = (
            RESULTS_DIR
            / "geometry"
            / "rotation"
            / category
        )

        filename = image_path.stem

        # Simpan hasil
        save_image(
            image_ccw,
            output_dir / f"{filename}_90ccw.png"
        )

        save_image(
            image_cw,
            output_dir / f"{filename}_90cw.png"
        )

        save_image(
            image_180,
            output_dir / f"{filename}_180.png"
        )

    print()
    print("Proses rotasi selesai.")
    print(f"Total citra     : {total_data}")
    print(f"NORMAL          : {normal_count}")
    print(f"PNEUMONIA       : {pneumonia_count}")

if __name__ == "__main__":

    jumlah_data = int(
        input("Masukkan jumlah data yang ingin di-rotate: ")
    )

    process_images(jumlah_data)