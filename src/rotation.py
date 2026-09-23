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
    """
    Rotasi 180° dengan melakukan
    rotasi 90° dua kali.
    """

    result = rotation90_ccw(image)
    result = rotation90_ccw(result)

    return result


def process_images(total_data):

    if total_data <= 0:
        raise ValueError(
            "Jumlah data harus lebih besar dari 0."
        )

    images = get_images("training")

    if total_data > len(images):
        raise ValueError(
            f"Data tidak mencukupi. "
            f"Diminta {total_data}, "
            f"tersedia {len(images)}."
        )

    selected_images = images[:total_data]

    output_dir = (
        RESULTS_DIR
        / "geometry"
        / "rotation"
    )

    for image_path in selected_images:

        print(
            f"Memproses: {image_path.name}"
        )

        image = load_image(image_path)

        rotated_ccw = rotation90_ccw(image)
        rotated_cw = rotation90_cw(image)
        rotated_180 = rotation180(image)

        filename = image_path.stem

        save_image(
            rotated_ccw,
            output_dir / f"{filename}_90ccw.png"
        )

        save_image(
            rotated_cw,
            output_dir / f"{filename}_90cw.png"
        )

        save_image(
            rotated_180,
            output_dir / f"{filename}_180.png"
        )

    print()
    print("Proses rotasi selesai.")
    print(f"Jumlah data : {total_data}")


if __name__ == "__main__":

    jumlah_data = int(
        input(
            "Masukkan jumlah data yang ingin di-rotate: "
        )
    )

    process_images(jumlah_data)