import numpy as np

from preprocessing import (
    load_image,
    save_image,
    get_images,
    RESULTS_DIR
)


def zoom_out(image):

    M, N = image.shape[:2]

    if image.ndim == 3:
        result = np.zeros(
            (M * 2, N * 2, image.shape[2]),
            dtype=image.dtype
        )
    else:
        result = np.zeros(
            (M * 2, N * 2),
            dtype=image.dtype
        )

    m = 0
    n = 0

    for i in range(M):
        for j in range(N):

            result[m, n] = image[i, j]
            result[m, n + 1] = image[i, j]
            result[m + 1, n] = image[i, j]
            result[m + 1, n + 1] = image[i, j]

            n += 2

        m += 2
        n = 0

    return result


def zoom_in(image):

    M, N = image.shape[:2]

    new_M = M // 2
    new_N = N // 2

    if image.ndim == 3:
        result = np.zeros(
            (new_M, new_N, image.shape[2]),
            dtype=np.uint8
        )
    else:
        result = np.zeros(
            (new_M, new_N),
            dtype=np.uint8
        )

    for i in range(new_M):
        for j in range(new_N):

            pixel1 = image[i * 2, j * 2]
            pixel2 = image[i * 2, j * 2 + 1]
            pixel3 = image[i * 2 + 1, j * 2]
            pixel4 = image[i * 2 + 1, j * 2 + 1]

            average = (
                pixel1.astype(np.float32)
                + pixel2.astype(np.float32)
                + pixel3.astype(np.float32)
                + pixel4.astype(np.float32)
            ) / 4

            result[i, j] = average.astype(np.uint8)

    return result


def process_images(total_data):

    if total_data <= 0:
        raise ValueError(
            "Jumlah data harus lebih besar dari 0."
        )

    images = get_images("training")

    if total_data > len(images):
        raise ValueError(
            f"Jumlah data tidak mencukupi. "
            f"Diminta: {total_data}, "
            f"tersedia: {len(images)}."
        )

    selected_images = images[:total_data]

    output_dir = (
        RESULTS_DIR
        / "geometry"
        / "zooming"
    )

    for image_path in selected_images:

        print(
            f"Memproses: {image_path.name}"
        )

        image = load_image(image_path)

        zoomed_out = zoom_out(image)

        zoomed_in = zoom_in(image)

        filename = image_path.stem

        save_image(
            zoomed_out,
            output_dir
            / f"{filename}_zoom2x.png"
        )

        save_image(
            zoomed_in,
            output_dir
            / f"{filename}_zoom05x.png"
        )

        # Informasi ukuran
        original_height, original_width = image.shape[:2]

        zoom2_height, zoom2_width = zoomed_out.shape[:2]

        zoom05_height, zoom05_width = zoomed_in.shape[:2]

        print(
            f"  Original : "
            f"{original_width} x {original_height}"
        )

        print(
            f"  Zoom 2x  : "
            f"{zoom2_width} x {zoom2_height}"
        )

        print(
            f"  Zoom 1/2 : "
            f"{zoom05_width} x {zoom05_height}"
        )

    print()
    print("Proses zooming selesai.")
    print(f"Jumlah data : {total_data}")

if __name__ == "__main__":

    jumlah_data = int(
        input(
            "Masukkan jumlah data yang ingin di-zoom: "
        )
    )

    process_images(jumlah_data)