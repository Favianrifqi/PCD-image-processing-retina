import numpy as np

from preprocessing import (
    load_image,
    save_image,
    get_images,
    RESULTS_DIR
)


def zoom_out(image):
    """
    Perbesaran citra dengan faktor skala 2.

    Setiap pixel pada citra asli disalin
    menjadi blok 2x2 pada citra hasil.

    Ukuran:
        M x N -> 2M x 2N
    """

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
    """
    Pengecilan citra dengan faktor skala 1/2.

    Setiap 4 pixel yang bertetangga
    dirata-ratakan menjadi 1 pixel.

    Ukuran:
        M x N -> M/2 x N/2
    """

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

if __name__ == "__main__":

    jumlah_data = int(
        input("Masukkan jumlah data yang ingin di-zoom: ")
    )

    if jumlah_data <= 0:
        raise ValueError(
            "Jumlah data harus lebih besar dari 0."
        )

    normal_count = jumlah_data // 2
    pneumonia_count = jumlah_data - normal_count

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
            "Jumlah data NORMAL tidak mencukupi."
        )

    if pneumonia_count > len(pneumonia_images):
        raise ValueError(
            "Jumlah data PNEUMONIA tidak mencukupi."
        )

    selected_images = [
        ("NORMAL", image)
        for image in normal_images[:normal_count]
    ]

    selected_images += [
        ("PNEUMONIA", image)
        for image in pneumonia_images[:pneumonia_count]
    ]

    for category, image_path in selected_images:

        print(
            f"Memproses: "
            f"{category} / {image_path.name}"
        )

        image = load_image(image_path)

        zoomed_out = zoom_out(image)

        zoomed_in = zoom_in(image)

        filename = image_path.stem

        output_dir = (
            RESULTS_DIR
            / "geometry"
            / "zooming"
            / category
        )

        save_image(
            zoomed_out,
            output_dir / f"{filename}_zoom2x.png"
        )

        save_image(
            zoomed_in,
            output_dir / f"{filename}_zoom05x.png"
        )

    print()
    print("Proses zooming selesai.")
    print(f"Total citra     : {jumlah_data}")
    print(f"NORMAL          : {normal_count}")
    print(f"PNEUMONIA       : {pneumonia_count}")
