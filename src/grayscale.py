import cv2
import os
import argparse


def grayscale_image(input_path, output_path):
    # Membaca citra
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(
            f"Citra tidak ditemukan atau gagal dibaca: {input_path}"
        )

    # Mengubah citra BGR menjadi grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Membuat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Menyimpan hasil grayscale
    cv2.imwrite(output_path, gray)

    print("Grayscale berhasil.")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mengubah citra berwarna menjadi grayscale."
    )

    parser.add_argument(
        "input",
        help="Path citra input"
    )

    parser.add_argument(
        "output",
        help="Path untuk menyimpan citra grayscale"
    )

    args = parser.parse_args()

    grayscale_image(args.input, args.output)