import cv2
import os
import argparse


def negative_image(input_path, output_path):
    # Membaca citra
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(
            f"Citra tidak ditemukan atau gagal dibaca: {input_path}"
        )

    # Membuat citra negatif
    negative = 255 - image

    # Membuat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Menyimpan hasil citra negatif
    cv2.imwrite(output_path, negative)

    print("Negative image berhasil.")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Membuat citra negatif."
    )

    parser.add_argument(
        "input",
        help="Path citra input"
    )

    parser.add_argument(
        "output",
        help="Path untuk menyimpan citra negatif"
    )

    args = parser.parse_args()

    negative_image(args.input, args.output)