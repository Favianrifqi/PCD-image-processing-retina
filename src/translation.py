from pathlib import Path
import cv2
import numpy as np

DATA_RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("results/geometry/translation")
TX = 50  #vertikal
TY = 30  #horizontal

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_images(folder: Path):
    """Ambil semua file citra di dalam folder (termasuk subfolder)."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")

    images = [
        path for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di: {folder}")

    return images


def save_image(image, output_path: Path):
    """Simpan citra ke output_path, otomatis membuat folder tujuan jika belum ada."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(f"Citra gagal disimpan: {output_path}")


def translate_image(image, m: int = TX, n: int = TY):
    """
    Translasi citra dengan pseudocode:
        B[i][j] = A[i+m][j+n]
    Piksel di luar batas citra A diisi hitam (0).
    """
    M, N = image.shape[:2]  

    if image.ndim == 3:
        B = np.zeros((M, N, image.shape[2]), dtype=image.dtype)
    else:
        B = np.zeros((M, N), dtype=image.dtype)

    for i in range(M):        
        for j in range(N):   
            src_i = i + m
            src_j = j + n
            if 0 <= src_i < M and 0 <= src_j < N:
                B[i][j] = image[src_i][src_j]

    return B


def main():
    image_paths = get_images(DATA_RAW_DIR)
    print(f"Ditemukan {len(image_paths)} citra di {DATA_RAW_DIR}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"  [SKIP] Gagal membaca: {image_path}")
            continue

        result = translate_image(image)

        relative_path = image_path.relative_to(DATA_RAW_DIR)
        output_path = OUTPUT_DIR / relative_path

        save_image(result, output_path)
        print(f"  [OK] {relative_path} -> {output_path}")

    print("Selesai! Semua hasil translation tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()