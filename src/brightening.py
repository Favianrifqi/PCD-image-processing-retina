from pathlib import Path
import cv2

DATA_RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("results/brightening")
BRIGHTNESS_VALUE = 100  

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


def brighten_image(image, b: int = BRIGHTNESS_VALUE):
    """
    Mencerahkan citra dengan pseudocode:
        temp = A[i][j] + b
        if temp < 0   -> B[i][j] = 0
        if temp > 255 -> B[i][j] = 255
        else          -> B[i][j] = temp
    """
    M, N = image.shape[:2]  
    B = image.copy()

    for i in range(M):        
        for j in range(N):    
            if image.ndim == 3:
                for c in range(image.shape[2]):  
                    temp = int(image[i][j][c]) + b
                    if temp < 0:
                        B[i][j][c] = 0
                    elif temp > 255:
                        B[i][j][c] = 255
                    else:
                        B[i][j][c] = temp
            else:
                temp = int(image[i][j]) + b
                if temp < 0:
                    B[i][j] = 0
                elif temp > 255:
                    B[i][j] = 255
                else:
                    B[i][j] = temp

    return B


def main():
    image_paths = get_images(DATA_RAW_DIR)
    print(f"Ditemukan {len(image_paths)} citra di {DATA_RAW_DIR}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"  [SKIP] Gagal membaca: {image_path}")
            continue

        result = brighten_image(image)

        relative_path = image_path.relative_to(DATA_RAW_DIR)
        output_path = OUTPUT_DIR / relative_path

        save_image(result, output_path)
        print(f"  [OK] {relative_path} -> {output_path}")

    print("Selesai! Semua hasil brightening tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()