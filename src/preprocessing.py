from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "raw" / "chest_xray"

RESULTS_DIR = BASE_DIR / "results"


def load_image(image_path):
  
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Citra tidak ditemukan: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Citra gagal dibaca: {image_path}"
        )

    return image


def save_image(image, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        image
    )

    if not success:
        raise IOError(
            f"Citra gagal disimpan: {output_path}"
        )


def get_images(split, category):

    folder = DATASET_DIR / split / category

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {folder}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    images = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    return sorted(images)