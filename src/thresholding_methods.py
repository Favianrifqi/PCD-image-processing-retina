import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def process_thresholding_techniques(input_path, output_dir):
    # 1. Baca Citra Grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan di: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. GLOBAL THRESHOLDING
    # ==========================================
    # a. Manual / Fixed Binary (T = 127)
    _, thresh_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # b. Otsu's Thresholding (Otomatis)
    t_otsu, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ==========================================
    # 2. LOCAL / ADAPTIVE THRESHOLDING
    # ==========================================
    # a. Adaptive Mean Thresholding
    thresh_adapt_mean = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # b. Adaptive Gaussian Thresholding
    thresh_adapt_gaussian = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # ==========================================
    # 3. MULTILEVEL THRESHOLDING (3 Kelas)
    # ==========================================
    # Membagi citra menjadi 3 tingkat intensitas dengan T1=80 dan T2=160
    t1, t2 = 80, 160
    thresh_multi = np.zeros_like(img)
    thresh_multi[img < t1] = 0        # Kelompok 1: Hitam
    thresh_multi[(img >= t1) & (img < t2)] = 127  # Kelompok 2: Abu-abu
    thresh_multi[img >= t2] = 255     # Kelompok 3: Putih

    # ==========================================
    # 4. VISUALISASI PERBANDINGAN
    # ==========================================
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Original Grayscale")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(thresh_manual, cmap='gray')
    plt.title("2. Global Manual (T=127)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f"3. Global Otsu (T={int(t_otsu)})")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(thresh_adapt_mean, cmap='gray')
    plt.title("4. Adaptive Mean")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(thresh_adapt_gaussian, cmap='gray')
    plt.title("5. Adaptive Gaussian")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(thresh_multi, cmap='gray')
    plt.title(f"6. Multilevel (T1={t1}, T2={t2})")
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "thresholding_all_methods_comparison.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Hasil perbandingan thresholding berhasil disimpan di: {output_path}")
    print(f"Nilai ambang (T) Otsu otomatis: {t_otsu}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P5 - All Thresholding Methods")
    parser.add_argument("input", type=str, help="Path ke file gambar input")
    parser.add_argument("output_dir", type=str, help="Folder untuk menyimpan hasil")

    args = parser.parse_args()
    process_thresholding_techniques(args.input, args.output_dir)