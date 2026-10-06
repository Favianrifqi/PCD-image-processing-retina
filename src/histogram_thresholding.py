import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def analyze_and_threshold(input_path, output_dir):
    # 1. Baca Citra Grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)

    # 2. Histogram Perataan / Pemerataan (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # 3. Penerapan Berbagai Teknik Thresholding
    # a. Global Thresholding Biasa (Nilai t=127)
    _, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # b. Otsu's Thresholding (Original)
    _, thresh_otsu_orig = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # c. Otsu's Thresholding pada Histogram Setelah CLAHE
    _, thresh_otsu_clahe = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # d. Adaptive Thresholding (Gaussian)
    thresh_adaptive = cv2.adaptiveThreshold(
        img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. Plot dan Perbandingan Histogram & Hasil Thresholding
    plt.figure(figsize=(15, 10))

    # Baris 1: Citra Asli, CLAHE, dan Histogram Masing-Masing
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Original Grayscale")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.title("2. After CLAHE (Equalized)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.hist(img.ravel(), 256, [0, 256], alpha=0.5, label='Original', color='red')
    plt.hist(img_clahe.ravel(), 256, [0, 256], alpha=0.5, label='CLAHE', color='blue')
    plt.title("Histogram Comparison")
    plt.legend()

    # Baris 2: Perbandingan Metode Thresholding
    plt.subplot(2, 3, 4)
    plt.imshow(thresh_otsu_orig, cmap='gray')
    plt.title("3. Otsu (Original)")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(thresh_otsu_clahe, cmap='gray')
    plt.title("4. Otsu (After CLAHE)")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(thresh_adaptive, cmap='gray')
    plt.title("5. Adaptive Gaussian (CLAHE)")
    plt.axis('off')

    plt.tight_layout()
    
    output_plot_path = os.path.join(output_dir, "thresholding_comparison.png")
    plt.savefig(output_plot_path)
    plt.close()
    
    print(f"Hasil analisis dan perbandingan berhasil disimpan di: {output_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Analysis and Thresholding Comparison")
    parser.add_argument("input", type=str, help="Path ke file citra input")
    parser.add_argument("output_dir", type=str, help="Folder untuk menyimpan hasil")
    
    args = parser.parse_args()
    analyze_and_threshold(args.input, args.output_dir)