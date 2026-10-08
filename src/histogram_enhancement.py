feature/negative-grayscale-thresholding
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def add_grayscale_colorbar(ax, max_val):
    """Menambahkan pita gradasi 0-255 dinamis di bawah grafik histogram"""
    gradient = np.linspace(0, 255, 256).reshape(1, 256)
    y_min = -max_val * 0.08
    y_max = -max_val * 0.01
    ax.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 255, y_min, y_max])
    ax.set_ylim([y_min, max_val * 1.05])

def process_histogram_techniques(input_path, output_dir):
    # 1. Baca Citra Grayscale & Warna
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(input_path)
    
    if img_gray is None or img_color is None:
        raise FileNotFoundError(f"Citra tidak ditemukan di: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # A. HISTOGRAM STRETCHING (Perenggangan)
    r_min = np.min(img_gray)
    r_max = np.max(img_gray)
    if r_max > r_min:
        img_stretched = ((img_gray - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    else:
        img_stretched = img_gray.copy()

    # B. HISTOGRAM EQUALIZATION (Perataan Global)
    img_equalized = cv2.equalizeHist(img_gray)

    # C. CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # D. VISUALISASI PERBANDINGAN ENHANCEMENT
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    # 1. Citra Asli vs Histogram Normalisasi
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("1. Citra Grayscale Asli")
    axes[0, 0].axis('off')
    
    hist_orig, _ = np.histogram(img_gray.ravel(), 256, [0, 256])
    hist_norm = hist_orig / hist_orig.sum()  # Rumus h_i = n_i / n
    axes[0, 1].plot(hist_norm, color='#3388ff')
    axes[0, 1].set_title("Histogram Normalisasi (Asli)")
    axes[0, 1].set_xlim([0, 255])
    add_grayscale_colorbar(axes[0, 1], max(hist_norm))

    # 2. Histogram Stretching
    axes[1, 0].imshow(img_stretched, cmap='gray')
    axes[1, 0].set_title(f"2. Contrast Stretching (min={r_min}, max={r_max})")
    axes[1, 0].axis('off')
    
    hist_str, _ = np.histogram(img_stretched.ravel(), 256, [0, 256])
    axes[1, 1].hist(img_stretched.ravel(), 256, [0, 256], color='#3388ff')
    axes[1, 1].set_title("Histogram After Stretching")
    axes[1, 1].set_xlim([0, 255])
    add_grayscale_colorbar(axes[1, 1], max(hist_str))

    # 3. Histogram Equalization
    axes[2, 0].imshow(img_equalized, cmap='gray')
    axes[2, 0].set_title("3. Global Histogram Equalization")
    axes[2, 0].axis('off')
    
    hist_eq, _ = np.histogram(img_equalized.ravel(), 256, [0, 256])
    axes[2, 1].hist(img_equalized.ravel(), 256, [0, 256], color='#3388ff')
    axes[2, 1].set_title("Histogram After Equalization")
    axes[2, 1].set_xlim([0, 255])
    add_grayscale_colorbar(axes[2, 1], max(hist_eq))

    # 4. CLAHE
    axes[3, 0].imshow(img_clahe, cmap='gray')
    axes[3, 0].set_title("4. CLAHE (Adaptive Equalization)")
    axes[3, 0].axis('off')
    
    hist_clahe, _ = np.histogram(img_clahe.ravel(), 256, [0, 256])
    axes[3, 1].hist(img_clahe.ravel(), 256, [0, 256], color='#3388ff')
    axes[3, 1].set_title("Histogram After CLAHE")
    axes[3, 1].set_xlim([0, 255])
    add_grayscale_colorbar(axes[3, 1], max(hist_clahe))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "histogram_enhancement_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    # E. VISUALISASI HISTOGRAM WARNA (RGB)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_color_rgb)
    plt.title("Citra Warna (RGB)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_color_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'Kanal {col.upper()}')
    plt.title("Histogram Warna (RGB)")
    plt.xlim([0, 255])
    plt.legend()

    plt.tight_layout()
    color_plot_path = os.path.join(output_dir, "histogram_rgb_analysis.png")
    plt.savefig(color_plot_path)
    plt.close()

    print(f"Hasil visualisasi berhasil disimpan di folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4 - Histogram Processing & Enhancement")
    parser.add_argument("input", type=str, help="Path ke file gambar input")
    parser.add_argument("output_dir", type=str, help="Folder untuk menyimpan hasil")

    args = parser.parse_args()
    process_histogram_techniques(args.input, args.output_dir)