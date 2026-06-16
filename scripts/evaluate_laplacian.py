import cv2
import numpy as np
import os

def laplacian_variance(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()


blur_folder = "./Motion_Deblurring/Datasets/GOPRO_Large_flat/test/blur"
output_folder = "./results/Test_Restormer_GoPro/visualization/GoProTestSet"

blur_scores = []
output_scores = []

# === HITUNG ===
for filename in os.listdir(output_folder):
    if "gt" not in filename:
        continue

    output_path = os.path.join(output_folder, filename)

    # ambil nama asli
    original_name = filename.replace("_gt", "")
    blur_path = os.path.join(blur_folder, original_name)

    if os.path.exists(blur_path):
        blur_score = laplacian_variance(blur_path)
        output_score = laplacian_variance(output_path)

        blur_scores.append(blur_score)
        output_scores.append(output_score)

# === HASIL ===
print("===== HASIL EVALUASI =====")
print(f"Jumlah gambar: {len(output_scores)}")
print(f"Rata-rata Laplacian BLUR   : {np.mean(blur_scores):.2f}")
print(f"Rata-rata Laplacian OUTPUT : {np.mean(output_scores):.2f}")
print(f"Improvement                : {np.mean(output_scores) - np.mean(blur_scores):.2f}")