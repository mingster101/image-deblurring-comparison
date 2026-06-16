"""
Unsharp Mask Hyperparameter Search + Evaluation
================================================
Step 1: Grid search optimal parameters on GoPro train subset (50 images)
Step 2: Apply best parameters to full GoPro test set (1111 images)
Step 3: Compute PSNR, SSIM, LPIPS
"""

import os
import cv2
import glob
import numpy as np
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from itertools import product
import random

# ============================================================
# KONFIGURASI
# ============================================================
TRAIN_BLUR_DIR  = r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\GOPRO_Large_flat\train\blur'
TRAIN_SHARP_DIR = r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\GOPRO_Large_flat\train\sharp'
TEST_BLUR_DIR   = r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\GOPRO_Large_flat\test\blur'
TEST_SHARP_DIR  = r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\GOPRO_Large_flat\test\sharp'
OUTPUT_DIR      = r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Evaluation\results\unsharp_mask'

VAL_SAMPLE_SIZE = 50   # gambar dari train set untuk grid search
RANDOM_SEED     = 42
# ============================================================

# Parameter grid
RADIUS_VALUES    = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
AMOUNT_VALUES    = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
THRESHOLD_VALUES = [0, 2, 3, 5, 7, 10, 15, 20]
# Total: 10 x 10 x 8 = 800 kombinasi


def apply_unsharp_mask(img_bgr, radius, amount, threshold):
    """
    Apply Unsharp Mask to BGR image.
    radius    : Gaussian blur sigma
    amount    : strength of sharpening (0-3)
    threshold : min difference to apply sharpening (0-255)
    """
    # Blur the image
    ksize = int(6 * radius + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), radius)

    # Compute difference (high-frequency detail)
    diff = img_bgr.astype(np.float32) - blurred.astype(np.float32)

    # Apply threshold
    if threshold > 0:
        mask = np.abs(diff) >= threshold
        sharpened = img_bgr.astype(np.float32) + amount * diff * mask
    else:
        sharpened = img_bgr.astype(np.float32) + amount * diff

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_psnr_batch(blur_paths, sharp_dir, radius, amount, threshold):
    """Compute average PSNR for a set of images with given USM parameters."""
    psnr_scores = []
    for blur_path in blur_paths:
        fname = os.path.basename(blur_path)
        sharp_path = os.path.join(sharp_dir, fname)
        if not os.path.exists(sharp_path):
            continue

        blur_img  = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)
        output    = apply_unsharp_mask(blur_img, radius, amount, threshold)

        psnr_val = calc_psnr(sharp_img, output, data_range=255)
        psnr_scores.append(psnr_val)

    return np.mean(psnr_scores) if psnr_scores else 0.0


# ============================================================
# STEP 1: GRID SEARCH ON VALIDATION SET
# ============================================================
print('=' * 60)
print('  Unsharp Mask Hyperparameter Grid Search')
print('=' * 60)

# Sample validation images from train set
random.seed(RANDOM_SEED)
all_train = sorted(glob.glob(os.path.join(TRAIN_BLUR_DIR, '*.png')))
val_images = random.sample(all_train, min(VAL_SAMPLE_SIZE, len(all_train)))
print(f'Validation set: {len(val_images)} images from train split')

# Build all combinations
all_params = list(product(RADIUS_VALUES, AMOUNT_VALUES, THRESHOLD_VALUES))
print(f'Grid size: {len(all_params)} combinations')
print(f'Estimated time: ~{len(all_params) * len(val_images) / 10000:.1f} minutes\n')

best_psnr   = 0.0
best_params = None
results_log = []

for i, (radius, amount, threshold) in enumerate(tqdm(all_params, desc='Grid Search')):
    avg_psnr = compute_psnr_batch(val_images, TRAIN_SHARP_DIR, radius, amount, threshold)
    results_log.append((avg_psnr, radius, amount, threshold))

    if avg_psnr > best_psnr:
        best_psnr   = avg_psnr
        best_params = (radius, amount, threshold)

# Sort and show top 10
results_log.sort(reverse=True)
print(f'\nTop 10 parameter combinations (by PSNR on validation set):')
print(f'  {"Rank":<6} {"PSNR":>8} {"radius":>8} {"amount":>8} {"threshold":>10}')
print(f'  {"-"*6} {"-"*8} {"-"*8} {"-"*8} {"-"*10}')
for rank, (psnr_val, r, a, t) in enumerate(results_log[:10], 1):
    marker = " ← BEST" if rank == 1 else ""
    print(f'  {rank:<6} {psnr_val:>8.4f} {r:>8.1f} {a:>8.1f} {t:>10}{marker}')

best_radius, best_amount, best_threshold = best_params
print(f'\nBest parameters: radius={best_radius}, amount={best_amount}, threshold={best_threshold}')
print(f'Validation PSNR: {best_psnr:.4f} dB')

# ============================================================
# STEP 2: EVALUATE ON TEST SET WITH BEST PARAMETERS
# ============================================================
print(f'\n{"=" * 60}')
print(f'  Evaluating on GoPro Test Set (1111 images)')
print(f'  Parameters: radius={best_radius}, amount={best_amount}, threshold={best_threshold}')
print(f'{"=" * 60}')

os.makedirs(OUTPUT_DIR, exist_ok=True)

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = lpips.LPIPS(net='alex').to(device)
print(f'LPIPS device: {device}')

test_files = sorted(glob.glob(os.path.join(TEST_BLUR_DIR, '*.png')))
print(f'Test images: {len(test_files)}')

psnr_scores, ssim_scores, lpips_scores = [], [], []
missing = 0

for blur_path in tqdm(test_files, desc='Evaluating'):
    fname      = os.path.basename(blur_path)
    sharp_path = os.path.join(TEST_SHARP_DIR, fname)

    if not os.path.exists(sharp_path):
        missing += 1
        continue

    blur_img  = cv2.imread(blur_path)
    sharp_img = cv2.imread(sharp_path)
    output    = apply_unsharp_mask(blur_img, best_radius, best_amount, best_threshold)

    # Save output
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), output)

    # PSNR
    psnr_scores.append(calc_psnr(sharp_img, output, data_range=255))

    # SSIM
    sharp_rgb  = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    ssim_scores.append(calc_ssim(sharp_rgb, output_rgb, channel_axis=2, data_range=255))

    # LPIPS
    def to_tensor(img_rgb):
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0 * 2 - 1)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        lpips_scores.append(loss_fn(to_tensor(output_rgb), to_tensor(sharp_rgb)).item())

if missing:
    print(f'Warning: {missing} GT files not found')

# ============================================================
# FINAL RESULTS
# ============================================================
print(f'\n{"=" * 60}')
print(f'  FINAL RESULTS — Unsharp Mask (Optimized Parameters)')
print(f'{"=" * 60}')
print(f'  Best parameters:')
print(f'    radius    = {best_radius}')
print(f'    amount    = {best_amount}')
print(f'    threshold = {best_threshold}')
print(f'  Images evaluated : {len(psnr_scores)}')
print(f'  PSNR             : {np.mean(psnr_scores):.4f} dB')
print(f'  SSIM             : {np.mean(ssim_scores):.4f}')
print(f'  LPIPS            : {np.mean(lpips_scores):.4f}')
print(f'{"=" * 60}')
print(f'\nOutput images saved to: {OUTPUT_DIR}')