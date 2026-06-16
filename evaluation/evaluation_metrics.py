import os
import glob
import torch
import lpips
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ============================================================
# KONFIGURASI DATASET
# ============================================================
DATASETS = {
    'GoPro': r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\GOPRO_Large_flat\test\sharp',
    'HIDE':  r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer\Motion_Deblurring\Datasets\HIDE_dataset\GT',
}

MODEL_OUTPUTS = {
    'GoPro': {
        'DiffIR':     r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\DiffIR\DiffIR-demotionblur\results\test_DiffIRS2_finetune\visualization\GoPro',
        'Real-ESRGAN': r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Evaluation\results\realesrgan',
        'Restormer':  r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\evaluation\results\Restormer',  # isi setelah inference selesai
    },
    'HIDE': {
        'DiffIR':    r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Evaluation\results\DiffIR',
        'Real-ESRGAN': r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Evaluation\results\real-esrgan',
        'Restormer':  r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Evaluation\results\Restormer_HIDE',   # isi setelah inference selesai
    },
}
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

loss_fn = lpips.LPIPS(net='alex').to(device)


def load_image_np(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_tensor(img_np):
    img = img_np.astype(np.float32) / 255.0
    img = img * 2 - 1
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_t


def get_gt_filename(filename):
    base_name = filename
    # buang prefix subfolder HIDE yang ditambahkan saat inference
    for prefix in ['test-close-ups_', 'test-long-shot_']:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break
    for suffix in ['_test_DiffIRS2_finetune', '_test_DiffIRS2', '_out']:
        base_name = base_name.replace(suffix, '')
    return base_name


def evaluate(model_name, dataset_name, output_dir, gt_dir):
    output_files = sorted(glob.glob(os.path.join(output_dir, '*.png')))

    if len(output_files) == 0:
        print(f'[{model_name} - {dataset_name}] No images found in {output_dir}')
        return None

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    missing = 0

    for out_path in tqdm(output_files, desc=f'[{model_name} - {dataset_name}]'):
        filename = os.path.basename(out_path)
        base_name = get_gt_filename(filename)
        gt_path = os.path.join(gt_dir, base_name)

        if not os.path.exists(gt_path):
            missing += 1
            continue

        out_np = load_image_np(out_path)
        gt_np  = load_image_np(gt_path)

        # Resize jika ukuran berbeda
        if out_np.shape != gt_np.shape:
            gt_np = cv2.resize(gt_np, (out_np.shape[1], out_np.shape[0]))

        # PSNR
        psnr_val = psnr(gt_np, out_np, data_range=255)
        psnr_scores.append(psnr_val)

        # SSIM
        ssim_val = ssim(gt_np, out_np, channel_axis=2, data_range=255)
        ssim_scores.append(ssim_val)

        # LPIPS
        out_t = load_image_tensor(out_np)
        gt_t  = load_image_tensor(gt_np)
        with torch.no_grad():
            lpips_val = loss_fn(out_t, gt_t).item()
        lpips_scores.append(lpips_val)

    if missing > 0:
        print(f'  Warning: {missing} GT files not found')

    result = {
        'psnr':  np.mean(psnr_scores),
        'ssim':  np.mean(ssim_scores),
        'lpips': np.mean(lpips_scores),
        'count': len(psnr_scores),
    }

    print(f'\n[{model_name} - {dataset_name}]')
    print(f'  Images evaluated : {result["count"]}')
    print(f'  PSNR             : {result["psnr"]:.4f} dB')
    print(f'  SSIM             : {result["ssim"]:.4f}')
    print(f'  LPIPS            : {result["lpips"]:.4f}')
    return result


if __name__ == '__main__':
    print('=' * 60)
    print('  Image Deblurring Evaluation (PSNR / SSIM / LPIPS)')
    print('=' * 60)

    all_results = {}

    for dataset_name, gt_dir in DATASETS.items():
        for model_name, output_dir in MODEL_OUTPUTS[dataset_name].items():
            if 'PATH_TO' in output_dir:
                print(f'\n[SKIP] {model_name} - {dataset_name}: path belum diisi')
                continue
            key = f'{model_name}_{dataset_name}'
            all_results[key] = evaluate(model_name, dataset_name, output_dir, gt_dir)

    # Summary table
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  {"Model":<15} {"Dataset":<10} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8}')
    print(f'  {"-"*15} {"-"*10} {"-"*8} {"-"*8} {"-"*8}')
    for key, res in all_results.items():
        if res is not None:
            model, dataset = key.rsplit('_', 1)
            print(f'  {model:<15} {dataset:<10} {res["psnr"]:>8.4f} {res["ssim"]:>8.4f} {res["lpips"]:>8.4f}')
    print('=' * 60)
    print('\nNB: PSNR/SSIM lebih tinggi = lebih baik | LPIPS lebih rendah = lebih baik')