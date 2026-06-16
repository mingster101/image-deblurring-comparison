"""
Inference script untuk evaluasi KADID-10k.
Menjalankan Restormer, DiffIR, dan Real-ESRGAN pada seluruh gambar distorsi KADID-10k.

Hasil disimpan di: evaluation/results/kadid10k/{ModelName}/
"""

import os
import sys
import csv
import re
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from skimage import img_as_ubyte

# ── Root paths ────────────────────────────────────────────────────────────────
EVAL_ROOT     = os.path.dirname(os.path.abspath(__file__))
SKRIPSI_ROOT  = os.path.dirname(EVAL_ROOT)

RESTORMER_ROOT = os.path.join(SKRIPSI_ROOT, 'Restormer')
DIFFIR_ROOT    = os.path.join(SKRIPSI_ROOT, 'DiffIR', 'DiffIR-demotionblur')
REALESRGAN_ROOT = os.path.join(SKRIPSI_ROOT, 'Real-ESRGAN')

KADID_IMAGES = os.path.join(EVAL_ROOT, 'kadid10k', 'images')
KADID_CSV    = os.path.join(EVAL_ROOT, 'kadid10k', 'dmos.csv')
RESULTS_ROOT = os.path.join(EVAL_ROOT, 'results', 'kadid10k')

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_distorted_images(csv_path, images_dir):
    """Baca CSV dan kembalikan list (dist_path, ref_path, dist_name, ref_name)."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_path = os.path.join(images_dir, row['dist_img'])
            ref_path  = os.path.join(images_dir, row['ref_img'])
            if os.path.exists(dist_path) and os.path.exists(ref_path):
                rows.append((dist_path, ref_path, row['dist_img'], row['ref_img']))
    return rows


def pad_to_multiple(tensor, factor=8):
    h, w   = tensor.shape[2], tensor.shape[3]
    padh   = (factor - h % factor) % factor
    padw   = (factor - w % factor) % factor
    if padh > 0 or padw > 0:
        tensor = F.pad(tensor, (0, padw, 0, padh), 'reflect')
    return tensor, h, w


def img_to_tensor(img_path, device):
    """Baca gambar BGR → float32 tensor [0,1], shape (1,3,H,W)."""
    img = np.float32(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)) / 255.
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_bgr(tensor):
    """Tensor (1,3,H,W) clamp [0,1] → uint8 BGR numpy."""
    arr = torch.clamp(tensor, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    return cv2.cvtColor(img_as_ubyte(arr), cv2.COLOR_RGB2BGR)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_restormer(device):
    import yaml
    arch_path = os.path.join(RESTORMER_ROOT, 'basicsr', 'models', 'archs', 'restormer_arch.py')
    spec = importlib.util.spec_from_file_location('restormer_arch', arch_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Restormer = mod.Restormer

    yaml_path = os.path.join(RESTORMER_ROOT, 'Motion_Deblurring', 'Options', 'test_restormer.yml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    net_cfg = {k: v for k, v in cfg['network_g'].items() if k != 'type'}

    weights = os.path.join(RESTORMER_ROOT, 'experiments', 'net_g_latest_Restormer.pth')
    model   = Restormer(**net_cfg)
    ckpt    = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['params'])
    model   = model.to(device)
    if device.type == 'cuda':
        model = nn.DataParallel(model)
    model.eval()
    print(f'  [Restormer] Weights: {os.path.basename(weights)}')
    return model


def load_diffir(device):
    import yaml
    if DIFFIR_ROOT not in sys.path:
        sys.path.insert(0, DIFFIR_ROOT)
    from DiffIR.archs.S2_arch import DiffIRS2

    yaml_path = os.path.join(DIFFIR_ROOT, 'options', 'test_DiffIRS2_finetune.yml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    net_cfg = {k: v for k, v in cfg['network_g'].items() if k != 'type'}

    weights = os.path.join(DIFFIR_ROOT, 'experiments',
                           'train_DiffIRS2_FT_LR5e5', 'models', 'net_g_40000.pth')
    model   = DiffIRS2(**net_cfg)
    ckpt    = torch.load(weights, map_location=device, weights_only=False)
    state   = ckpt.get('params_ema', ckpt.get('params', ckpt))
    model.load_state_dict(state, strict=False)
    model   = model.to(device)
    model.eval()
    print(f'  [DiffIR] Weights: {os.path.basename(weights)}')
    return model


def load_realesrgan(device):
    from basicsr.archs.rrdbnet_arch import RRDBNet

    weights = os.path.join(REALESRGAN_ROOT, 'experiments',
                           'finetune_RealESRGAN_deblur', 'models', 'net_g_latest.pth')
    # scale=1: BasicSR RRDBNet(scale=1) melakukan pixel_unshuffle(4) di awal sehingga
    # conv_first menerima num_in_ch*16 = 3*16 = 48 channel (sesuai checkpoint).
    # Output model berukuran sama dengan input — tidak perlu resize manual.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=1)
    ckpt  = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt.get('params_ema', ckpt.get('params', ckpt))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f'  [Real-ESRGAN] Weights: {os.path.basename(weights)}')
    return model


# ── Inference functions ───────────────────────────────────────────────────────

def infer_restormer(model, img_path, device):
    inp = img_to_tensor(img_path, device)
    inp, h, w = pad_to_multiple(inp, 8)
    with torch.no_grad():
        out = model(inp)
    return tensor_to_bgr(out[:, :, :h, :w])


def infer_diffir(model, img_path, device):
    inp = img_to_tensor(img_path, device)
    inp, h, w = pad_to_multiple(inp, 8)
    with torch.no_grad():
        out = model(inp)
    return tensor_to_bgr(out[:, :, :h, :w])


def infer_realesrgan(model, img_path, device):
    """
    RRDBNet(scale=1): pixel_unshuffle(4) di forward → output = ukuran input.
    Cukup pad ke kelipatan 4 agar pixel_unshuffle tidak error.
    """
    inp = img_to_tensor(img_path, device)
    inp, h, w = pad_to_multiple(inp, factor=4)
    with torch.no_grad():
        out = model(inp)   # output: sama dengan ukuran input (setelah padding)
    return tensor_to_bgr(out[:, :, :h, :w])


# ── Run loop ──────────────────────────────────────────────────────────────────

def run_model(model_name, infer_fn, image_list, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0

    for dist_path, _, dist_name, _ in tqdm(image_list, desc=f'  {model_name}'):
        out_path = os.path.join(output_dir, dist_name)
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            result = infer_fn(dist_path)
            cv2.imwrite(out_path, result)
        except Exception as e:
            tqdm.write(f'  [WARN] {dist_name}: {e}')

    done = len(image_list) - skipped
    print(f'  Selesai: {done} diproses, {skipped} dilewati (sudah ada)')
    print(f'  Tersimpan di: {output_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    print(f'\nMembaca KADID-10k CSV ...')
    image_list = get_distorted_images(KADID_CSV, KADID_IMAGES)
    print(f'Jumlah gambar distorsi : {len(image_list)}')

    # ── Restormer ─────────────────────────────────────────────────────────────
    sep = '=' * 55
    print(f'\n{sep}')
    print('  FASE 1 — RESTORMER INFERENCE')
    print(sep)
    restormer = load_restormer(device)
    run_model(
        'Restormer',
        lambda p: infer_restormer(restormer, p, device),
        image_list,
        os.path.join(RESULTS_ROOT, 'Restormer'),
        device,
    )
    del restormer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── DiffIR ────────────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print('  FASE 2 — DIFFIR INFERENCE')
    print(sep)
    diffir = load_diffir(device)
    run_model(
        'DiffIR',
        lambda p: infer_diffir(diffir, p, device),
        image_list,
        os.path.join(RESULTS_ROOT, 'DiffIR'),
        device,
    )
    del diffir
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Real-ESRGAN ───────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print('  FASE 3 — REAL-ESRGAN INFERENCE')
    print(sep)
    realesrgan = load_realesrgan(device)
    run_model(
        'Real-ESRGAN',
        lambda p: infer_realesrgan(realesrgan, p, device),
        image_list,
        os.path.join(RESULTS_ROOT, 'Real-ESRGAN'),
        device,
    )
    del realesrgan

    print(f'\n{"=" * 55}')
    print('  SEMUA INFERENCE SELESAI')
    print(f'  Hasil di : {RESULTS_ROOT}')
    print(f'{"=" * 55}')
    print('Langkah berikutnya: jalankan kadid10k_analysis.py')
