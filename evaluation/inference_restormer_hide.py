import os
import importlib.util

RESTORMER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Restormer'))

_arch_path = os.path.join(RESTORMER_ROOT, 'basicsr', 'models', 'archs', 'restormer_arch.py')
_spec = importlib.util.spec_from_file_location('restormer_arch', _arch_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Restormer = _mod.Restormer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import yaml
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from skimage import img_as_ubyte

# ── Paths ──────────────────────────────────────────────────────────────────────
WEIGHTS   = os.path.join(RESTORMER_ROOT, 'experiments', 'net_g_latest_Restormer.pth')
YAML      = os.path.join(RESTORMER_ROOT, 'Motion_Deblurring', 'Options', 'test_restormer.yml')
HIDE_ROOT = os.path.join(RESTORMER_ROOT, 'Motion_Deblurring', 'Datasets', 'HIDE_dataset', 'test')
OUTPUT    = os.path.join(os.path.dirname(__file__), 'results', 'Restormer_HIDE')

INPUT_DIRS = {
    'test-close-ups': os.path.join(HIDE_ROOT, 'test-close-ups'),
    'test-long-shot':  os.path.join(HIDE_ROOT, 'test-long-shot'),
}

os.makedirs(OUTPUT, exist_ok=True)

# ── Load model config ──────────────────────────────────────────────────────────
with open(YAML, 'r') as f:
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    cfg = yaml.load(f, Loader=Loader)

net_cfg = {k: v for k, v in cfg['network_g'].items() if k != 'type'}

# ── Build model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Restormer(**net_cfg)
checkpoint = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(checkpoint['params'])
model = model.to(device)
if device.type == 'cuda':
    model = nn.DataParallel(model)
model.eval()
print(f'Model loaded: {WEIGHTS}')

# ── Inference per subfolder ────────────────────────────────────────────────────
factor = 8

for split_name, input_dir in INPUT_DIRS.items():
    files = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))
    print(f'\n[{split_name}] {len(files)} images')

    with torch.no_grad():
        for file_ in tqdm(files, desc=split_name):
            if device.type == 'cuda':
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            img    = np.float32(cv2.cvtColor(cv2.imread(file_), cv2.COLOR_BGR2RGB)) / 255.
            input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            h, w   = input_.shape[2], input_.shape[3]
            padh   = (factor - h % factor) % factor
            padw   = (factor - w % factor) % factor
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model(input_)
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            out_name = f"{split_name}_{os.path.splitext(os.path.basename(file_))[0]}.png"
            cv2.imwrite(os.path.join(OUTPUT, out_name), cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR))

print(f'\nDone! Hasil tersimpan di: {OUTPUT}')
