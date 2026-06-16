"""
Restormer inference wrapper.

Fine-tuned weight (hasil training skripsi): net_g_latest_Restormer.pth
Place in: weights/net_g_latest_Restormer.pth

Model repo (needed for arch code):
  git clone https://github.com/swz30/Restormer
  Set RESTORMER_REPO in config.py to that path.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
from PIL import Image

from .base_model import BaseModel
import config


class RestormerModel(BaseModel):
    def __init__(self):
        super().__init__('Restormer')

    def load(self) -> None:
        self._init_device()
        print(f'\n[Restormer] Loading model on {self.device} ...', flush=True)
        t0 = time.time()
        try:
            import torch

            print('[Restormer] Importing architecture ...', flush=True)
            repo = config.RESTORMER_REPO
            # Import restormer_arch.py directly via file path to avoid
            # conflicts with the installed basicsr package
            import importlib.util
            arch_file = os.path.join(repo, 'basicsr', 'models', 'archs', 'restormer_arch.py')
            if not os.path.exists(arch_file):
                raise FileNotFoundError(
                    f"restormer_arch.py not found at: {arch_file}\n"
                    "Set RESTORMER_REPO in config.py to the cloned Restormer repo path."
                )
            _spec = importlib.util.spec_from_file_location('restormer_arch', arch_file)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            RestormerArch = _mod.Restormer

            weights_path = config.RESTORMER_WEIGHTS
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights not found: {weights_path}\n"
                    "Place net_g_latest_Restormer.pth in weights/"
                )

            print('[Restormer] Building network ...', flush=True)
            self.net = RestormerArch(
                inp_channels=3,
                out_channels=3,
                dim=48,
                num_blocks=[4, 6, 6, 8],
                num_refinement_blocks=4,
                heads=[1, 2, 4, 8],
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                dual_pixel_task=False,
            )

            wname = os.path.basename(weights_path)
            print(f'[Restormer] Loading weights from {wname} ...', flush=True)
            ckpt = torch.load(weights_path, map_location=self.device)
            self.net.load_state_dict(ckpt['params'])
            self.net.to(self.device)
            self.net.eval()

            self.torch = torch
            self.loaded = True
            elapsed = round(time.time() - t0, 1)
            print(f'[Restormer] Ready ({elapsed}s)\n', flush=True)

        except Exception as e:
            self.loaded = False
            self.load_error = str(e)
            print(f'[Restormer] FAILED: {e}\n', flush=True)

    def predict(self, image: Image.Image) -> Image.Image:
        import torch.nn.functional as F

        img_np = np.array(image).astype(np.float32) / 255.0
        img_t = self.torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to multiple of 8 (required by encoder-decoder architecture)
        _, _, h, w = img_t.shape
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')

        with self.torch.no_grad():
            out_t = self.net(img_t)

        # Crop back to original size
        out_t = out_t[:, :, :h, :w]

        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)
