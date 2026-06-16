"""
DiffIR inference wrapper (DiffIR-S2 for motion deblurring).

Pre-trained weight: DiffIR-deblurring.pth
Download from:
  https://github.com/Zj-BinXia/DiffIR  (releases or Google Drive link in repo README)
Place in: weights/DiffIR-deblurring.pth

Model repo needed for arch code:
  git clone https://github.com/Zj-BinXia/DiffIR
  Set DIFFIR_REPO in config.py to that path.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
from PIL import Image

from .base_model import BaseModel
import config

# DiffIR inference timesteps (T=4 as per paper for efficiency)
_TIMESTEPS = 4


class DiffIRModel(BaseModel):
    def __init__(self):
        super().__init__('DiffIR')

    def load(self) -> None:
        self._init_device()
        print(f'\n[DiffIR] Loading model on {self.device} ...', flush=True)
        t0 = time.time()
        try:
            import torch

            repo = config.DIFFIR_REPO
            if repo and os.path.exists(repo) and repo not in sys.path:
                sys.path.insert(0, repo)

            # Import S2_arch directly via importlib to bypass DiffIR/__init__.py
            # which auto-scans all archs and causes registry conflicts with basicsr.
            print('[DiffIR] Importing architecture ...', flush=True)
            import importlib.util
            _arch_path = os.path.join(repo, 'DiffIR', 'archs', 'S2_arch.py')
            _spec = importlib.util.spec_from_file_location(
                'DiffIR.archs.S2_arch',
                _arch_path,
            )
            _mod = importlib.util.module_from_spec(_spec)
            # Pre-register the module in sys.modules BEFORE exec_module so that
            # DiffIR/archs/__init__.py (triggered by 'import DiffIR.archs.common')
            # finds S2_arch already loaded and skips re-importing it, preventing
            # the duplicate ARCH_REGISTRY registration error.
            sys.modules['DiffIR.archs.S2_arch'] = _mod
            _spec.loader.exec_module(_mod)
            DiffIRS2 = _mod.DiffIRS2

            weights_path = config.DIFFIR_WEIGHTS
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights not found: {weights_path}\n"
                    "Download DiffIR-deblurring.pth and place it in weights/"
                )

            print('[DiffIR] Building network ...', flush=True)
            self.net = DiffIRS2(
                n_encoder_res=5,
                inp_channels=3,
                out_channels=3,
                dim=48,
                num_blocks=[3, 5, 6, 6],
                num_refinement_blocks=4,
                heads=[1, 2, 4, 8],
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type='WithBias',
                n_denoise_res=1,
                linear_start=0.1,
                linear_end=0.99,
                timesteps=_TIMESTEPS,
            )

            wname = os.path.basename(weights_path)
            print(f'[DiffIR] Loading weights from {wname} (215 MB, harap tunggu) ...', flush=True)
            ckpt = torch.load(weights_path, map_location=self.device)
            state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
            self.net.load_state_dict(state, strict=False)
            self.net.to(self.device)
            self.net.eval()

            self.torch = torch
            self.loaded = True
            elapsed = round(time.time() - t0, 1)
            print(f'[DiffIR] Ready ({elapsed}s)\n', flush=True)

        except Exception as e:
            self.loaded = False
            self.load_error = str(e)
            print(f'[DiffIR] FAILED: {e}\n', flush=True)

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
            out_t = self.net(img_t, _TIMESTEPS)

        # Crop back to original size
        out_t = out_t[:, :, :h, :w]

        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)
