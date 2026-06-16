"""
Real-ESRGAN inference wrapper (scale x1 for deblurring, no upscaling).

Install: pip install realesrgan basicsr

Pre-trained weight: RealESRGAN_x4plus.pth  (we run it at scale=1)
Download from:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
Place in: weights/RealESRGAN_x4plus.pth
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
from PIL import Image

from .base_model import BaseModel
import config


class RealESRGANModel(BaseModel):
    def __init__(self):
        super().__init__('Real-ESRGAN')

    def load(self) -> None:
        self._init_device()
        print(f'\n[Real-ESRGAN] Loading model on {self.device} ...', flush=True)
        t0 = time.time()
        try:
            import torch

            print('[Real-ESRGAN] Importing architecture ...', flush=True)
            repo = config.REALESRGAN_REPO
            if repo and os.path.exists(repo) and repo not in sys.path:
                sys.path.insert(0, repo)

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            weights_path = config.REALESRGAN_WEIGHTS
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights not found: {weights_path}\n"
                    "Download RealESRGAN_x4plus.pth and place it in weights/"
                )

            print('[Real-ESRGAN] Building network ...', flush=True)
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=4,
            )

            wname = os.path.basename(weights_path)
            print(f'[Real-ESRGAN] Loading weights from {wname} ...', flush=True)
            gpu_id = 0 if self.device == 'cuda' else None
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=weights_path,
                model=net,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=gpu_id,
            )
            self.loaded = True
            elapsed = round(time.time() - t0, 1)
            print(f'[Real-ESRGAN] Ready ({elapsed}s)\n', flush=True)

        except Exception as e:
            self.loaded = False
            self.load_error = str(e)
            print(f'[Real-ESRGAN] FAILED: {e}\n', flush=True)

    def predict(self, image: Image.Image) -> Image.Image:
        import cv2

        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Pad to multiple of 8 to avoid pixel_unshuffle errors on odd dimensions
        h, w = img_bgr.shape[:2]
        factor = 8
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            img_bgr = cv2.copyMakeBorder(
                img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )

        # outscale=1 keeps original resolution (deblur only, no upscaling)
        out_bgr, _ = self.upsampler.enhance(img_bgr, outscale=1)

        # Crop back to original size
        out_bgr = out_bgr[:h, :w]

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)
