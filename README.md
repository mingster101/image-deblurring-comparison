# Image Deblurring Comparison: Restormer, DiffIR, Real-ESRGAN

Repositori ini berisi konfigurasi training, konfigurasi inference, dan skrip evaluasi yang digunakan dalam penelitian skripsi perbandingan model image deblurring.

## Deskripsi

Penelitian ini membandingkan tiga model deep learning untuk task **motion deblurring** pada dataset GoPro dan HIDE, serta evaluasi generalisasi pada dataset KADID-10k (25 jenis distorsi).

| Model | Arsitektur | Paper |
|-------|-----------|-------|
| **Restormer** | Transformer-based | [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881) |
| **DiffIR** | Diffusion-based | [DiffIR: Efficient Diffusion Model for Image Restoration](https://arxiv.org/abs/2303.09472) |
| **Real-ESRGAN** | GAN-based | [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833) |

---

## Struktur Repositori

```
├── configs/
│   ├── restormer/
│   │   ├── Deblurring_Restormer.yml   # Konfigurasi fine-tuning Restormer
│   │   └── test_restormer.yml         # Konfigurasi inference Restormer
│   ├── diffir/
│   │   ├── train_DiffIRS2.yml         # Konfigurasi fine-tuning DiffIR Stage 2
│   │   └── test_DiffIRS2_finetune.yml # Konfigurasi inference DiffIR
│   └── realesrgan/
│       └── finetune_realesrgan_x4plus_pairdata.yml  # Konfigurasi fine-tuning Real-ESRGAN
├── evaluation/
│   ├── inference_restormer.py         # Inference Restormer pada GoPro test set
│   ├── inference_restormer_hide.py    # Inference Restormer pada HIDE dataset
│   ├── inference_diffir.py            # Inference DiffIR pada HIDE dataset
│   ├── kadid10k_inference.py          # Inference ketiga model pada KADID-10k
│   ├── evaluation_metrics.py          # Hitung PSNR, SSIM, LPIPS (GoPro & HIDE)
│   ├── kadid10k_analysis.py           # Analisis & visualisasi hasil KADID-10k
│   ├── unsharp_mask.py                # Baseline unsharp mask
│   └── lilscript.py                   # Utility script
└── scripts/
    ├── evaluate_laplacian.py          # Evaluasi ketajaman berbasis Laplacian
    ├── diffusion_test.py              # Eksperimen diffusion model
    └── app.py                         # Aplikasi demo inference
```

---

## Kode Training (Repo Asli)

Training dilakukan menggunakan kode dari repo publik berikut **tanpa modifikasi**. Konfigurasi yang dipakai ada di folder `configs/` repositori ini.

| Model | Repo | Training Script |
|-------|------|----------------|
| Restormer | [swz30/Restormer](https://github.com/swz30/Restormer) | `basicsr/train.py -opt configs/restormer/Deblurring_Restormer.yml` |
| DiffIR | [Zj-BinXia/DiffIR](https://github.com/Zj-BinXia/DiffIR) | `DiffIR/train.py -opt configs/diffir/train_DiffIRS2.yml` |
| Real-ESRGAN | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `realesrgan/train.py -opt configs/realesrgan/finetune_realesrgan_x4plus_pairdata.yml` |

---

## Dataset

| Dataset | Digunakan untuk | Link |
|---------|----------------|------|
| **GoPro Large** | Training & evaluasi utama | [Download](https://seungjunnah.github.io/Datasets/gopro) |
| **HIDE** | Evaluasi generalisasi (motion blur) | [Download](https://github.com/joanshen0508/HA_deblur) |
| **KADID-10k** | Evaluasi generalisasi (25 distorsi) | [Download](http://database.mmsp-kn.de/kadid-10k-database.html) |

Dataset **tidak disertakan** di repo ini karena ukurannya besar. Unduh dan letakkan sesuai path yang tertulis di masing-masing file konfigurasi.

---

## Pretrained Models

Pretrained weights yang digunakan sebagai titik awal fine-tuning:

| Model | Weights | Sumber |
|-------|---------|--------|
| Restormer | `motion_deblurring.pth` | [Google Drive (Restormer repo)](https://github.com/swz30/Restormer) |
| DiffIR S1 | `Deblurring-DiffIRS1.pth` | [Google Drive (DiffIR repo)](https://github.com/Zj-BinXia/DiffIR) |
| DiffIR S2 | `Deblurring-DiffIRS2.pth` | [Google Drive (DiffIR repo)](https://github.com/Zj-BinXia/DiffIR) |
| Real-ESRGAN | `RealESRNet_x4plus.pth` | [Google Drive (Real-ESRGAN repo)](https://github.com/xinntao/Real-ESRGAN) |

---

## Metrik Evaluasi

- **PSNR** (Peak Signal-to-Noise Ratio) — lebih tinggi lebih baik
- **SSIM** (Structural Similarity Index) — lebih tinggi lebih baik
- **LPIPS** (Learned Perceptual Image Patch Similarity) — lebih rendah lebih baik

