# Setup Guide — Image Deblurring Website

## 1. Install Python (jika belum ada)

Download **Python 3.10** dari https://www.python.org/downloads/  
Saat install, centang **"Add Python to PATH"**.

---

## 2. Buat virtual environment

Buka **Command Prompt** di folder `Web_skripsi/`, lalu:

```bat
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

```bat
pip install flask Pillow numpy opencv-python-headless
```

Install PyTorch (pilih versi CUDA jika punya GPU NVIDIA, CPU jika tidak):

```bat
# CPU only (aman untuk semua laptop):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (jika ada GPU NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install model dependencies:

```bat
pip install basicsr realesrgan
```

---

## 4. Download pre-trained weights

Taruh file `.pth` di folder `weights/`:

| File | Link |
|------|------|
| `motion_deblurring.pth` | [Restormer GitHub Releases](https://github.com/swz30/Restormer/releases) |
| `RealESRGAN_x4plus.pth` | [Real-ESRGAN GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases) |
| `DiffIR-deblurring.pth` | [DiffIR Google Drive (see repo README)](https://github.com/Zj-BinXia/DiffIR) |

---

## 5. Set path repo model di `config.py`

Edit file `config.py`:

```python
RESTORMER_REPO  = r'C:\Users\Asus\Restormer'    # path clone repo Restormer
DIFFIR_REPO     = r'C:\Users\Asus\DiffIR'        # path clone repo DiffIR
REALESRGAN_REPO = r''                             # kosong, sudah via pip
```

---

## 6. Jalankan website

```bat
python app.py
```

Buka browser: **http://localhost:5000**

---

## Struktur folder akhir

```
Web_skripsi/
├── app.py
├── config.py
├── requirements.txt
├── models/
│   ├── restormer_model.py
│   ├── realesrgan_model.py
│   └── diffir_model.py
├── utils/
│   └── metrics.py
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/main.js
└── weights/
    ├── motion_deblurring.pth       ← download manual
    ├── RealESRGAN_x4plus.pth       ← download manual
    └── DiffIR-deblurring.pth       ← download manual
```
