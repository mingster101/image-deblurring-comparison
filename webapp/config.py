import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

# === PATHS TO MODEL REPOSITORIES ===
# Edit these to point to where you cloned each repo.
# Leave as empty string if the repo is already in PYTHONPATH.
RESTORMER_REPO  = os.getenv('RESTORMER_REPO',  r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\Restormer')
DIFFIR_REPO     = os.getenv('DIFFIR_REPO',     r'C:\Users\MM-417\Desktop\Punyamingsterplsjgndisentuh\EksplorasiSkripsi\DiffIR\DiffIR-demotionblur')
REALESRGAN_REPO = os.getenv('REALESRGAN_REPO', r'')

# === FINE-TUNED WEIGHT FILES (hasil training skripsi) ===
RESTORMER_WEIGHTS  = os.path.join(WEIGHTS_DIR, 'net_g_latest_Restormer.pth')
REALESRGAN_WEIGHTS = os.path.join(WEIGHTS_DIR, 'Real-ESRGAN_latest.pth')
DIFFIR_WEIGHTS     = os.path.join(WEIGHTS_DIR, 'DiffIR_deblur.pth')

# === INFERENCE SETTINGS ===
MAX_IMAGE_SIZE = 1280   # Resize longest edge to this before inference (saves VRAM)
DEVICE = 'auto'         # 'auto' = use CUDA if available, else CPU
