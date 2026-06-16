"""
Analisis evaluasi KADID-10k.

Fase 1 — Komputasi metrik (PSNR, SSIM, LPIPS) per gambar, disimpan ke CSV.
Fase 2 — Agregasi per tipe distorsi, level, dan kelompok.
Fase 3 — Visualisasi dan ringkasan insight teks.

Jalankan SETELAH kadid10k_inference.py selesai.
"""

import os
import csv
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lpips
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_ROOT    = os.path.dirname(os.path.abspath(__file__))
KADID_IMAGES = os.path.join(EVAL_ROOT, 'kadid10k', 'images')
KADID_CSV    = os.path.join(EVAL_ROOT, 'kadid10k', 'dmos.csv')
RESULTS_ROOT = os.path.join(EVAL_ROOT, 'results', 'kadid10k')
OUT_DIR      = os.path.join(EVAL_ROOT, 'kadid10k_analysis')
PLOTS_DIR    = os.path.join(OUT_DIR, 'plots')
METRICS_CSV  = os.path.join(OUT_DIR, 'kadid10k_metrics.csv')
INSIGHTS_TXT = os.path.join(OUT_DIR, 'insights_summary.txt')

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Label KADID-10k ───────────────────────────────────────────────────────────
DISTORTION_NAMES = {
     1: 'Gaussian Blur',
     2: 'Lens Blur',
     3: 'Motion Blur',
}

DISTORTION_GROUPS = {
    'Blur': [1, 2, 3],
}

TYPE_TO_GROUP = {t: g for g, types in DISTORTION_GROUPS.items() for t in types}

MODELS        = ['Restormer', 'DiffIR', 'Real-ESRGAN']
MODEL_COLORS  = {'Restormer': '#2196F3', 'DiffIR': '#4CAF50', 'Real-ESRGAN': '#FF5722'}
MODEL_MARKERS = {'Restormer': 'o', 'DiffIR': 's', 'Real-ESRGAN': '^'}

SHORT = {  # singkatan label untuk sumbu yang sempit
    'Gaussian Blur': 'GaussBlur', 'Lens Blur': 'LensBlur', 'Motion Blur': 'MotBlur',
}

# ── Utilitas ──────────────────────────────────────────────────────────────────

def parse_distortion(filename):
    """Ekstrak (dist_type, dist_level) dari nama file seperti I01_03_02.png."""
    m = re.match(r'I\d+_(\d+)_(\d+)\.png', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Tidak dapat membaca: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_psnr(a, b):
    return psnr_fn(a, b, data_range=255)


def compute_ssim(a, b):
    return ssim_fn(a, b, channel_axis=2, data_range=255)


def to_lpips_tensor(img_rgb, device):
    t = img_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — KOMPUTASI METRIK
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Cek model output yang tersedia
    available_models = [m for m in MODELS
                        if os.path.isdir(os.path.join(RESULTS_ROOT, m))]
    if not available_models:
        print('[ERROR] Tidak ada hasil inference ditemukan di:', RESULTS_ROOT)
        print('Jalankan kadid10k_inference.py terlebih dahulu.')
        return None
    print(f'Model tersedia : {available_models}')

    # Baca CSV
    rows = []
    with open(KADID_CSV, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Cek apakah metrics CSV sudah ada (untuk resume)
    existing_keys = set()
    if os.path.exists(METRICS_CSV):
        df_ex = pd.read_csv(METRICS_CSV)
        existing_keys = set(df_ex['dist_img'].tolist())
        print(f'Resume: {len(existing_keys)} gambar sudah dihitung.')

    fieldnames = ['dist_img', 'ref_img', 'dist_type', 'dist_level', 'dist_group',
                  'dmos', 'baseline_psnr', 'baseline_ssim', 'baseline_lpips']
    for m in available_models:
        fieldnames += [f'{m}_psnr', f'{m}_ssim', f'{m}_lpips']

    write_header = not os.path.exists(METRICS_CSV)
    csv_file = open(METRICS_CSV, 'a', newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for row in tqdm(rows, desc='Menghitung metrik'):
        dist_name = row['dist_img']
        if dist_name in existing_keys:
            continue

        dist_type, dist_level = parse_distortion(dist_name)
        if dist_type is None:
            continue

        # Hanya proses 3 distorsi blur
        if dist_type not in DISTORTION_NAMES:
            continue

        ref_path  = os.path.join(KADID_IMAGES, row['ref_img'])
        dist_path = os.path.join(KADID_IMAGES, dist_name)
        if not os.path.exists(ref_path) or not os.path.exists(dist_path):
            continue

        try:
            ref_rgb  = load_rgb(ref_path)
            dist_rgb = load_rgb(dist_path)
        except Exception:
            continue

        record = {
            'dist_img'   : dist_name,
            'ref_img'    : row['ref_img'],
            'dist_type'  : dist_type,
            'dist_level' : dist_level,
            'dist_group' : TYPE_TO_GROUP.get(dist_type, 'Unknown'),
            'dmos'       : float(row['dmos']),
        }

        # Baseline: distorted vs reference
        try:
            record['baseline_psnr'] = compute_psnr(ref_rgb, dist_rgb)
            record['baseline_ssim'] = compute_ssim(ref_rgb, dist_rgb)
            with torch.no_grad():
                record['baseline_lpips'] = loss_fn(
                    to_lpips_tensor(ref_rgb, device),
                    to_lpips_tensor(dist_rgb, device)
                ).item()
        except Exception:
            record.update({'baseline_psnr': np.nan, 'baseline_ssim': np.nan,
                           'baseline_lpips': np.nan})

        # Per-model
        for model_name in available_models:
            out_path = os.path.join(RESULTS_ROOT, model_name, dist_name)
            if not os.path.exists(out_path):
                record.update({f'{model_name}_psnr': np.nan,
                                f'{model_name}_ssim': np.nan,
                                f'{model_name}_lpips': np.nan})
                continue
            try:
                out_rgb = load_rgb(out_path)
                if out_rgb.shape != ref_rgb.shape:
                    out_rgb = cv2.resize(out_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]))
                record[f'{model_name}_psnr'] = compute_psnr(ref_rgb, out_rgb)
                record[f'{model_name}_ssim'] = compute_ssim(ref_rgb, out_rgb)
                with torch.no_grad():
                    record[f'{model_name}_lpips'] = loss_fn(
                        to_lpips_tensor(ref_rgb, device),
                        to_lpips_tensor(out_rgb, device)
                    ).item()
            except Exception as e:
                tqdm.write(f'[WARN] {model_name}/{dist_name}: {e}')
                record.update({f'{model_name}_psnr': np.nan,
                                f'{model_name}_ssim': np.nan,
                                f'{model_name}_lpips': np.nan})

        # Isi NaN untuk model yang tidak tersedia
        for m in MODELS:
            if m not in available_models:
                for metric in ['psnr', 'ssim', 'lpips']:
                    record.setdefault(f'{m}_{metric}', np.nan)

        writer.writerow(record)

    csv_file.close()
    print(f'\nMetrik disimpan di: {METRICS_CSV}')
    return pd.read_csv(METRICS_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — AGREGASI
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(df):
    available_models = [m for m in MODELS if f'{m}_psnr' in df.columns
                        and df[f'{m}_psnr'].notna().any()]

    # Hitung delta (improvement vs baseline)
    for m in available_models:
        df[f'{m}_delta_psnr']  = df[f'{m}_psnr']  - df['baseline_psnr']
        df[f'{m}_delta_ssim']  = df[f'{m}_ssim']  - df['baseline_ssim']
        df[f'{m}_delta_lpips'] = df['baseline_lpips'] - df[f'{m}_lpips']  # turun = lebih baik

    # Per distortion type
    type_agg = {}
    for dt in sorted(DISTORTION_NAMES.keys()):
        sub = df[df['dist_type'] == dt]
        if sub.empty:
            continue
        entry = {'name': DISTORTION_NAMES[dt], 'group': TYPE_TO_GROUP.get(dt, '?'), 'n': len(sub)}
        for m in available_models:
            for metric in ['psnr', 'ssim', 'lpips', 'delta_psnr', 'delta_ssim', 'delta_lpips']:
                col = f'{m}_{metric}'
                if col in sub.columns:
                    entry[col] = sub[col].mean()
        type_agg[dt] = entry

    # Per level (1-5)
    level_agg = {}
    for lv in range(1, 6):
        sub = df[df['dist_level'] == lv]
        entry = {'n': len(sub)}
        for m in available_models:
            for metric in ['psnr', 'ssim', 'lpips']:
                col = f'{m}_{metric}'
                if col in sub.columns:
                    entry[col] = sub[col].mean()
        level_agg[lv] = entry

    # Per group
    group_agg = {}
    for grp in DISTORTION_GROUPS:
        sub = df[df['dist_group'] == grp]
        if sub.empty:
            continue
        entry = {'n': len(sub)}
        for m in available_models:
            for metric in ['psnr', 'ssim', 'lpips', 'delta_psnr']:
                col = f'{m}_{metric}'
                if col in sub.columns:
                    entry[col] = sub[col].mean()
        group_agg[grp] = entry

    # Overall
    overall = {}
    for m in available_models:
        overall[m] = {
            'psnr' : df[f'{m}_psnr'].mean(),
            'ssim' : df[f'{m}_ssim'].mean(),
            'lpips': df[f'{m}_lpips'].mean(),
            'delta_psnr': df[f'{m}_delta_psnr'].mean(),
        }

    return available_models, type_agg, level_agg, group_agg, overall


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3A — VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(type_agg, available_models, metric, title, cmap, fname):
    """Heatmap: Model (baris) × Distortion Type (kolom)."""
    types  = sorted(type_agg.keys())
    labels = [SHORT.get(type_agg[t]['name'], type_agg[t]['name']) for t in types]
    data   = np.array([[type_agg[t].get(f'{m}_{metric}', np.nan) for t in types]
                       for m in available_models])

    fig, ax = plt.subplots(figsize=(18, max(3, len(available_models) * 1.4)))
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(available_models)))
    ax.set_yticklabels(available_models)
    # Annotate values
    for i in range(len(available_models)):
        for j in range(len(types)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if abs(val - np.nanmean(data)) > np.nanstd(data) else 'black')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    # Add group separators
    group_ends = {}
    for t in types:
        grp = type_agg[t]['group']
        group_ends[grp] = types.index(t)
    prev_grp = type_agg[types[0]]['group']
    for t in types[1:]:
        grp = type_agg[t]['group']
        if grp != prev_grp:
            x_pos = types.index(t) - 0.5
            ax.axvline(x=x_pos, color='white', linewidth=1.5, linestyle='--')
            prev_grp = grp
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_performance_vs_level(df, available_models, metric, ylabel, fname, higher_better=True):
    """Line plot: metrik vs distortion level (1–5)."""
    fig, axes = plt.subplots(1, len(DISTORTION_GROUPS), figsize=(18, 4), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, (grp, types) in zip(axes, DISTORTION_GROUPS.items()):
        sub_grp = df[df['dist_group'] == grp]
        for m in available_models:
            col = f'{m}_{metric}'
            if col not in sub_grp.columns:
                continue
            y_vals = [sub_grp[sub_grp['dist_level'] == lv][col].mean() for lv in range(1, 6)]
            ax.plot(range(1, 6), y_vals,
                    color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                    label=m, linewidth=1.8, markersize=5)
        # Baseline
        bl_col = f'baseline_{metric}'
        if bl_col in sub_grp.columns:
            bl_vals = [sub_grp[sub_grp['dist_level'] == lv][bl_col].mean() for lv in range(1, 6)]
            ax.plot(range(1, 6), bl_vals,
                    color='gray', linestyle='--', linewidth=1.2, label='Baseline', alpha=0.7)
        ax.set_title(grp, fontsize=9, fontweight='bold')
        ax.set_xlabel('Distortion Level', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)
    # Legend
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                      label=m, linewidth=1.8, markersize=5)
               for m in available_models]
    handles.append(Line2D([0], [0], color='gray', linestyle='--', label='Baseline'))
    axes[-1].legend(handles=handles, fontsize=7, loc='best')
    plt.suptitle(f'{ylabel} vs Distortion Level per Group', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_radar(group_agg, available_models, metric, title, fname, higher_better=True):
    """Radar chart per distortion group."""
    groups = list(DISTORTION_GROUPS.keys())
    N = len(groups)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})

    for m in available_models:
        vals = [group_agg.get(g, {}).get(f'{m}_{metric}', np.nan) for g in groups]
        vals_plot = [v if not np.isnan(v) else 0 for v in vals]
        vals_plot += vals_plot[:1]
        ax.plot(angles, vals_plot, color=MODEL_COLORS[m], linewidth=2, label=m)
        ax.fill(angles, vals_plot, color=MODEL_COLORS[m], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_best_model_per_type(type_agg, available_models, metric, fname, higher_better=True):
    """Bar chart: PSNR terbaik per distortion type, warna = model terbaik."""
    types  = sorted(type_agg.keys())
    labels = [SHORT.get(type_agg[t]['name'], type_agg[t]['name']) for t in types]
    bars   = []
    colors = []
    winners = []
    for t in types:
        vals = {m: type_agg[t].get(f'{m}_{metric}', np.nan) for m in available_models}
        valid = {m: v for m, v in vals.items() if not np.isnan(v)}
        if not valid:
            bars.append(0); colors.append('gray'); winners.append('N/A')
            continue
        winner = max(valid, key=valid.get) if higher_better else min(valid, key=valid.get)
        bars.append(valid[winner])
        colors.append(MODEL_COLORS[winner])
        winners.append(winner)

    fig, ax = plt.subplots(figsize=(18, 5))
    x = range(len(types))
    rects = ax.bar(x, bars, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel(f'Best {metric.upper()}', fontsize=10)
    ax.set_title(f'Best {metric.upper()} per Distortion Type (warna = model terbaik)',
                 fontsize=11, fontweight='bold')
    # Legend
    legend_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in available_models]
    ax.legend(handles=legend_patches, fontsize=9)
    # Group separators
    prev_grp = type_agg[types[0]]['group']
    for i, t in enumerate(types[1:], 1):
        grp = type_agg[t]['group']
        if grp != prev_grp:
            ax.axvline(x=i - 0.5, color='gray', linewidth=1, linestyle=':')
            ax.text(i - 0.5, ax.get_ylim()[1] * 0.98, grp[:4], ha='center',
                    fontsize=6.5, color='gray', va='top')
            prev_grp = grp
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_delta_vs_dmos(df, available_models, fname):
    """Scatter: Improvement delta PSNR vs DMOS score."""
    fig, axes = plt.subplots(1, len(available_models), figsize=(5 * len(available_models), 4),
                             sharey=False)
    if len(available_models) == 1:
        axes = [axes]
    for ax, m in zip(axes, available_models):
        col = f'{m}_delta_psnr'
        if col not in df.columns:
            continue
        sub = df[[col, 'dmos', 'dist_group']].dropna()
        for grp in DISTORTION_GROUPS:
            s = sub[sub['dist_group'] == grp]
            ax.scatter(s['dmos'], s[col], alpha=0.25, s=6, label=grp)
        # Trend line
        x, y = sub['dmos'].values, sub[col].values
        if len(x) > 2:
            m_coef, b = np.polyfit(x, y, 1)
            ax.plot(sorted(x), [m_coef * xi + b for xi in sorted(x)],
                    color='black', linewidth=1.2, linestyle='--')
            r, p = pearsonr(x, y)
            ax.set_title(f'{m}\nr={r:.3f}, p={p:.3e}', fontsize=9, fontweight='bold')
        ax.axhline(0, color='red', linewidth=0.8, linestyle='-')
        ax.set_xlabel('DMOS (quality score, higher=better)', fontsize=8)
        ax.set_ylabel('PSNR Improvement vs Baseline (dB)', fontsize=8)
        ax.legend(fontsize=6, markerscale=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Improvement PSNR vs Kualitas Awal (DMOS)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_overall_comparison(overall, available_models, fname):
    """Bar chart: perbandingan overall PSNR, SSIM, LPIPS ketiga model."""
    metrics    = ['psnr', 'ssim', 'lpips']
    ylabels    = ['PSNR (dB)', 'SSIM', 'LPIPS']
    titles     = ['PSNR (higher=better)', 'SSIM (higher=better)', 'LPIPS (lower=better)']
    fig, axes  = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, ylabel, title in zip(axes, metrics, ylabels, titles):
        vals = [overall[m][metric] for m in available_models if m in overall]
        bars = ax.bar(available_models, vals,
                      color=[MODEL_COLORS[m] for m in available_models if m in overall],
                      edgecolor='white')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002 * bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('Perbandingan Overall Ketiga Model pada KADID-10k', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


def plot_level_overall(df, available_models, fname):
    """Line plot overall: metrik rata-rata vs distortion level, semua distorsi."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics   = ['psnr', 'ssim', 'lpips']
    ylabels   = ['PSNR (dB)', 'SSIM', 'LPIPS']
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for m in available_models:
            col = f'{m}_{metric}'
            if col not in df.columns:
                continue
            y_vals = [df[df['dist_level'] == lv][col].mean() for lv in range(1, 6)]
            ax.plot(range(1, 6), y_vals, color=MODEL_COLORS[m],
                    marker=MODEL_MARKERS[m], label=m, linewidth=2, markersize=6)
        bl_col = f'baseline_{metric}'
        if bl_col in df.columns:
            bl_vals = [df[df['dist_level'] == lv][bl_col].mean() for lv in range(1, 6)]
            ax.plot(range(1, 6), bl_vals, color='gray', linestyle='--',
                    linewidth=1.5, label='Baseline', alpha=0.8)
        ax.set_title(f'{ylabel} vs Level', fontsize=10, fontweight='bold')
        ax.set_xlabel('Distortion Level (1=ringan, 5=berat)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Performa vs Tingkat Keparahan Distorsi (Semua Tipe)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {fname}')


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3B — RINGKASAN INSIGHT TEKS
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(df, available_models, type_agg, level_agg, group_agg, overall):
    lines = []
    sep   = '=' * 70

    lines += [sep, '  EVALUASI KADID-10k — RINGKASAN INSIGHT', sep, '']
    lines += [f'  Jumlah gambar dievaluasi : {len(df):,}',
              f'  Model yang dievaluasi    : {", ".join(available_models)}',
              f'  Jumlah tipe distorsi     : {len(DISTORTION_NAMES)} (Gaussian Blur, Lens Blur, Motion Blur)', '']

    # ── Insight 1: Overall
    lines += [sep, '  1. PERBANDINGAN OVERALL', sep]
    lines.append(f'  {"Model":<15} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8} {"ΔPSNR":>8}')
    lines.append(f'  {"-"*15} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    ranked = sorted(available_models, key=lambda m: overall[m]['psnr'], reverse=True)
    for m in ranked:
        o = overall[m]
        lines.append(f'  {m:<15} {o["psnr"]:>8.4f} {o["ssim"]:>8.4f} '
                     f'{o["lpips"]:>8.4f} {o["delta_psnr"]:>+8.4f}')
    best_overall = ranked[0]
    lines += ['', f'  >> Model TERBAIK secara keseluruhan: {best_overall} '
              f'(PSNR={overall[best_overall]["psnr"]:.4f} dB)', '']

    # ── Insight 2: Per distortion group
    lines += [sep, '  2. PERFORMA PER KELOMPOK DISTORSI', sep]
    for grp in DISTORTION_GROUPS:
        if grp not in group_agg:
            continue
        vals = {m: group_agg[grp].get(f'{m}_psnr', np.nan) for m in available_models}
        valid = {m: v for m, v in vals.items() if not np.isnan(v)}
        if not valid:
            continue
        best_in_grp = max(valid, key=valid.get)
        lines.append(f'\n  [{grp}]')
        for m in available_models:
            v = vals.get(m, np.nan)
            ssim_v = group_agg[grp].get(f'{m}_ssim', np.nan)
            lpips_v = group_agg[grp].get(f'{m}_lpips', np.nan)
            delta = group_agg[grp].get(f'{m}_delta_psnr', np.nan)
            indicator = ' ★' if m == best_in_grp else ''
            lines.append(f'    {m:<15} PSNR={v:.4f}  SSIM={ssim_v:.4f}  LPIPS={lpips_v:.4f}  '
                         f'ΔPSNR={delta:+.4f}{indicator}')

    lines += ['']

    # ── Insight 3: Tipe distorsi mana tiap model unggul
    lines += [sep, '  3. TIPE DISTORSI TERKUAT PER MODEL', sep]
    for m in available_models:
        psnr_per_type = {t: type_agg[t].get(f'{m}_psnr', np.nan) for t in type_agg}
        valid = {t: v for t, v in psnr_per_type.items() if not np.isnan(v)}
        if not valid:
            continue
        sorted_types = sorted(valid, key=valid.get, reverse=True)
        top5  = sorted_types[:5]
        bot5  = sorted_types[-5:]
        lines.append(f'\n  [{m}]')
        lines.append('    5 TIPE TERKUAT:')
        for t in top5:
            ssim_v = type_agg[t].get(f'{m}_ssim', np.nan)
            lpips_v = type_agg[t].get(f'{m}_lpips', np.nan)
            lines.append(f'      - {DISTORTION_NAMES[t]:<22} PSNR={valid[t]:.4f}  '
                         f'SSIM={ssim_v:.4f}  LPIPS={lpips_v:.4f}')
        lines.append('    5 TIPE TERLEMAH:')
        for t in bot5:
            ssim_v = type_agg[t].get(f'{m}_ssim', np.nan)
            lpips_v = type_agg[t].get(f'{m}_lpips', np.nan)
            lines.append(f'      - {DISTORTION_NAMES[t]:<22} PSNR={valid[t]:.4f}  '
                         f'SSIM={ssim_v:.4f}  LPIPS={lpips_v:.4f}')

    lines += ['']

    # ── Insight 4: Model terbaik per tipe distorsi
    lines += [sep, '  4. MODEL TERBAIK PER TIPE DISTORSI', sep]
    win_count = {m: 0 for m in available_models}
    for dt in sorted(type_agg.keys()):
        vals = {m: type_agg[dt].get(f'{m}_psnr', np.nan) for m in available_models}
        valid = {m: v for m, v in vals.items() if not np.isnan(v)}
        if not valid:
            continue
        winner = max(valid, key=valid.get)
        win_count[winner] = win_count.get(winner, 0) + 1
        ssim_v = type_agg[dt].get(f'{winner}_ssim', np.nan)
        lpips_v = type_agg[dt].get(f'{winner}_lpips', np.nan)
        lines.append(f'  {DISTORTION_NAMES[dt]:<25} → {winner} '
                     f'(PSNR={valid[winner]:.4f}, SSIM={ssim_v:.4f}, LPIPS={lpips_v:.4f})')
    lines += ['', '  Jumlah kemenangan per model:']
    for m, cnt in sorted(win_count.items(), key=lambda x: -x[1]):
        lines.append(f'    {m:<15} : {cnt} / {len(type_agg)} tipe distorsi')
    lines += ['']

    # ── Insight 5: Degradasi vs severity
    lines += [sep, '  5. KETAHANAN TERHADAP KEPARAHAN DISTORSI', sep]
    lines.append('  Penurunan PSNR dari Level 1 ke Level 5:\n')
    for m in available_models:
        col = f'{m}_psnr'
        if col not in df.columns:
            continue
        psnr_l1 = df[df['dist_level'] == 1][col].mean()
        psnr_l5 = df[df['dist_level'] == 5][col].mean()
        drop = psnr_l1 - psnr_l5
        lines.append(f'  {m:<15} Level1={psnr_l1:.4f}  Level5={psnr_l5:.4f}  '
                     f'Drop={drop:.4f} dB')
    most_robust = min(
        [m for m in available_models if f'{m}_psnr' in df.columns],
        key=lambda m: (df[df['dist_level'] == 1][f'{m}_psnr'].mean()
                       - df[df['dist_level'] == 5][f'{m}_psnr'].mean())
    )
    lines += ['', f'  >> Model paling ROBUST terhadap distorsi berat: {most_robust}', '']

    # ── Insight 6: Korelasi PSNR dengan DMOS
    lines += [sep, '  6. KORELASI PERFORMA MODEL vs PERSEPSI MANUSIA (DMOS)', sep]
    lines.append('  Pearson r antara DMOS dan PSNR model:\n')
    for m in available_models:
        col = f'{m}_psnr'
        if col not in df.columns:
            continue
        sub = df[['dmos', col]].dropna()
        if len(sub) < 10:
            continue
        r_pear, p_pear = pearsonr(sub['dmos'], sub[col])
        r_spear, _     = spearmanr(sub['dmos'], sub[col])
        lines.append(f'  {m:<15} Pearson r={r_pear:.4f} (p={p_pear:.2e})  '
                     f'Spearman ρ={r_spear:.4f}')
    lines += ['']

    # ── Insight 7: Model yang aktif memperbaiki vs memperburuk
    lines += [sep, '  7. ANALISIS PERBAIKAN (ΔPSNR vs BASELINE)', sep]
    lines.append('  Persentase gambar yang membaik (ΔPSNR > 0):\n')
    for m in available_models:
        dcol = f'{m}_delta_psnr'
        if dcol not in df.columns:
            continue
        sub = df[dcol].dropna()
        improved   = (sub > 0).sum()
        worsened   = (sub <= 0).sum()
        pct        = improved / len(sub) * 100
        avg_impr   = sub[sub > 0].mean()
        avg_worse  = sub[sub <= 0].mean()
        lines.append(f'  {m:<15} membaik={improved:5d} ({pct:.1f}%)  '
                     f'memburuk={worsened:5d}  '
                     f'avg_gain={avg_impr:+.4f}  avg_loss={avg_worse:+.4f}')
    lines += ['']

    lines += [sep, '  END OF REPORT', sep]
    report = '\n'.join(lines)
    with open(INSIGHTS_TXT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    sep = '=' * 55

    # Fase 1: Komputasi metrik
    print(f'\n{sep}')
    print('  FASE 1 — KOMPUTASI METRIK')
    print(sep)
    if os.path.exists(METRICS_CSV):
        already = pd.read_csv(METRICS_CSV)
        if len(already) >= 10:  # ada data
            print(f'  Ditemukan metrics CSV ({len(already)} baris). Loading...')
            df = already
        else:
            df = compute_metrics()
    else:
        df = compute_metrics()

    if df is None or df.empty:
        print('[ERROR] Tidak ada data metrik. Hentikan.')
        exit(1)

    df['dist_type']  = df['dist_type'].astype(int)
    df['dist_level'] = df['dist_level'].astype(int)
    print(f'  Total baris: {len(df):,}')

    # Fase 2: Agregasi
    print(f'\n{sep}')
    print('  FASE 2 — AGREGASI')
    print(sep)
    available_models, type_agg, level_agg, group_agg, overall = aggregate(df)
    print(f'  Model: {available_models}')
    print(f'  Tipe distorsi terdapat: {len(type_agg)}')
    print(f'  Level tersedia: {sorted(level_agg.keys())}')

    # Fase 3: Visualisasi
    print(f'\n{sep}')
    print('  FASE 3 — VISUALISASI')
    print(sep)

    plot_overall_comparison(overall, available_models, 'overall_comparison.png')
    plot_heatmap(type_agg, available_models, 'psnr',
                 'PSNR per Model × Distortion Type', 'Blues', 'heatmap_psnr.png')
    plot_heatmap(type_agg, available_models, 'ssim',
                 'SSIM per Model × Distortion Type', 'Greens', 'heatmap_ssim.png')
    plot_heatmap(type_agg, available_models, 'lpips',
                 'LPIPS per Model × Distortion Type (lower=better)', 'Reds_r', 'heatmap_lpips.png')
    plot_heatmap(type_agg, available_models, 'delta_psnr',
                 'ΔPSNR vs Baseline (+ = membaik)', 'RdYlGn', 'heatmap_delta_psnr.png')
    plot_performance_vs_level(df, available_models, 'psnr', 'PSNR (dB)', 'level_psnr.png')
    plot_performance_vs_level(df, available_models, 'ssim', 'SSIM', 'level_ssim.png')
    plot_level_overall(df, available_models, 'level_overall.png')
    plot_best_model_per_type(type_agg, available_models, 'psnr', 'best_model_psnr.png')
    plot_delta_vs_dmos(df, available_models, 'delta_vs_dmos.png')

    # Fase 3B: Insight teks
    print(f'\n{sep}')
    print('  FASE 3B — INSIGHT SUMMARY')
    print(sep)
    generate_insights(df, available_models, type_agg, level_agg, group_agg, overall)

    # Simpan agregasi JSON untuk referensi
    agg_out = {
        'overall': {m: {k: float(v) for k, v in overall[m].items()} for m in available_models},
        'by_group': {
            g: {k: float(v) if isinstance(v, float) else v
                for k, v in d.items()} for g, d in group_agg.items()
        },
    }
    with open(os.path.join(OUT_DIR, 'aggregated_results.json'), 'w') as f:
        json.dump(agg_out, f, indent=2)

    print(f'\n{sep}')
    print('  SELESAI!')
    print(f'  Plots   : {PLOTS_DIR}')
    print(f'  Metrics : {METRICS_CSV}')
    print(f'  Insights: {INSIGHTS_TXT}')
    print(sep)
