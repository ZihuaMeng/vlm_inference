#!/usr/bin/env python3
"""Generate presentation plots from experimental data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'plots')
os.makedirs(OUT, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
QWEN_COLOR = '#2563eb'
KONTEXT_COLOR = '#d97706'
ROBUST_COLOR = '#16a34a'
FRAGILE_COLOR = '#dc2626'


# ═══════════════════════════════════════════════
# Plot 1: Cross-Model Speedup Comparison
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

configs = [
    'CFG-skip\nonly',
    'Temporal\ncache only',
    'Combined\n(best)',
]
qwen_speedups = [1.60, 1.67, 2.35]
kontext_speedups = [None, 2.15, None]  # Kontext has no CFG-skip or combined

x = np.arange(len(configs))
w = 0.32

bars_q = ax.bar(x - w/2, qwen_speedups, w, label='Qwen-Image-Edit-2511', color=QWEN_COLOR, alpha=0.85)
kontext_vals = [0 if v is None else v for v in kontext_speedups]
kontext_mask = [v is not None for v in kontext_speedups]
bars_k = ax.bar(x + w/2, kontext_vals, w, label='FLUX.1 Kontext-dev', color=KONTEXT_COLOR, alpha=0.85)

# Gray out N/A bars
for i, avail in enumerate(kontext_mask):
    if not avail:
        bars_k[i].set_color('#e5e7eb')
        bars_k[i].set_edgecolor('#d1d5db')
        bars_k[i].set_height(0)

for bar in bars_q:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{bar.get_height():.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
for i, bar in enumerate(bars_k):
    if kontext_mask[i]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f'{bar.get_height():.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.08,
                'N/A', ha='center', va='bottom', fontsize=9, color='#9ca3af', style='italic')

ax.set_ylabel('Speedup (x)')
ax.set_title('Cross-Model Speedup Comparison')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0, 2.8)
ax.legend(loc='upper left')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='_nolegend_')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cross_model_speedup.png'), dpi=150)
plt.close(fig)
print('✓ cross_model_speedup.png')


# ═══════════════════════════════════════════════
# Plot 2: Quality Distribution Comparison
# ═══════════════════════════════════════════════
# Qwen M9 combined SSIM values (18 cases)
qwen_ssim = [0.955, 0.949, 0.947, 0.931, 0.926, 0.911, 0.905, 0.895, 0.890,
             0.864, 0.853, 0.832, 0.799, 0.766, 0.687, 0.665, 0.642, 0.637]

# Kontext F2 conservative SSIM values (12 cases)
kontext_ssim = [0.987, 0.934, 0.906, 0.909, 0.874, 0.834, 0.741, 0.710,
                0.686, 0.681, 0.415, 0.366]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

bins = np.arange(0.3, 1.05, 0.05)

ax1.hist(qwen_ssim, bins=bins, color=QWEN_COLOR, alpha=0.75, edgecolor='white', linewidth=0.8)
ax1.axvline(np.mean(qwen_ssim), color=QWEN_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(qwen_ssim):.3f}')
ax1.axvline(np.median(qwen_ssim), color=QWEN_COLOR, linestyle=':', linewidth=2, label=f'Median: {np.median(qwen_ssim):.3f}')
ax1.set_xlabel('SSIM vs. Baseline')
ax1.set_ylabel('Number of Test Cases')
ax1.set_title(f'Qwen Combined (2.35x, n={len(qwen_ssim)})')
ax1.legend(fontsize=9)
ax1.set_xlim(0.3, 1.02)

ax2.hist(kontext_ssim, bins=bins, color=KONTEXT_COLOR, alpha=0.75, edgecolor='white', linewidth=0.8)
ax2.axvline(np.mean(kontext_ssim), color=KONTEXT_COLOR, linestyle='--', linewidth=2, label=f'Mean: {np.mean(kontext_ssim):.3f}')
ax2.axvline(np.median(kontext_ssim), color=KONTEXT_COLOR, linestyle=':', linewidth=2, label=f'Median: {np.median(kontext_ssim):.3f}')
ax2.set_xlabel('SSIM vs. Baseline')
ax2.set_title(f'Kontext Conservative (2.15x, n={len(kontext_ssim)})')
ax2.legend(fontsize=9)
ax2.set_xlim(0.3, 1.02)

fig.suptitle('Quality Distribution Under Acceleration (Both Models)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'quality_distribution_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('✓ quality_distribution_comparison.png')


# ═══════════════════════════════════════════════
# Plot 3: Content-Type Quality Comparison (matched types)
# ═══════════════════════════════════════════════
matched_types = [
    'simple_object', 'arrangement', 'portrait', 'interior',
    'nature', 'identity', 'style_transfer', 'texture', 'complex_scene'
]
qwen_matched = [0.926, 0.955, 0.911, 0.890, 0.687, 0.931, 0.642, 0.665, 0.637]
kontext_matched = [0.987, 0.906, 0.909, 0.834, 0.710, 0.686, 0.681, 0.366, 0.415]

# Sort by Qwen SSIM descending
order = np.argsort(qwen_matched)[::-1]
matched_types = [matched_types[i] for i in order]
qwen_matched = [qwen_matched[i] for i in order]
kontext_matched = [kontext_matched[i] for i in order]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(matched_types))
w = 0.35

bars_q = ax.bar(x - w/2, qwen_matched, w, label='Qwen (2.35x combined)', color=QWEN_COLOR, alpha=0.85)
bars_k = ax.bar(x + w/2, kontext_matched, w, label='Kontext (2.15x temporal)', color=KONTEXT_COLOR, alpha=0.85)

ax.axhline(y=0.85, color=ROBUST_COLOR, linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(len(matched_types)-0.5, 0.86, 'SSIM 0.85 threshold', ha='right', fontsize=8, color=ROBUST_COLOR)

ax.set_ylabel('SSIM vs. Baseline')
ax.set_title('Quality by Content Type (Cross-Model Comparison)')
ax.set_xticks(x)
ax.set_xticklabels([t.replace('_', '\n') for t in matched_types], fontsize=9)
ax.set_ylim(0.2, 1.05)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'content_type_comparison.png'), dpi=150)
plt.close(fig)
print('✓ content_type_comparison.png')


# ═══════════════════════════════════════════════
# Plot 4: F4 Ablation — KV Reuse vs Temporal Caching
# ═══════════════════════════════════════════════
cases = ['simple\nobject', 'object\nmodify', 'portrait', 'arrange-\nment',
         'style\ntransfer', 'fine\ntexture', 'complex\nscene', 'nature\ndetail']
kv_ssim = [0.990, 0.931, 0.947, 0.955, 0.902, 0.927, 0.732, 0.952]
tc_ssim = [0.987, 0.888, 0.836, 0.885, 0.812, 0.807, 0.442, 0.778]
kv_speed = [1.016]*8
tc_speed = [2.154]*8

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1.5]})

x = np.arange(len(cases))
w = 0.35

# Quality subplot
bars1 = ax1.bar(x - w/2, kv_ssim, w, label='KV reuse (condition K/V cached)', color='#7c3aed', alpha=0.85)
bars2 = ax1.bar(x + w/2, tc_ssim, w, label='Temporal cache (whole step cached)', color='#0891b2', alpha=0.85)

ax1.axhline(y=0.85, color=ROBUST_COLOR, linestyle='--', alpha=0.5, linewidth=1.2)
ax1.set_ylabel('SSIM vs. Baseline')
ax1.set_title('F4 Mechanistic Ablation: Where Does Quality Loss Come From?')
ax1.set_xticks(x)
ax1.set_xticklabels(cases, fontsize=9)
ax1.set_ylim(0.3, 1.05)
ax1.legend(loc='lower left', fontsize=9)

# Annotate the gap on the most dramatic case (complex_scene)
idx = 6  # complex_scene
ax1.annotate('', xy=(idx + w/2, tc_ssim[idx]), xytext=(idx - w/2, kv_ssim[idx]),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax1.text(idx, (kv_ssim[idx] + tc_ssim[idx])/2, f'+{kv_ssim[idx]-tc_ssim[idx]:.2f}',
         ha='center', va='center', fontsize=9, color='red', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))

# Speedup subplot
ax2.bar(x - w/2, kv_speed, w, color='#7c3aed', alpha=0.85)
ax2.bar(x + w/2, tc_speed, w, color='#0891b2', alpha=0.85)
ax2.set_ylabel('Speedup (x)')
ax2.set_xticks(x)
ax2.set_xticklabels(cases, fontsize=9)
ax2.set_ylim(0, 2.5)
ax2.text(3.5, 2.3, 'Temporal cache: 2.15x', ha='center', fontsize=10, color='#0891b2', fontweight='bold')
ax2.text(3.5, 1.15, 'KV reuse: 1.015x', ha='center', fontsize=10, color='#7c3aed', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'f4_mechanistic_ablation.png'), dpi=150)
plt.close(fig)
print('✓ f4_mechanistic_ablation.png')


# ═══════════════════════════════════════════════
# Plot 5: Temporal Plateau Comparison (normalized trajectory)
# ═══════════════════════════════════════════════
trajectory_frac = [0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00]
qwen_cos =       [0.981, 0.995, 0.999, 0.9996, 0.9999, 0.9999, 0.9999, 0.9998, 0.9995, 0.9991, 0.998, 0.816]
kontext_cos =    [0.957, 0.995, 0.999, 0.9997, 0.9999, 0.9999, 0.9997, 0.9995, 0.993,  0.982,  None,  0.981]

# Filter None
k_frac = [f for f, c in zip(trajectory_frac, kontext_cos) if c is not None]
k_cos = [c for c in kontext_cos if c is not None]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(trajectory_frac, qwen_cos, 'o-', color=QWEN_COLOR, linewidth=2, markersize=6,
        label='Qwen (40 steps, true CFG)')
ax.plot(k_frac, k_cos, 's-', color=KONTEXT_COLOR, linewidth=2, markersize=6,
        label='Kontext (28 steps, guidance-distilled)')

# Shade plateau region
ax.axvspan(0.10, 0.75, alpha=0.08, color='green', label='Plateau region (cos > 0.999)')

# Annotate divergence
ax.annotate('Qwen: dramatic\ndivergence (0.816)',
            xy=(1.0, 0.816), xytext=(0.82, 0.87),
            arrowprops=dict(arrowstyle='->', color=QWEN_COLOR),
            fontsize=9, color=QWEN_COLOR)
ax.annotate('Kontext: mild\ndivergence (0.981)',
            xy=(1.0, 0.981), xytext=(0.82, 0.955),
            arrowprops=dict(arrowstyle='->', color=KONTEXT_COLOR),
            fontsize=9, color=KONTEXT_COLOR)

ax.set_xlabel('Normalized Trajectory Fraction (0 = start, 1 = end)')
ax.set_ylabel('Cosine Similarity (step t vs step t-1)')
ax.set_title('Temporal Redundancy Profile: Three-Phase Pattern on Both Models')
ax.set_ylim(0.80, 1.002)
ax.legend(loc='lower left', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'temporal_plateau_comparison.png'), dpi=150)
plt.close(fig)
print('✓ temporal_plateau_comparison.png')


# ═══════════════════════════════════════════════
# Plot 6: Quality by Difficulty (both models)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

diffs = ['Easy', 'Medium', 'Hard']
qwen_diff = [0.943, 0.886, 0.753]
kontext_diff = [0.892, 0.826, 0.543]

x = np.arange(len(diffs))
w = 0.3

ax.bar(x - w/2, qwen_diff, w, label='Qwen (2.35x)', color=QWEN_COLOR, alpha=0.85)
ax.bar(x + w/2, kontext_diff, w, label='Kontext (2.15x)', color=KONTEXT_COLOR, alpha=0.85)

for i in range(len(diffs)):
    ax.text(x[i] - w/2, qwen_diff[i] + 0.015, f'{qwen_diff[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(x[i] + w/2, kontext_diff[i] + 0.015, f'{kontext_diff[i]:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=0.85, color=ROBUST_COLOR, linestyle='--', alpha=0.5, linewidth=1.2)
ax.set_ylabel('Mean SSIM')
ax.set_title('Quality by Difficulty Level')
ax.set_xticks(x)
ax.set_xticklabels(diffs)
ax.set_ylim(0.4, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'quality_by_difficulty.png'), dpi=150)
plt.close(fig)
print('✓ quality_by_difficulty.png')

print('\nAll plots saved to', os.path.abspath(OUT))
