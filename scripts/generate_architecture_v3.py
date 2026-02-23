#!/usr/bin/env python3
"""Generate compact architecture pipeline diagram (architecture_v3.pdf).

Flat, wide layout: 4 phases as squat horizontal boxes with internal components
arranged horizontally instead of vertically. Minimal height, maximum width usage.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Colour palette ──
C_PHASE1 = '#E3F2FD'; C_BORDER1 = '#1565C0'
C_PHASE2 = '#E8F5E9'; C_BORDER2 = '#2E7D32'
C_PHASE3 = '#FFF3E0'; C_BORDER3 = '#E65100'
C_PHASE4 = '#F3E5F5'; C_BORDER4 = '#6A1B9A'
C_ARROW  = '#37474F'; C_TEXT = '#212121'; C_NUM = '#FFFFFF'

# ── Canvas: wide and short ──
fig, ax = plt.subplots(1, 1, figsize=(17, 3.4), dpi=300)
ax.set_xlim(-0.5, 17.0)
ax.set_ylim(-0.05, 3.35)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helpers ──
def phase_box(x, y, w, h, color, border, label, num, ax):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=border, linewidth=1.8, zorder=1)
    ax.add_patch(box)
    # Header bar
    hh = 0.38
    header = FancyBboxPatch((x+0.03, y+h-hh-0.02), w-0.06, hh,
                            boxstyle="round,pad=0.05", facecolor=border,
                            edgecolor='none', alpha=0.85, zorder=2)
    ax.add_patch(header)
    num_text = '\u2460\u2461\u2462\u2463'[int(num)-1]
    ax.text(x+0.26, y+h-hh/2-0.02, num_text, ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_NUM, zorder=4)
    ax.text(x+0.48, y+h-hh/2-0.02, label, ha='left', va='center',
            fontsize=7.2, fontweight='bold', color=C_NUM, zorder=4,
            fontfamily='sans-serif')

def inner_box(x, y, w, h, text, ax, fontsize=6.2, color='#FAFAFA',
              border_color='#B0BEC5', bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                         facecolor=color, edgecolor=border_color, linewidth=0.8, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x+w/2, y+h/2, text, ha='center', va='center',
            fontsize=fontsize, color=C_TEXT, zorder=4,
            fontfamily='sans-serif', fontweight=weight, linespacing=1.15)

def draw_arrow(x1, y1, x2, y2, ax, lw=1.8, color=C_ARROW, shrinkA=2, shrinkB=2,
               head_width=0.3, head_length=0.15):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'-|>,head_width={head_width},head_length={head_length}',
                                color=color, lw=lw, connectionstyle='arc3,rad=0',
                                shrinkA=shrinkA, shrinkB=shrinkB), zorder=6)

def horiz_arrow(x_l, x_r, y, ax, lw=0.8, color='#999'):
    draw_arrow(x_l, y, x_r, y, ax, lw=lw, color=color, shrinkA=1, shrinkB=1,
               head_width=0.18, head_length=0.10)

# ── Layout ──
Y_BASE = 0.05
PHASE_H = 2.85
GAP = 0.28

PW1 = 3.40; PW2 = 2.55; PW3 = 5.60; PW4 = 3.30
X1 = 0.50
X2 = X1 + PW1 + GAP
X3 = X2 + PW2 + GAP
X4 = X3 + PW3 + GAP

# ── Phase boxes ──
phase_box(X1, Y_BASE, PW1, PHASE_H, C_PHASE1, C_BORDER1, 'Deep Interaction Crawler', '1', ax)
phase_box(X2, Y_BASE, PW2, PHASE_H, C_PHASE2, C_BORDER2, 'EITG Builder', '2', ax)
phase_box(X3, Y_BASE, PW3, PHASE_H, C_PHASE3, C_BORDER3, 'Multi-View Feature Extraction (125D)', '3', ax)
phase_box(X4, Y_BASE, PW4, PHASE_H, C_PHASE4, C_BORDER4, 'SMVE Classifier', '4', ax)

# ══ Phase 1: Crawler — 2-column layout ══
iw1 = 1.42; ih1 = 0.44; igap_h = 0.14; igap_v = 0.12
ix1_l = X1 + 0.18; ix1_r = ix1_l + iw1 + igap_h
y1_top = Y_BASE + PHASE_H - 0.55

comp1 = [
    ('Headless\nChromium', 'DOM Scanner\n& Discovery'),
    ('Form Filler\n(Credentials)', 'Recursive\nDepth Crawl'),
]
for row, (left, right) in enumerate(comp1):
    yy = y1_top - row*(ih1 + igap_v)
    inner_box(ix1_l, yy, iw1, ih1, left, ax, fontsize=5.8, color='#DCEEFB', border_color='#5C9BD4')
    inner_box(ix1_r, yy, iw1, ih1, right, ax, fontsize=5.8, color='#DCEEFB', border_color='#5C9BD4')
    if row == 0:
        horiz_arrow(ix1_l+iw1, ix1_r, yy+ih1/2, ax, color=C_BORDER1)

# Vertical arrows between rows
for col_x in [ix1_l + iw1/2, ix1_r + iw1/2]:
    draw_arrow(col_x, y1_top - ih1, col_x, y1_top - ih1 - igap_v + ih1, ax,
               lw=0.7, color=C_BORDER1, shrinkA=0, shrinkB=0, head_width=0.15, head_length=0.08)

# Output label
ih1_out = 0.35
y1_out = y1_top - 2*(ih1 + igap_v) + 0.02
inner_box(ix1_l, y1_out, PW1-0.36, ih1_out, 'Post-Submit Analysis', ax,
          fontsize=5.8, color='#DCEEFB', border_color='#5C9BD4')

ax.text(X1+PW1/2, Y_BASE+0.18, 'Interaction Trace', ha='center', va='center',
        fontsize=6.0, fontweight='bold', color=C_BORDER1, fontstyle='italic',
        fontfamily='sans-serif')

# ══ Phase 2: EITG — 3 vertically stacked boxes ══
iw2 = PW2 - 0.40; ih2 = 0.50; igap2 = 0.12
ix2 = X2 + (PW2 - iw2)/2
y2_top = Y_BASE + PHASE_H - 0.55

comp2 = ['Event Stream\nParser', 'Graph Build\nG = (V, E)', 'DRP Pruning\n(Thm 1)']
for i, label in enumerate(comp2):
    yy = y2_top - i*(ih2 + igap2)
    inner_box(ix2, yy, iw2, ih2, label, ax, fontsize=5.8, color='#D4EDDA', border_color='#5CB85C')
    if i > 0:
        draw_arrow(ix2+iw2/2, yy+ih2+igap2, ix2+iw2/2, yy+ih2, ax,
                   lw=0.7, color=C_BORDER2, shrinkA=0, shrinkB=0, head_width=0.15, head_length=0.08)

ax.text(X2+PW2/2, Y_BASE+0.18, 'EITG G\' = (V, E)', ha='center', va='center',
        fontsize=6.0, fontweight='bold', color=C_BORDER2, fontstyle='italic',
        fontfamily='sans-serif')

# ══ Phase 3: Multi-View — 6 views in 2×3 grid ══
col_w = 1.65; col_gap = 0.18; ih3 = 0.50; igap3 = 0.12
ix3_l = X3 + (PW3 - 2*col_w - col_gap)/2
ix3_r = ix3_l + col_w + col_gap
y3_top = Y_BASE + PHASE_H - 0.55

views_left  = [('URL Structure (21D)', '#FFF0DB', '#E68A00'),
               ('Network Traffic (18D)', '#FFF0DB', '#E68A00'),
               ('Redirect Chain (14D)', '#FFF0DB', '#E68A00')]
views_right = [('Interaction Events (18D)', '#FFF0DB', '#E68A00'),
               ('ITG + Engineered (42D)', '#FFF0DB', '#E68A00'),
               ('Cross-View (12D)', '#FFF0DB', '#E68A00')]

for i, (name, c, bc) in enumerate(views_left):
    yy = y3_top - i*(ih3 + igap3)
    inner_box(ix3_l, yy, col_w, ih3, name, ax, fontsize=5.6, color=c, border_color=bc)

for i, (name, c, bc) in enumerate(views_right):
    yy = y3_top - i*(ih3 + igap3)
    inner_box(ix3_r, yy, col_w, ih3, name, ax, fontsize=5.6, color=c, border_color=bc)

ax.text(X3+PW3/2, Y_BASE+0.18, '125-D Feature Vector → 6 Views',
        ha='center', va='center', fontsize=6.0, fontweight='bold',
        color=C_BORDER3, fontstyle='italic', fontfamily='sans-serif')

# ══ Phase 4: SMVE — horizontal stack ══
iw4 = PW4 - 0.40; ix4 = X4 + (PW4-iw4)/2
ih4 = 0.55; igap4 = 0.14
y4_top = Y_BASE + PHASE_H - 0.55

# Level-0
inner_box(ix4, y4_top, iw4, ih4,
          'Level-0: Per-View RF/GBM\n(6 base classifiers)', ax,
          fontsize=5.8, color='#E8D5F5', border_color='#8E44AD')

# OOF posteriors
y4_p = y4_top - ih4 - igap4
inner_box(ix4, y4_p, iw4, ih4*0.75,
          'OOF Posteriors\n(6 probability vectors)', ax,
          fontsize=5.5, color='#F3E8FF', border_color='#7C3AED')
draw_arrow(ix4+iw4/2, y4_top, ix4+iw4/2, y4_p+ih4*0.75, ax,
           lw=0.7, color=C_BORDER4, shrinkA=0, shrinkB=0, head_width=0.15, head_length=0.08)

# Level-1
y4_l1 = y4_p - ih4*0.75 - igap4
inner_box(ix4, y4_l1, iw4, ih4,
          'Level-1 Meta-Learner\nLR + GBM soft vote', ax,
          fontsize=5.8, color='#E8D5F5', border_color='#8E44AD')
draw_arrow(ix4+iw4/2, y4_p, ix4+iw4/2, y4_l1+ih4, ax,
           lw=0.7, color=C_BORDER4, shrinkA=0, shrinkB=0, head_width=0.15, head_length=0.08)

# Decision
y4_d = y4_l1 - ih4*0.65 - igap4*0.7
out_w = (iw4 - 0.18)/2
inner_box(ix4, y4_d, out_w, ih4*0.65, 'Phishing', ax,
          fontsize=6.0, color='#FFCDD2', border_color='#C62828', bold=True)
inner_box(ix4+out_w+0.18, y4_d, out_w, ih4*0.65, 'Benign', ax,
          fontsize=6.0, color='#C8E6C9', border_color='#2E7D32', bold=True)
draw_arrow(ix4+iw4/2-0.25, y4_l1, ix4+out_w/2, y4_d+ih4*0.65, ax,
           lw=0.7, color=C_BORDER4, head_width=0.15, head_length=0.08)
draw_arrow(ix4+iw4/2+0.25, y4_l1, ix4+out_w+0.18+out_w/2, y4_d+ih4*0.65, ax,
           lw=0.7, color=C_BORDER4, head_width=0.15, head_length=0.08)

# ── Input arrow: "Target URL" → Phase 1 ──
mid_y = Y_BASE + PHASE_H/2
url_bx = X1 - 0.42
inner_box(url_bx-0.18, mid_y-0.22, 0.36, 0.44, 'Target\nURL', ax,
          fontsize=5.8, color='#ECEFF1', border_color='#607D8B', bold=True)
draw_arrow(url_bx+0.20, mid_y, X1, mid_y, ax, lw=2.0, color=C_ARROW)

# ── Inter-phase arrows ──
draw_arrow(X1+PW1, mid_y, X2, mid_y, ax, lw=2.5, color=C_ARROW,
           shrinkA=1, shrinkB=1, head_width=0.35, head_length=0.18)
ax.text((X1+PW1+X2)/2, mid_y+0.20, 'trace', ha='center', va='center',
        fontsize=5.5, color='#333', fontstyle='italic', fontfamily='sans-serif', fontweight='bold')

draw_arrow(X2+PW2, mid_y, X3, mid_y, ax, lw=2.5, color=C_ARROW,
           shrinkA=1, shrinkB=1, head_width=0.35, head_length=0.18)
ax.text((X2+PW2+X3)/2, mid_y+0.20, 'graph', ha='center', va='center',
        fontsize=5.5, color='#333', fontstyle='italic', fontfamily='sans-serif', fontweight='bold')

draw_arrow(X3+PW3, mid_y, X4, mid_y, ax, lw=2.5, color=C_ARROW,
           shrinkA=1, shrinkB=1, head_width=0.35, head_length=0.18)
ax.text((X3+PW3+X4)/2, mid_y+0.20, '125-D', ha='center', va='center',
        fontsize=5.5, color='#333', fontstyle='italic', fontfamily='sans-serif', fontweight='bold')

# Bypass dashed arrow: raw trace Phase 1 → Phase 3
ax.annotate('', xy=(X3+0.3, Y_BASE+PHASE_H+0.01),
            xytext=(X1+PW1-0.3, Y_BASE+PHASE_H+0.01),
            arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.12',
                            color='#90A4AE', lw=1.0, connectionstyle='arc3,rad=-0.18',
                            linestyle='dashed', shrinkA=2, shrinkB=2), zorder=5)
ax.text((X1+PW1+X3)/2, Y_BASE+PHASE_H+0.25,
        'raw trace (URL, network, redirect, interaction views)',
        ha='center', va='center', fontsize=4.8, color='#78909C',
        fontstyle='italic', fontfamily='sans-serif')

# ── Save ──
outdir = Path(__file__).resolve().parent.parent / 'iccs2026latex' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)

pdf_path = outdir / 'architecture_v3.pdf'
fig.savefig(str(pdf_path), format='pdf', bbox_inches='tight', pad_inches=0.06)
plt.close(fig)
print(f"Saved: {pdf_path}")
