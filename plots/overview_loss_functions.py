import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse
from scipy.interpolate import make_interp_spline
import seaborn as sns

from plots.settings import BASELINE_COLORS, set_size, CORL_SETTINGS_FULL_WIDTH, FIGURES_DIR, FIG_FORMAT
plt.rcParams.update(CORL_SETTINGS_FULL_WIDTH)

TW = np.array([[20, 47], [55, 93], [100, 69], [145, 109], [185, 87]])
OFFSETS = np.array([0, -6, 8, -5, 10])
PRED_MEANS = TW.copy().astype(float)
PRED_MEANS[:, 1] += OFFSETS

S_ELBO_COLOR = BASELINE_COLORS["s-NLL"]
M_ELBO_COLOR = BASELINE_COLORS["m-NLL"]
M_KDE_COLOR = BASELINE_COLORS["m-KDE"]
M_FCRPS_COLOR = BASELINE_COLORS["m-CRPS"]
M_ECRPS_COLOR = BASELINE_COLORS["m-arCRPS"]

TEAL = S_ELBO_COLOR
BLUE = M_ELBO_COLOR
ORANGE = M_FCRPS_COLOR
PURPLE = M_ECRPS_COLOR
sample_line_alpha = 0.3

# Shared cone endpoint styling (used across panels for consistency)
CONE_BODY_ALPHA = 0.10
CONE_BODY_LW = 0.6
CONE_END_ALPHA = 0.20
CONE_END_LW = 1.0
CONE_CROSS_LW = 1.0
CONE_CROSS_ALPHA = 0.8


def smooth_pts(pts):
    t = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, 100)
    return (make_interp_spline(t, pts[:, 0], k=3)(t_new),
            make_interp_spline(t, pts[:, 1], k=3)(t_new))


def draw_pred_cone(ax, start, end, spread, col):
    dx = end[0] - start[0]
    pinch, mid_x = spread * 0.12, start[0] + dx * 0.55
    verts = [
        (start[0], start[1] - pinch),
        (mid_x, start[1] - pinch), (mid_x, end[1] - spread), (end[0], end[1] - spread),
        (end[0] + 8, end[1] - spread), (end[0] + 8, end[1] + spread), (end[0], end[1] + spread),
        (mid_x, end[1] + spread), (mid_x, start[1] + pinch), (start[0], start[1] + pinch),
        (start[0], start[1] - pinch)
    ]
    codes = [Path.MOVETO] + [Path.CURVE4]*9 + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes),
                           facecolor=col, alpha=CONE_BODY_ALPHA, edgecolor=col, lw=CONE_BODY_LW, zorder=3))
    ax.add_patch(Ellipse((end[0], end[1]), 12, spread*2,
                          facecolor=col, alpha=CONE_END_ALPHA, edgecolor=col, lw=CONE_END_LW, zorder=3))
    ax.plot([end[0]-3, end[0]+3], [end[1], end[1]], color=col, lw=CONE_CROSS_LW, alpha=CONE_CROSS_ALPHA, zorder=3)
    ax.plot([end[0], end[0]], [end[1]-3, end[1]+3], color=col, lw=CONE_CROSS_LW, alpha=CONE_CROSS_ALPHA, zorder=3)


def draw_penalty_inset(ax_parent, fig, is_crps=False):
    bbox = ax_parent.get_position()
    inset_w = bbox.width * 0.28
    inset_h = bbox.height * 0.32
    inset_x = bbox.x0 + bbox.width * 0.68
    inset_y = bbox.y0 + bbox.height * 0.07
    ax_in = fig.add_axes([inset_x, inset_y, inset_w, inset_h])

    e = np.linspace(-2.5, 2.5, 100)
    if is_crps:
        penalty = np.abs(e)
        col = M_FCRPS_COLOR
        label = r'$|e|$'
        ylim_top = 2.8
    else:
        penalty = e**2
        col = BLUE
        label = r'$e^2$'
        ylim_top = 4.5

    ax_in.fill_between(e, 0, penalty, facecolor=col, alpha=0.12)
    ax_in.plot(e, penalty, color=col, lw=1.2)
    ax_in.set_xlim(-2.5, 2.5)
    ax_in.set_ylim(-0.15, ylim_top)
    ax_in.axhline(0, color='gray', lw=0.4)
    ax_in.axvline(0, color='gray', lw=0.4)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_xlabel(r'$\delta - \hat{\delta}$', fontsize=5, labelpad=1)
    ax_in.set_ylabel(label, fontsize=5, labelpad=1, rotation=0, ha='right')
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('gray')
    ax_in.patch.set_alpha(0.85)
    ax_in.patch.set_facecolor('white')
    return ax_in


# --- Main Plot ---
x_size, y_size = set_size(397)
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(x_size, 0.4 * y_size))

subs   = [r"\texttt{s-NLL}", r"\texttt{m-NLL}", r"\texttt{m-CRPS}", r"\texttt{m-arCRPS}"]

CONE_SPREAD = 16

for i, ax in enumerate(axes):
    ax.set_xlim(15, 205)
    ax.set_ylim(25, 150)
    ax.set_aspect('equal')

    xticks = TW[:, 0]
    yticks = [25, 50, 75, 100, 125, 150]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(r'Time $\mathcal{T}\quad \rightarrow$')
    ax.tick_params(left=False)

    for x in xticks:
        ax.vlines(x, min(yticks), max(yticks),
                  color='gray', linestyle='-', alpha=0.2, lw=0.8, zorder=0)
    for y in yticks:
        ax.hlines(y, min(xticks), max(xticks),
                  color='gray', linestyle='-', alpha=0.2, lw=0.8, zorder=0)

    gx, gy = smooth_pts(TW)

    # ================================================================
    # PANEL 0 — SS-ELBO (H=1)
    # ================================================================
    if i == 0:
        idx_first = np.where((gx >= TW[0, 0]) & (gx <= TW[1, 0]))[0]
        idx_rest  = np.where(gx > TW[1, 0])[0]
        ax.plot(gx[idx_first], gy[idx_first],
                color='k', lw=1.2, ls='--', alpha=0.7, zorder=2)
        ax.plot(gx[idx_rest], gy[idx_rest],
                color='tab:gray', lw=1.2, ls='--', alpha=0.5, zorder=2)
        ax.scatter(TW[:, 0], TW[:, 1], c='tab:gray', s=9, zorder=5)

        draw_pred_cone(ax, TW[0], PRED_MEANS[1], 20, TEAL)
        ax.scatter(TW[:2, 0], TW[:2, 1], c='black', s=9, zorder=5)
        ax.scatter(TW[0, 0], TW[0, 1], c='black', s=12)

    # ================================================================
    # PANEL 1 — MS-ELBO: teacher-forced, two W samples, constant cones
    # ================================================================
    elif i == 1:
        ax.plot(gx, gy, color='k', lw=1.2, ls='--', alpha=0.7, zorder=2)

        rng = np.random.default_rng(42)
        W_COLORS = [M_ELBO_COLOR]
        for w_idx in range(len(W_COLORS)):
            col_w = W_COLORS[w_idx]
            w_preds = PRED_MEANS.copy()
            base_shift = 2 * (1 if w_idx == 0 else -1.2)
            for h in range(1, len(TW)):
                step_var = (rng.random() - 0.5) * 8 * h
                w_preds[h, 1] += base_shift * (0.8 + 0.2*h) + step_var

            for h in range(len(TW) - 1):
                draw_pred_cone(ax, TW[h], w_preds[h+1], CONE_SPREAD, col_w)

        ax.scatter(TW[:, 0], TW[:, 1], c='black', s=9, zorder=5)
        ax.scatter(TW[0, 0], TW[0, 1], c='black', s=12)

    # ================================================================
    # PANEL 2 — MS-fCRPS: teacher-forced, two W samples, constant cones
    # ================================================================
    elif i == 2:
        ax.plot(gx, gy, color='k', lw=1.2, ls='--', alpha=0.7, zorder=2)

        rng = np.random.default_rng(42)
        W_COLORS = [M_FCRPS_COLOR, "tab:brown"]
        for w_idx in range(2):
            col_w = W_COLORS[w_idx]
            w_preds = PRED_MEANS.copy()
            base_shift = 6 * (1 if w_idx == 0 else -1.2)
            for h in range(1, len(TW)):
                step_var = (rng.random() - 0.5) * 8 * h
                w_preds[h, 1] += base_shift * (0.8 + 0.2*h) + step_var

            for h in range(len(TW) - 1):
                draw_pred_cone(ax, TW[h], w_preds[h+1], CONE_SPREAD, col_w)

        ax.scatter(TW[:, 0], TW[:, 1], c='black', s=9, zorder=5)
        ax.scatter(TW[0, 0], TW[0, 1], c='black', s=12)

    # ================================================================
    # PANEL 3 — ar-CRPS: deterministic rollouts (W-only, no process
    #   noise), each surrounded by a continuously expanding tube
    #   whose half-width grows as sqrt(t) * base_width, reflecting
    #   the accumulated process-noise variance.
    # ================================================================
    elif i == 3:
        ax.plot(gx, gy, color='k', lw=1.2, ls='--', alpha=0.7, zorder=2)

        rng = np.random.default_rng(42)
        N_W = 2
        W_COLORS = [M_ECRPS_COLOR, "tab:pink"]
        BASE_WIDTH = 1.1 * float(CONE_SPREAD)  # make terminal Gaussian comparable to cone endings
        y_min, y_max = ax.get_ylim()
        y_clip_margin = BASE_WIDTH + 1.0

        for w_idx in range(N_W):
            col_w = W_COLORS[w_idx]

            # Deterministic rollout: autoregressive, no process noise
            pts = TW.copy().astype(float)
            w_bias = 6 * (0.5 if w_idx == 0 else -1.2)
            for h in range(1, 5):
                step_var = (rng.random() - 0.5) * 2 * h
                pts[h, 1] = np.clip(
                    pts[h-1, 1] + (TW[h, 1] - TW[h-1, 1])
                    + w_bias * (0.6 + 0.15 * h) + step_var,
                    y_min + y_clip_margin,
                    y_max - y_clip_margin,
                )

            # Smooth the rollout trajectory
            sx, sy = smooth_pts(pts)
            n = len(sx)

            end = pts[-1]
            final_half_width = BASE_WIDTH

            # Width grows smoothly from 0 to the chosen final half-width
            # with zero slope at both ends for a softer terminal transition.
            t = np.linspace(0, 1, n)
            smooth_t = t * t * (3.0 - 2.0 * t)
            shift = final_half_width * t ** 0.9

            # Upper and lower envelopes: same x, shifted y.
            # Flatten the final segment so the tube meets the terminal
            # Gaussian ellipse with a cone-like horizontal tangent.
            upper_x = sx.copy()
            lower_x = sx.copy()
            upper_y = sy + shift
            lower_y = sy - shift

            blend_frac = 0.15
            u = np.linspace(0.0, 1.0, n)
            blend = np.clip((u - (1.0 - blend_frac)) / blend_frac, 0.0, 1.0)
            blend_smooth = blend * blend * (3.0 - 2.0 * blend)
            top_end = end[1] + final_half_width
            bot_end = end[1] - final_half_width
            # upper_y = (1.0 - blend_smooth) * upper_y + blend_smooth * top_end
            lower_y = (1.0 - blend_smooth) * lower_y + blend_smooth * bot_end

            # Fill one tube per W with a rounded terminal cap to blend into
            # the final Gaussian ellipse more naturally.
            cap_rx = 6.0
            theta = np.linspace(np.pi / 2.0, -np.pi / 2.0, 32)
            cap_x = end[0] + cap_rx * np.cos(theta)
            cap_y = end[1] + final_half_width * np.sin(theta)
            tube_x = np.concatenate([upper_x, cap_x, lower_x[::-1]])
            tube_y = np.concatenate([upper_y, cap_y, lower_y[::-1]])
            ax.fill(
                tube_x, tube_y,
                facecolor=col_w, alpha=CONE_BODY_ALPHA,
                edgecolor=col_w, lw=CONE_BODY_LW, zorder=3,
                joinstyle='round'
            )

            # Draw the centre rollout line
            ax.plot(sx, sy, color=col_w, lw=1.0, alpha=0.5, zorder=4)

            # Show discrete autoregressive predicted states at each horizon.
            # These points make the rollout dependency chain explicit.
            step_sizes = np.linspace(9, 9, len(pts) - 2)
            ax.scatter(
                pts[1:-1, 0], pts[1:-1, 1],
                s=step_sizes,
                c=col_w,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.35,
                zorder=6,
            )
                
            # Terminal Gaussian ellipse + crosshair (same style as other panels)
            ax.add_patch(Ellipse((end[0], end[1]), 12, 2 * final_half_width,
                                  facecolor=col_w, alpha=CONE_END_ALPHA, edgecolor=col_w,
                                  lw=CONE_END_LW, zorder=5))
            ax.plot([end[0]-3, end[0]+3], [end[1], end[1]],
                    color=col_w, lw=CONE_CROSS_LW, alpha=CONE_CROSS_ALPHA, zorder=5)
            ax.plot([end[0], end[0]], [end[1]-3, end[1]+3],
                    color=col_w, lw=CONE_CROSS_LW, alpha=CONE_CROSS_ALPHA, zorder=5)

        ax.scatter(TW[:, 0], TW[:, 1], c='black', s=9, zorder=12)
        ax.scatter(TW[0, 0], TW[0, 1], c='black', s=12)

    ax.text(0.5, 0.97, subs[i],
            transform=ax.transAxes, ha='center', color='k',
            family='monospace', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='none', alpha=1))
    
    ax.text(0.05, 0.95, r"$\mathcal{X}$",
        transform=ax.transAxes, ha='left', va='top', color='k',
        family='monospace', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', edgecolor='none', alpha=1))

    sns.despine(ax=ax, left=True, offset=3, trim=True)

fig.subplots_adjust(wspace=0.05, top=0.98, right=1.0, left=0.0, bottom=0.12)

draw_penalty_inset(axes[1], fig, is_crps=False)
draw_penalty_inset(axes[2], fig, is_crps=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / f"overview_loss_functions.{FIG_FORMAT}")
# PNG for the README.
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/overview_loss_functions.png", dpi=300, facecolor="white")
plt.close()

print("Done.")
