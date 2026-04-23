import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def make_stm_top_view_png(output_path: str = "./graph/stm_top_view_slide.png") -> None:
    # Geometry from tracking_fast_feedback.py
    point_num = 8
    radius_mm = 23.5
    angle_offset = np.pi / 8

    angles = angle_offset + 2.0 * np.pi * np.arange(point_num) / point_num
    xs = radius_mm * np.cos(angles)
    ys = radius_mm * np.sin(angles)

    # Slide-oriented style
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 24,
            "axes.labelsize": 14,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.6), dpi=220)  # 16:9 slide-friendly
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Background ring
    ring = Circle((0, 0), radius_mm, edgecolor="#0F766E", facecolor="none", lw=3.2, alpha=0.92)
    ax.add_patch(ring)

    # Center focus
    ax.scatter([0], [0], s=300, c="#C2410C", marker="o", zorder=5)
    # ax.text(1.6, 1.6, "Center focus (tx, ty, tz)", color="#9A3412", weight="bold")

    # Foci points and sequence arrows
    ax.scatter(xs, ys, s=400, c="#0EA5E9", edgecolors="#075985", linewidths=1.2, zorder=6)

    label_offsets = {
        4: (-3.0, 2.0),
        7: (2.0, -2.0),
    }

    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        dx, dy = label_offsets.get(i, (1.3, 1.3))
        ax.text(x + dx, y + dy, f"{i}", color="#0C4A6E", weight="bold", fontsize=20)

    for i in range(point_num):
        j = (i + 1) % point_num
        ax.annotate(
            "",
            xy=(xs[j], ys[j]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(
                arrowstyle="->",
                color="#0369A1",
                lw=1.8,
                alpha=0.95,
                shrinkA=15,
                shrinkB=15,
            ),
            zorder=4,
        )

    # Radius indication
    ax.plot([0, xs[0]], [0, ys[0]], color="#B45309", lw=2.0, ls="--", alpha=0.9)
    ax.text(xs[0] * 0.15 + 1.4, ys[0] * 0.1 - 1.4, f"R = {radius_mm:.1f} mm", color="#92400E", weight="bold", fontsize=18)

    # Title and spec box
    # ax.set_title("Circular STM Foci (Top View)", color="#111827", pad=18, weight="bold")

    # spec_text = (
    #     "Spec\n"
    #     "- Number of foci: 8\n"
    #     "- Angular offset: pi/8\n"
    #     "- STM update: 100 Hz\n"
    #     "- Motion: F1 -> F2 -> ... -> F8 -> F1"
    # )
    # ax.text(
    #     1.02,
    #     0.05,
    #     spec_text,
    #     transform=ax.transAxes,
    #     va="bottom",
    #     ha="left",
    #     fontsize=12.5,
    #     bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.3),
    #     color="#3F3F46",
    # )

    # Axis formatting
    lim = 30
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]", fontsize=20)
    ax.set_ylabel("y [mm]", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, color="#D1D5DB", lw=0.8, alpha=0.75)

    # Light axes lines
    ax.axhline(0, color="#9CA3AF", lw=1.0, alpha=0.9)
    ax.axvline(0, color="#9CA3AF", lw=1.0, alpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_stm_top_view_png()
    print("Saved: ./graph/stm_top_view_slide.png")
