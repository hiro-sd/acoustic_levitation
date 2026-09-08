from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


OUTPUT_DIR = Path(__file__).resolve().parent

BLUE = "#159bc4"
BLUE_DARK = "#086f94"
ORANGE = "#d94801"
FIELD_ORANGE = "#e4773d"
PURPLE = "#6b45ed"
ARRAY_DARK = "#25394a"
ARRAY_GRID = "#8393a1"
RED = "#d95643"


def _draw_autd_array_3d(ax) -> None:
    # The actual aperture is much larger than the circular trajectory. It is
    # therefore drawn as one continuous schematic plane filling the footprint,
    # rather than as nine small modules inside a 60-mm square.
    extent = 34.0
    vertices = [
        (-extent, -extent, 0.0),
        (extent, -extent, 0.0),
        (extent, extent, 0.0),
        (-extent, extent, 0.0),
    ]
    array_plane = Poly3DCollection(
        [vertices],
        facecolor=ARRAY_DARK,
        edgecolor="#122331",
        linewidth=1.2,
        alpha=0.97,
    )
    array_plane.set_zorder(1)
    ax.add_collection3d(array_plane)

    dot_positions = np.linspace(-31.0, 31.0, 15)
    gx, gy = np.meshgrid(dot_positions, dot_positions)
    ax.scatter(
        gx.ravel(),
        gy.ravel(),
        np.full(gx.size, 0.8),
        s=4.2,
        c="#b5c1ca",
        depthshade=False,
        zorder=2,
    )


def make_levitation_geometry() -> None:
    radius = 23.5
    z_center = 400.0
    sphere_radius = 25.0
    point_angles = np.pi / 8.0 + np.arange(8) * 2.0 * np.pi / 8.0
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 360)

    fig = plt.figure(figsize=(7.4, 7.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    _draw_autd_array_3d(ax)

    rx = radius * np.cos(ring_angles)
    ry = radius * np.sin(ring_angles)
    ax.plot(rx, ry, np.full_like(rx, z_center), color=FIELD_ORANGE, linewidth=2.7, zorder=6)

    px = radius * np.cos(point_angles)
    py = radius * np.sin(point_angles)
    pz = np.full(8, z_center)
    ax.scatter(
        px,
        py,
        pz,
        s=118,
        c=FIELD_ORANGE,
        edgecolors="#b34c1f",
        linewidths=1.0,
        depthshade=False,
        zorder=8,
    )

    for idx, (x, y, z) in enumerate(zip(px, py, pz), start=1):
        ax.text(
            x * 1.12,
            y * 1.12,
            z + 4.0,
            str(idx),
            color="#a6421a",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=10,
        )

    # Center, trajectory radius, and height retain the information in the original plot.
    ax.scatter([0.0], [0.0], [z_center], s=96, c=ORANGE, depthshade=False, zorder=10)
    ax.plot([0.0, px[0]], [0.0, py[0]], [z_center, z_center], color=ORANGE, linewidth=2.0, zorder=9)
    ax.text(
        px[0] * 0.48,
        py[0] * 0.48,
        z_center + 7.0,
        "23.5 mm",
        color="#9f3508",
        fontsize=10,
        fontweight="bold",
        zorder=10,
    )

    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, z_center], color=PURPLE, linestyle="--", linewidth=1.8, zorder=4)
    ax.text(
        2.0,
        -2.0,
        205.0,
        "z = 400 mm",
        color=PURPLE,
        fontsize=11,
        fontweight="bold",
        zorder=10,
    )

    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(-35.0, 35.0)
    ax.set_zlim(0.0, 450.0)
    ax.set_xticks(np.arange(-30, 31, 10))
    ax.set_yticks(np.arange(-30, 31, 10))
    ax.set_zticks(np.arange(0, 451, 50))
    ax.set_xlabel("x [mm]", labelpad=8, fontsize=11)
    ax.set_ylabel("y [mm]", labelpad=8, fontsize=11)
    ax.set_zlabel("z [mm]", labelpad=8, fontsize=11)
    ax.tick_params(labelsize=9, pad=1)
    ax.view_init(elev=27.0, azim=-56.0)
    ax.set_box_aspect((1.0, 1.0, 1.45))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.45))
        axis.pane.set_edgecolor((0.55, 0.60, 0.65, 0.45))
        axis._axinfo["grid"]["color"] = (0.60, 0.64, 0.68, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.55

    # A physically 50-mm sphere becomes visually flattened when the 0-450 mm
    # z-axis is compressed into this compact 3D plot. Draw its projected
    # silhouette as a true circle behind the equatorial focal trajectory.
    fig.canvas.draw()

    def projected_display(point: tuple[float, float, float]) -> np.ndarray:
        x_2d, y_2d, _ = proj3d.proj_transform(*point, ax.get_proj())
        return np.asarray(ax.transData.transform((x_2d, y_2d)), dtype=float)

    center_display = projected_display((0.0, 0.0, z_center))
    radius_display = max(
        np.linalg.norm(projected_display((sphere_radius, 0.0, z_center)) - center_display),
        np.linalg.norm(projected_display((0.0, sphere_radius, z_center)) - center_display),
    )
    center_axes = ax.transAxes.inverted().transform(center_display)
    axes_box = ax.get_window_extent()
    sphere_width = 2.0 * radius_display / axes_box.width
    sphere_height = 2.0 * radius_display / axes_box.height

    sphere_patch = patches.Ellipse(
        center_axes,
        sphere_width,
        sphere_height,
        transform=ax.transAxes,
        facecolor="#d9e2ea",
        edgecolor="#6f8292",
        linewidth=2.2,
        alpha=0.66,
        zorder=3,
    )
    sphere_patch.do_3d_projection = lambda: 0
    ax.add_patch(sphere_patch)
    sphere_highlight = patches.Ellipse(
        (center_axes[0] - 0.16 * sphere_width, center_axes[1] + 0.18 * sphere_height),
        0.34 * sphere_width,
        0.18 * sphere_height,
        transform=ax.transAxes,
        facecolor="#ffffff",
        edgecolor="none",
        alpha=0.48,
        zorder=4,
    )
    sphere_highlight.do_3d_projection = lambda: 0
    ax.add_patch(sphere_highlight)

    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.05, top=0.98)
    fig.savefig(OUTPUT_DIR / "challenge_levitation_with_sphere_array.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "challenge_levitation_with_sphere_array.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_sphere(ax, x: float, y: float, radius: float, alpha: float) -> None:
    sphere = patches.Circle(
        (x, y),
        radius,
        facecolor="#e2e9ef",
        edgecolor="#637a8d",
        linewidth=3.0,
        alpha=alpha,
        zorder=8,
    )
    ax.add_patch(sphere)
    highlight = patches.Ellipse(
        (x - 0.13 * radius, y + 0.23 * radius),
        0.62 * radius,
        0.30 * radius,
        facecolor="white",
        edgecolor="none",
        alpha=0.52 * alpha,
        zorder=9,
    )
    ax.add_patch(highlight)


def make_unstable_levitation() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Stylized 3 x 3 AUTD arrangement.
    array = patches.Polygon(
        [(0.60, 0.20), (5.40, 0.20), (4.85, 1.12), (1.15, 1.12)],
        closed=True,
        facecolor=ARRAY_DARK,
        edgecolor="#132532",
        linewidth=2.0,
        zorder=1,
    )
    ax.add_patch(array)
    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        x_bottom = 0.60 + fraction * 4.80
        x_top = 1.15 + fraction * 3.70
        ax.plot([x_bottom, x_top], [0.20, 1.12], color="#d7dfe5", linewidth=4.0, zorder=2)
    for row_fraction in (1.0 / 3.0, 2.0 / 3.0):
        y = 0.20 + row_fraction * 0.92
        left = 0.60 + (y - 0.20) / 0.92 * 0.55
        right = 5.40 - (y - 0.20) / 0.92 * 0.55
        ax.plot([left, right], [y, y], color="#d7dfe5", linewidth=4.0, zorder=2)

    center_x, center_y, ball_radius = 3.0, 3.72, 0.55

    # The cross-section of the time-averaged toroidal field closely follows
    # the sphere's lower hemisphere.
    ax.add_patch(
        patches.Arc(
            (center_x, center_y + 0.03),
            1.62,
            1.55,
            theta1=188,
            theta2=352,
            color=FIELD_ORANGE,
            linewidth=7.0,
            zorder=3,
        )
    )

    # Nearby translucent exposures and short bidirectional arrows show that
    # the sphere oscillates around its nominal position.
    ghost_offset = 0.50
    _draw_sphere(ax, center_x - ghost_offset, center_y, ball_radius, 0.16)
    _draw_sphere(ax, center_x + ghost_offset, center_y, ball_radius, 0.16)
    _draw_sphere(ax, center_x, center_y + ghost_offset, ball_radius, 0.16)
    _draw_sphere(ax, center_x, center_y - ghost_offset, ball_radius, 0.16)

    arrow_style = dict(
        arrowstyle="<->",
        color="#415462",
        linewidth=2.8,
        mutation_scale=15,
        shrinkA=0,
        shrinkB=0,
        zorder=7,
    )
    ax.add_patch(
        patches.FancyArrowPatch(
            (center_x - 0.90, center_y + 0.72),
            (center_x - 0.28, center_y + 0.72),
            **arrow_style,
        )
    )
    ax.add_patch(
        patches.FancyArrowPatch(
            (center_x + 0.76, center_y - 0.30),
            (center_x + 0.76, center_y + 0.30),
            **arrow_style,
        )
    )

    # Draw the nominal position last so it remains the visual anchor.
    _draw_sphere(ax, center_x, center_y, ball_radius, 1.0)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(OUTPUT_DIR / "challenge_unstable_levitation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "challenge_unstable_levitation.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_levitation_geometry()
    make_unstable_levitation()
