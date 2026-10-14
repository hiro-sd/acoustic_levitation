from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUTPUT_DIR = Path(__file__).resolve().parent

ORANGE = "#d94801"
FIELD_ORANGE = "#e4773d"
CENTER_RED = "#d62728"
ARRAY_DARK = "#25394a"
POINT_LABEL_SIZE = 15
DIMENSION_LABEL_SIZE = 10.5
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12


def draw_autd_array_3d(ax) -> None:
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


def make_capture_levitation_figure() -> None:
    # The capture experiment uses a 19-mm spherical radius and a matching
    # circular trajectory. The absolute z position is intentionally omitted,
    # because the target is in free fall during capture.
    radius = 19.0
    sphere_radius = 19.0
    z_center = 400.0
    point_angles = np.pi / 8.0 + np.arange(8) * 2.0 * np.pi / 8.0
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 360)

    fig = plt.figure(figsize=(7.4, 7.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    draw_autd_array_3d(ax)

    rx = radius * np.cos(ring_angles)
    ry = radius * np.sin(ring_angles)
    ax.plot(
        rx,
        ry,
        np.full_like(rx, z_center),
        color=FIELD_ORANGE,
        linewidth=2.7,
        zorder=6,
    )

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

    # The red point marks the shared center of the sphere and circular path.
    ax.scatter(
        [0.0],
        [0.0],
        [z_center],
        s=104,
        c=CENTER_RED,
        edgecolors="#a91016",
        linewidths=0.8,
        depthshade=False,
        zorder=10,
    )
    ax.plot(
        [0.0, px[0]],
        [0.0, py[0]],
        [z_center, z_center],
        color=ORANGE,
        linewidth=2.0,
        zorder=9,
    )
    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(-35.0, 35.0)
    ax.set_zlim(0.0, 450.0)
    ax.set_xticks(np.arange(-30, 31, 10))
    ax.set_yticks(np.arange(-30, 31, 10))
    ax.set_zticks(np.arange(0, 451, 50))
    ax.set_zticklabels([])
    ax.set_xlabel("x [mm]", labelpad=12, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("y [mm]", labelpad=12, fontsize=AXIS_LABEL_SIZE)
    ax.set_zlabel("z [mm]", labelpad=12, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE, pad=2)
    ax.view_init(elev=27.0, azim=-56.0)
    ax.set_box_aspect((1.0, 1.0, 1.45))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.45))
        axis.pane.set_edgecolor((0.55, 0.60, 0.65, 0.45))
        axis._axinfo["grid"]["color"] = (0.60, 0.64, 0.68, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.55

    # Draw the projected sphere silhouette as a true circle. This prevents the
    # compressed 3D z scale from making the physical sphere look flattened.
    fig.canvas.draw()

    def projected_display(point: tuple[float, float, float]) -> np.ndarray:
        x_2d, y_2d, _ = proj3d.proj_transform(*point, ax.get_proj())
        return np.asarray(ax.transData.transform((x_2d, y_2d)), dtype=float)

    def projected_data(point: tuple[float, float, float]) -> tuple[float, float]:
        x_2d, y_2d, _ = proj3d.proj_transform(*point, ax.get_proj())
        return x_2d, y_2d

    center_display = projected_display((0.0, 0.0, z_center))
    radius_display = max(
        np.linalg.norm(
            projected_display((sphere_radius, 0.0, z_center)) - center_display
        ),
        np.linalg.norm(
            projected_display((0.0, sphere_radius, z_center)) - center_display
        ),
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
        (
            center_axes[0] - 0.16 * sphere_width,
            center_axes[1] + 0.18 * sphere_height,
        ),
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

    # Position each number radially outward in screen space. Using display
    # offsets keeps every label clear of its marker despite 3D projection.
    for idx, (x, y, z) in enumerate(zip(px, py, pz), start=1):
        point = (float(x), float(y), float(z))
        point_display = projected_display(point)
        outward = point_display - center_display
        outward /= np.linalg.norm(outward)
        ax.annotate(
            str(idx),
            xy=projected_data(point),
            xytext=tuple(outward * 17.0),
            textcoords="offset points",
            color="#a6421a",
            fontsize=POINT_LABEL_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            annotation_clip=False,
            zorder=11,
        )

    # Place the dimension above the radial line, away from the center and the
    # numbered trajectory points.
    endpoint = (float(px[0]), float(py[0]), z_center)
    endpoint_display = projected_display(endpoint)
    line_direction = endpoint_display - center_display
    line_direction /= np.linalg.norm(line_direction)
    label_normal = np.asarray((-line_direction[1], line_direction[0]))
    if label_normal[1] < 0.0:
        label_normal *= -1.0
    midpoint = (float(px[0] * 0.35), float(py[0] * 0.35), z_center)
    ax.annotate(
        "19.0 mm",
        xy=projected_data(midpoint),
        xytext=tuple(label_normal * 12.0),
        textcoords="offset points",
        color="#9f3508",
        fontsize=DIMENSION_LABEL_SIZE,
        fontweight="bold",
        ha="center",
        va="center",
        annotation_clip=False,
        zorder=11,
    )

    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.05, top=0.98)
    fig.savefig(
        OUTPUT_DIR / "challenge_levitation_with_sphere_array.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    make_capture_levitation_figure()
