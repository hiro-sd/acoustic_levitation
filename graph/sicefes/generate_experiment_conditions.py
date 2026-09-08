from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches


OUTPUT_DIR = Path(__file__).resolve().parent

TEAL = "#15909d"
TEAL_DARK = "#086f78"
ORANGE = "#e4773d"
ORANGE_DARK = "#b34c1f"
INK = "#263b4a"
MUTED = "#82919c"
PALE = "#d8e1e7"
ARRAY = "#25394a"


def draw_array(ax, center_x: float) -> None:
    left, right = center_x - 2.15, center_x + 2.15
    bottom, top = 0.28, 0.88
    top_inset = 0.34
    vertices = [
        (left, bottom),
        (right, bottom),
        (right - top_inset, top),
        (left + top_inset, top),
    ]
    ax.add_patch(
        patches.Polygon(
            vertices,
            closed=True,
            facecolor=ARRAY,
            edgecolor="#132532",
            linewidth=2.0,
            zorder=1,
        )
    )

    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        x_bottom = left + fraction * (right - left)
        x_top = left + top_inset + fraction * (right - left - 2.0 * top_inset)
        ax.plot(
            [x_bottom, x_top],
            [bottom, top],
            color="#d7dfe5",
            linewidth=3.2,
            zorder=2,
        )

    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        y = bottom + fraction * (top - bottom)
        inset = top_inset * fraction
        ax.plot(
            [left + inset, right - inset],
            [y, y],
            color="#d7dfe5",
            linewidth=3.2,
            zorder=2,
        )


def draw_sphere(ax, x: float, y: float, radius: float = 0.47) -> None:
    ax.add_patch(
        patches.Circle(
            (x, y),
            radius,
            facecolor="#e2e9ef",
            edgecolor="#637a8d",
            linewidth=3.0,
            zorder=8,
        )
    )
    ax.add_patch(
        patches.Ellipse(
            (x - 0.08, y + 0.13),
            0.30,
            0.14,
            facecolor="white",
            edgecolor="none",
            alpha=0.62,
            zorder=9,
        )
    )


def draw_field(
    ax,
    x: float,
    y: float,
    *,
    color: str,
    alpha: float = 1.0,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        patches.Arc(
            (x, y + 0.02),
            1.42,
            1.30,
            theta1=188,
            theta2=352,
            color=color,
            linewidth=6.0,
            alpha=alpha,
            linestyle=(0, (1.5, 2.2)) if dashed else "-",
            zorder=5,
        )
    )


def draw_camera(ax, x: float, y: float, facing: str) -> tuple[float, float]:
    if facing == "down":
        width, height = 0.25, 0.42
    else:
        width, height = 0.42, 0.25
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=INK,
            edgecolor=INK,
            linewidth=1.2,
            zorder=10,
        )
    )
    if facing == "down":
        lens_x, lens_y = x, y - (height / 2.0 + 0.10)
        triangle = [
            (x - 0.10, y - height / 2.0),
            (x + 0.10, y - height / 2.0),
            (lens_x, lens_y),
        ]
    else:
        direction = 1.0 if facing == "right" else -1.0
        lens_x, lens_y = x + direction * (width / 2.0 + 0.10), y
        triangle = [
            (x + direction * width / 2.0, y - 0.10),
            (x + direction * width / 2.0, y + 0.10),
            (lens_x, lens_y),
        ]
    ax.add_patch(
        patches.Polygon(
            triangle,
            closed=True,
            facecolor=INK,
            edgecolor=INK,
            zorder=10,
        )
    )
    return lens_x, lens_y


def make_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.4), dpi=180)
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")

    fixed_center = 3.0
    updated_reference = 9.0
    ball_y = 2.36

    ax.text(
        fixed_center,
        4.08,
        "Fixed field",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        updated_reference,
        4.08,
        "Feedback-controlled field",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )

    ax.plot([6.0, 6.0], [0.22, 4.12], color=PALE, linewidth=2.0, zorder=0)
    ax.text(6.0, 2.18, "VS", ha="center", va="center", fontsize=17, fontweight="bold", color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=4.0), zorder=20)

    # Fixed condition: both the sphere and the fixed field are centered on the
    # reference axis.
    draw_array(ax, fixed_center)
    ax.plot(
        [fixed_center, fixed_center],
        [0.94, 3.25],
        color=MUTED,
        linewidth=2.0,
        linestyle=(0, (3, 4)),
        zorder=0,
    )
    draw_field(ax, fixed_center, ball_y, color=ORANGE)
    ax.add_patch(
        patches.Circle((fixed_center, ball_y - 0.62), 0.075, facecolor=ORANGE_DARK, edgecolor="white", linewidth=1.0, zorder=7)
    )
    draw_sphere(ax, fixed_center, ball_y)

    # Feedback condition: when the sphere moves right, the controller shifts
    # the field center left to generate a restoring force toward the reference.
    draw_array(ax, updated_reference)
    ax.plot(
        [updated_reference, updated_reference],
        [0.94, 3.25],
        color=MUTED,
        linewidth=2.0,
        linestyle=(0, (3, 4)),
        zorder=0,
    )
    sphere_x = updated_reference + 0.14
    updated_center = updated_reference - 0.16
    draw_field(ax, updated_reference, ball_y, color=TEAL, alpha=0.24, dashed=True)
    draw_field(ax, updated_center, ball_y, color=TEAL)
    ax.add_patch(
        patches.Circle((updated_center, ball_y - 0.62), 0.075, facecolor=TEAL_DARK, edgecolor="white", linewidth=1.0, zorder=7)
    )
    draw_sphere(ax, sphere_x, ball_y)

    # The short arrow indicates the restoring-force direction.
    ax.add_patch(
        patches.FancyArrowPatch(
            (sphere_x + 0.30, ball_y + 0.61),
            (sphere_x - 0.34, ball_y + 0.61),
            arrowstyle="-|>",
            color=TEAL_DARK,
            linewidth=3.0,
            mutation_scale=17,
            zorder=11,
        )
    )

    top_lens = draw_camera(ax, updated_reference, 3.62, "down")
    side_lens = draw_camera(ax, 11.10, ball_y, "left")
    for lens in (top_lens, side_lens):
        ax.plot(
            [lens[0], sphere_x],
            [lens[1], ball_y],
            color=MUTED,
            linewidth=1.5,
            linestyle=(0, (2, 3)),
            alpha=0.78,
            zorder=3,
        )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    fig.savefig(
        OUTPUT_DIR / "experiment_fixed_vs_realtime.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "experiment_fixed_vs_realtime.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
