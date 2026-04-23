import numpy as np
import matplotlib.pyplot as plt


def make_stm_3d_view_png(output_path: str = "./graph/stm_3d_view.png") -> None:
    # Geometry consistent with tracking_fast_feedback.py
    point_num = 8
    radius_mm = 23.5
    angle_offset = np.pi / 8
    center_x_mm = 0.0
    center_y_mm = 0.0
    center_z_mm = 400.0

    angles = angle_offset + 2.0 * np.pi * np.arange(point_num) / point_num
    xs = center_x_mm + radius_mm * np.cos(angles)
    ys = center_y_mm + radius_mm * np.sin(angles)
    zs = np.full(point_num, center_z_mm)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )

    fig = plt.figure(figsize=(13.5, 7.6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # XY plane at z=0 (transducer plane reference)
    plane_lim = 30
    px = np.linspace(-plane_lim, plane_lim, 2)
    py = np.linspace(-plane_lim, plane_lim, 2)
    PX, PY = np.meshgrid(px, py)
    PZ = np.zeros_like(PX)
    ax.plot_surface(PX, PY, PZ, color="#E5E7EB", alpha=0.35, linewidth=0)

    # Circle path and foci at levitation height
    ring_x = center_x_mm + radius_mm * np.cos(np.linspace(0.0, 2.0 * np.pi, 300))
    ring_y = center_y_mm + radius_mm * np.sin(np.linspace(0.0, 2.0 * np.pi, 300))
    ring_z = np.full_like(ring_x, center_z_mm)
    ax.plot(ring_x, ring_y, ring_z, color="#0F766E", lw=3.0, alpha=0.95)

    # Center focus and foci points
    ax.scatter([center_x_mm], [center_y_mm], [center_z_mm], s=300, c="#C2410C", depthshade=False)
    ax.scatter(xs, ys, zs, s=400, c="#0EA5E9", edgecolors="#075985", linewidths=1.0, depthshade=False)

    # Sequence direction arrows (short tangent vectors)
    # for i in range(point_num):
    #     j = (i + 1) % point_num
    #     dx = xs[j] - xs[i]
    #     dy = ys[j] - ys[i]
    #     ax.quiver(
    #         xs[i],
    #         ys[i],
    #         zs[i],
    #         dx,
    #         dy,
    #         0.0,
    #         color="#0369A1",
    #         linewidth=1.2,
    #         arrow_length_ratio=0.22,
    #         alpha=0.95,
    #         length=0.55,
    #         normalize=False,
    #     )

    # Labels F1...F8
    label_offsets = {
        4: (-5.0, 5.0, 0.0),
        5: (-5.0, -1.5, 0.0),
        6: (4.0, -12.0, 0.0),
        7: (5.0, -12.0, 0.0),
        8: (6.5, -4.0, 0.0),
    }

    for i, (x, y, z) in enumerate(zip(xs, ys, zs), start=1):
        dx, dy, dz = label_offsets.get(i, (2.0, 2.0, 5.0))
        ax.text(x + dx, y + dy, z + dz, f"{i}", color="#0C4A6E", weight="bold", fontsize=20)

    # Radius and levitation height guides
    ax.plot([center_x_mm, xs[0]], [center_y_mm, ys[0]], [center_z_mm, center_z_mm], color="#B45309", ls="--", lw=1.8)
    ax.text(
        center_x_mm + 0.15 * (xs[0] - center_x_mm) + 2.0,
        center_y_mm + 0.15 * (ys[0] - center_y_mm),
        center_z_mm + 5.0,
        f"{radius_mm:.1f} mm",
        color="#92400E",
        weight="bold",
        fontsize=18,
    )

    ax.plot([center_x_mm, center_x_mm], [center_y_mm, center_y_mm], [0.0, center_z_mm], color="#7C3AED", ls="--", lw=1.6)
    ax.text(4.0, -2.0, center_z_mm * 0.45, f"z = {center_z_mm:.0f} mm", color="#6D28D9", fontsize=20, weight="bold")

    # Axis settings
    ax.set_xlim(-plane_lim, plane_lim)
    ax.set_ylim(-plane_lim, plane_lim)
    ax.set_zlim(0, 450)
    ax.set_xlabel("x [mm]", labelpad=12, fontsize=20)
    ax.set_ylabel("y [mm]", labelpad=12, fontsize=20)
    ax.set_zlabel("z [mm]", labelpad=12, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.zaxis.set_tick_params(labelsize=18)
    ax.view_init(elev=24, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 1.05))
    ax.grid(True, alpha=0.35)

    # ax.set_title("Circular STM in 3D Space", pad=20, fontsize=23, color="#111827", weight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_stm_3d_view_png()
    print("Saved: ./graph/stm_3d_view.png")
