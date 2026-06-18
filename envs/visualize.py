"""
2D visualization of PCB board with placed test points and routed traces.
"""

import numpy as np
from typing import List, Tuple, Optional
from envs.board import BoardSpec, TP_TO_TP_MIN

# matplotlib is imported lazily inside plot_board() so this module — and the PIL
# renderer below — work in environments without matplotlib installed.


def render_board_png(board: BoardSpec, test_points=None, paths=None,
                     filename: str = "board.png", scale: int = 6,
                     title: Optional[str] = None) -> str:
    """Render board, obstacles, starts, test points and routed traces to a PNG
    using PIL only (no matplotlib). Returns the filename."""
    from PIL import Image, ImageDraw
    import colorsys

    pad = 22
    head = 26 if title else 0
    W = int(board.width * scale)
    H = int(board.height * scale)
    img = Image.new("RGB", (W + 2 * pad, H + 2 * pad + head), "white")
    d = ImageDraw.Draw(img)

    def wx(x):
        return pad + (x - board.x_min) * scale

    def wy(y):  # flip vertical so +y is up
        return pad + head + (board.height - (y - board.y_min)) * scale

    # board outline
    d.rectangle([wx(board.x_min), wy(board.y_max), wx(board.x_max), wy(board.y_min)],
                outline="black", width=2)
    # rectangular obstacles
    for o in board.rect_obstacles:
        xn, yn, xx, yx = o.bounds
        d.rectangle([wx(xn), wy(yx), wx(xx), wy(yn)], fill=(255, 175, 175),
                    outline=(190, 0, 0))
    # circular obstacles
    for o in board.circ_obstacles:
        d.ellipse([wx(o.cx - o.radius), wy(o.cy + o.radius),
                   wx(o.cx + o.radius), wy(o.cy - o.radius)],
                  fill=(255, 150, 150), outline=(150, 0, 0))
    # connector outline
    if board.connector_w > 0:
        d.rectangle([wx(board.connector_x), wy(board.connector_y + board.connector_h),
                     wx(board.connector_x + board.connector_w), wy(board.connector_y)],
                    outline=(150, 0, 190), width=2)

    n = max(len(board.traces), 1)
    colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / n, 0.85, 0.9))
              for i in range(n)]
    # routed traces
    if paths:
        for i, p in enumerate(paths):
            if p:
                d.line([(wx(x), wy(y)) for x, y in p], fill=colors[i % n], width=2)
    # starts (black squares)
    for t in board.traces:
        x, y = wx(t.start_x), wy(t.start_y)
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill="black")
    # test points (colored, black edge)
    if test_points:
        for i, (tx, ty) in enumerate(test_points):
            x, y = wx(tx), wy(ty)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=colors[i % n], outline="black")
    if title:
        d.text((pad, 7), title, fill="black")
    img.save(filename)
    return filename


def plot_board(
    board: BoardSpec,
    test_points: Optional[List[Tuple[float, float]]] = None,
    paths: Optional[List[Optional[List[Tuple[float, float]]]]] = None,
    candidates: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
    title: str = "PCB Test Point Placement",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot the board with obstacles, starting points, test points, and traces.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Board outline
    board_rect = patches.Rectangle(
        (board.x_min, board.y_min), board.width, board.height,
        linewidth=2, edgecolor='black', facecolor='#f5f5f5', zorder=0,
    )
    ax.add_patch(board_rect)

    # TP edge clearance zone (dashed)
    from envs.board import TP_TO_EDGE_MIN
    inner_rect = patches.Rectangle(
        (board.x_min + TP_TO_EDGE_MIN, board.y_min + TP_TO_EDGE_MIN),
        board.width - 2 * TP_TO_EDGE_MIN,
        board.height - 2 * TP_TO_EDGE_MIN,
        linewidth=1, edgecolor='gray', facecolor='none',
        linestyle='--', zorder=1, label='TP edge clearance',
    )
    ax.add_patch(inner_rect)

    # Connector outline
    if board.connector_w > 0:
        conn_rect = patches.Rectangle(
            (board.connector_x, board.connector_y),
            board.connector_w, board.connector_h,
            linewidth=1.5, edgecolor='purple', facecolor='#e8d5f5',
            alpha=0.5, zorder=2, label='Connector',
        )
        ax.add_patch(conn_rect)

    # Rectangular obstacles
    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1, edgecolor='red', facecolor='#ffcccc',
            alpha=0.7, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(obs.cx, obs.cy, obs.name, fontsize=6, ha='center', va='center',
                color='red', zorder=10)

    # Circular obstacles (UPTHs)
    for obs in board.circ_obstacles:
        circle = patches.Circle(
            (obs.cx, obs.cy), obs.radius,
            linewidth=1, edgecolor='darkred', facecolor='#ff9999',
            alpha=0.7, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(obs.cx, obs.cy - obs.radius - 0.5, obs.name,
                fontsize=6, ha='center', color='darkred', zorder=10)

    # Candidate positions
    if candidates is not None:
        if candidate_mask is not None:
            valid = candidates[candidate_mask[:len(candidates)] > 0]
            invalid = candidates[candidate_mask[:len(candidates)] == 0]
            if len(invalid) > 0:
                ax.scatter(invalid[:, 0], invalid[:, 1], c='lightgray',
                          s=8, marker='.', zorder=2, alpha=0.5)
            if len(valid) > 0:
                ax.scatter(valid[:, 0], valid[:, 1], c='lightblue',
                          s=12, marker='.', zorder=2, alpha=0.7, label='Valid candidates')
        else:
            ax.scatter(candidates[:, 0], candidates[:, 1], c='lightblue',
                      s=12, marker='.', zorder=2, alpha=0.7, label='Candidates')

    # Starting points
    trace_colors = plt.cm.tab20(np.linspace(0, 1, len(board.traces)))
    for i, trace in enumerate(board.traces):
        ax.plot(trace.start_x, trace.start_y, 's', color=trace_colors[i],
                markersize=5, zorder=5)

    # Routed traces
    if paths is not None:
        for i, path in enumerate(paths):
            if path is None:
                continue
            color = trace_colors[i % len(trace_colors)]
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '-', color=color, linewidth=1.0, alpha=0.8, zorder=4)

    # Test points
    if test_points is not None:
        for i, (tx, ty) in enumerate(test_points):
            color = trace_colors[i % len(trace_colors)]
            ax.plot(tx, ty, 'o', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=6)
            # Draw 13mm exclusion zone
            excl = patches.Circle(
                (tx, ty), TP_TO_TP_MIN / 2,
                linewidth=0.5, edgecolor=color, facecolor='none',
                linestyle=':', alpha=0.3, zorder=2,
            )
            ax.add_patch(excl)

    # Starting point labels
    ax.plot([], [], 's', color='gray', markersize=5, label='Starting points')
    if test_points:
        ax.plot([], [], 'o', color='gray', markersize=8,
                markeredgecolor='black', label='Test points')

    ax.set_xlim(board.x_min - 2, board.x_max + 2)
    ax.set_ylim(board.y_min - 2, board.y_max + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.close()
    return fig