#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install with `pip install numpy` "
        "or your environment manager, then rerun."
    ) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib pillow` "
        "or your environment manager, then rerun."
    ) from exc


L = 0.05
DC = 2.6429
NUM_STATE = 19
NUM_CONTROL = 10
STEP_WIDTH = NUM_STATE + NUM_CONTROL


def load_trajectory(path: Path) -> np.ndarray:
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        if raw.size % STEP_WIDTH != 0:
            raise ValueError(f"Invalid flat trajectory size: {raw.size}, not divisible by {STEP_WIDTH}.")
        return raw.reshape((-1, STEP_WIDTH))
    if raw.shape[1] == STEP_WIDTH:
        return raw
    if raw.shape[0] == STEP_WIDTH:
        return raw.T
    raise ValueError(f"Invalid trajectory shape: {raw.shape}, expected (*, {STEP_WIDTH}) or ({STEP_WIDTH}, *).")


def t_shape_local_vertices() -> np.ndarray:
    x1 = -2.0 * L
    x2 = -0.5 * L
    x3 = 0.5 * L
    x4 = 2.0 * L
    y1 = -DC * L
    y2 = (3.0 - DC) * L
    y3 = (4.0 - DC) * L
    return np.array(
        [
            [x1, y3],
            [x4, y3],
            [x4, y2],
            [x3, y2],
            [x3, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
    )


def transform(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x, y])


def pick_default_input(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "src/build/examples/pushT_solution.txt",
        repo_root / "src/build/examples/pushT_solution_seg_00.txt",
        Path.cwd() / "pushT_solution.txt",
        Path.cwd() / "pushT_solution_seg_00.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No trajectory file found. Run ./src/build/examples/pushT_example first, "
        "or pass --input /path/to/pushT_solution.txt"
    )


def parse_segment_index(path: Path) -> int:
    match = re.search(r"_seg_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return 10**9


def discover_segment_files(batch_dir: Optional[str]) -> List[Path]:
    if batch_dir:
        roots = [Path(batch_dir).expanduser().resolve()]
    else:
        repo_root = Path(__file__).resolve().parents[3]
        roots = [
            Path.cwd().resolve(),
            repo_root.resolve(),
            repo_root / "src/build",
            repo_root / "src/build/examples",
        ]

    paths: List[Path] = []
    seen = set()
    searched_roots: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        searched_roots.append(root)
        for p in root.rglob("pushT_solution_seg_*.txt"):
            rp = p.resolve()
            if rp not in seen:
                paths.append(rp)
                seen.add(rp)

    paths.sort(key=parse_segment_index)
    if not paths:
        searched_str = "\n".join([f"  - {r}" for r in searched_roots]) if searched_roots else "  (no existing roots)"
        raise FileNotFoundError(
            "No pushT_solution_seg_*.txt found. "
            "Run pushT_example first, or pass --batch-dir to the folder containing segment files.\n"
            f"Searched roots:\n{searched_str}"
        )
    return paths


def writer_name(fmt: str) -> str:
    return "pillow" if fmt == "gif" else "ffmpeg"


def extension_for(fmt: str) -> str:
    return ".gif" if fmt == "gif" else ".mp4"


def build_animation(
    traj: np.ndarray,
    title: str,
    show_contact: bool,
    target_state: Tuple[float, float, float],
    fps: int,
):
    px = traj[:, 0]
    py = traj[:, 1]
    theta_raw = traj[:, 2]
    cx_local = traj[:, 3]
    cy_local = traj[:, 4]

    c_theta = traj[:, 27]
    s_theta = traj[:, 28]
    theta = np.arctan2(s_theta, c_theta)
    bad = ~np.isfinite(theta)
    theta[bad] = theta_raw[bad]

    local_shape = t_shape_local_vertices()
    shape_radius = np.max(np.linalg.norm(local_shape, axis=1))
    target_x, target_y, target_theta = target_state

    c = np.cos(theta)
    s = np.sin(theta)
    contact_x = px + c * cx_local - s * cy_local
    contact_y = py + s * cx_local + c * cy_local

    target_poly = transform(local_shape, target_x, target_y, target_theta)
    target_poly_x = target_poly[:, 0]
    target_poly_y = target_poly[:, 1]

    margin = 0.08
    x_min = min(np.min(px), np.min(contact_x), np.min(target_poly_x), target_x) - shape_radius - margin
    x_max = max(np.max(px), np.max(contact_x), np.max(target_poly_x), target_x) + shape_radius + margin
    y_min = min(np.min(py), np.min(contact_y), np.min(target_poly_y), target_y) - shape_radius - margin
    y_max = max(np.max(py), np.max(contact_y), np.max(target_poly_y), target_y) + shape_radius + margin

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)

    ax.plot(px, py, "--", color="#1f77b4", linewidth=1.5, alpha=0.7, label="COM path")
    if show_contact:
        ax.plot(contact_x, contact_y, "--", color="#444444", linewidth=1.0, alpha=0.7, label="Contact path")

    target_patch = Polygon(
        target_poly,
        closed=True,
        facecolor="none",
        edgecolor="#2d6a4f",
        linewidth=2.0,
        linestyle="--",
        label="Target T",
    )
    ax.add_patch(target_patch)
    target_label = f"Target ({target_x:.3f},{target_y:.3f},{target_theta:.3f})"
    ax.plot([target_x], [target_y], marker="x", color="#2d6a4f", markersize=8, markeredgewidth=2.0, label=target_label)

    initial_poly = transform(local_shape, px[0], py[0], theta[0])
    t_patch = Polygon(initial_poly, closed=True, facecolor="#86c5ff", edgecolor="#004e98", linewidth=2.0, alpha=0.95)
    ax.add_patch(t_patch)

    (com_dot,) = ax.plot(px[0], py[0], "o", color="#d90429", markersize=6, label="COM")
    (contact_dot,) = ax.plot(contact_x[0], contact_y[0], "o", color="#1b1b1b", markersize=5, label="Contact")
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="upper right")

    def update(frame: int):
        poly = transform(local_shape, px[frame], py[frame], theta[frame])
        t_patch.set_xy(poly)
        com_dot.set_data([px[frame]], [py[frame]])
        contact_dot.set_data([contact_x[frame]], [contact_y[frame]])
        step_text.set_text(f"step: {frame + 1}/{len(px)}")
        return t_patch, com_dot, contact_dot, step_text

    anim = FuncAnimation(fig, update, frames=len(px), interval=1000.0 / max(fps, 1), blit=False)
    return fig, anim


def save_animation(anim, out_path: Path, fmt: str, fps: int) -> None:
    try:
        anim.save(out_path, writer=writer_name(fmt), fps=fps)
    except Exception as exc:
        raise SystemExit(
            f"Failed to save animation to {out_path}: {exc}. "
            "For GIF install pillow; for MP4 install ffmpeg."
        ) from exc


def render_single(args) -> None:
    traj_path = pick_default_input(args.input)
    traj = load_trajectory(traj_path)
    title = f"pushT trajectory - {traj_path.stem}"
    target_state = (args.target_x, args.target_y, args.target_theta)
    fig, anim = build_animation(traj, title, args.show_contact, target_state, args.fps)

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        save_animation(anim, out_path, args.format, args.fps)
        print(f"Saved animation to: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def render_batch(args) -> None:
    segment_files = discover_segment_files(args.batch_dir)
    out_dir = Path(args.save_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_state = (args.target_x, args.target_y, args.target_theta)
    ext = extension_for(args.format)

    print(f"Found {len(segment_files)} segment files. Saving to {out_dir}")
    for idx, seg_file in enumerate(segment_files):
        traj = load_trajectory(seg_file)
        title = f"pushT trajectory - {seg_file.stem}"
        fig, anim = build_animation(traj, title, args.show_contact, target_state, args.fps)
        out_path = out_dir / f"{seg_file.stem}{ext}"
        save_animation(anim, out_path, args.format, args.fps)
        plt.close(fig)
        print(f"[{idx + 1}/{len(segment_files)}] Saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CRISP pushT trajectory.")
    parser.add_argument("--input", type=str, default=None, help="Path to one trajectory txt file")
    parser.add_argument("--save", type=str, default=None, help="Single-mode output path")
    parser.add_argument("--fps", type=int, default=12, help="Animation FPS for display/save")
    parser.add_argument("--show-contact", action="store_true", help="Show contact point trajectory")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Saved animation format")
    parser.add_argument("--target-x", type=float, default=0.0, help="Target x position")
    parser.add_argument("--target-y", type=float, default=0.0, help="Target y position")
    parser.add_argument("--target-theta", type=float, default=0.0, help="Target heading in radians")
    parser.add_argument("--batch", action="store_true", help="Batch-render all pushT_solution_seg_*.txt")
    parser.add_argument("--batch-dir", type=str, default=None, help="Folder containing segment txt files")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="pushT_visualizations",
        help="Batch-mode output folder (one file per segment)",
    )
    args = parser.parse_args()

    if args.batch:
        render_batch(args)
    else:
        render_single(args)


if __name__ == "__main__":
    main()
