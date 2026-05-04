"""
Visualization helper for the heat diffusion project.

Modes:
1) scaling: read results/scaling.csv and print runtime summaries.
2) heatmap: load binary snapshots and show a time-step slider heatmap.
3) surface: load binary snapshots and show a 3D surface with slider.
4) export: write one plain-text field file (csv/dat/out) for MATLAB.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


SNAPSHOT_RE = re.compile(r"rank_(\d{3})_step_(\d{6})\.bin$")


def parse_int(value: str) -> int:
    """Parse integer-like strings safely; return -1 when unavailable."""
    if value is None:
        return -1
    text = str(value).strip()
    if not text:
        return -1
    try:
        return int(text)
    except ValueError:
        return -1


def parse_block_size(row: list, header_idx: dict) -> int:
    """
    Parse one normalized block size.

    Some CSV files append two extra columns where both represent the same
    block size (e.g., block_x and block_y). We collapse those into one value.
    """
    candidates = []
    for key in ("block_size", "block", "block_x", "block_y", "bx", "by"):
        if key in header_idx and header_idx[key] < len(row):
            candidates.append(parse_int(row[header_idx[key]].strip()))

    # Backward-compatibility path for files that have unnamed trailing columns.
    if len(row) > 7:
        candidates.append(parse_int(row[7].strip()))
    if len(row) > 8:
        candidates.append(parse_int(row[8].strip()))

    positive_vals = [v for v in candidates if v > 0]
    if not positive_vals:
        return 0

    # Columns 8 and 9 are expected to match; if they differ, keep first valid.
    return positive_vals[0]


def series_label(variant: str, threads: int) -> str:
    """Build plotting label; include thread count for OMP-based variants."""
    if variant in ("omp", "hybrid") and threads > 0:
        return f"{variant}-t{threads}"
    return variant


def load_scaling_runs(csv_path: str) -> list:
    """Read CSV rows and return normalized scaling run dictionaries."""
    parsed_runs = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    header_idx = {name: idx for idx, name in enumerate(header)}

    for row in data_rows:
        if not row:
            continue

        variant = ""
        nx = ""
        ny = ""
        nranks = ""
        nsteps = ""
        threads = ""
        t = None
        block_size = 0

        # Preferred parse path: named columns.
        if "variant" in header_idx and header_idx["variant"] < len(row):
            variant = row[header_idx["variant"]].strip()
        elif "version" in header_idx and header_idx["version"] < len(row):
            variant = row[header_idx["version"]].strip()
        if "nx" in header_idx and header_idx["nx"] < len(row):
            nx = row[header_idx["nx"]].strip()
        if "ny" in header_idx and header_idx["ny"] < len(row):
            ny = row[header_idx["ny"]].strip()
        if "nranks" in header_idx and header_idx["nranks"] < len(row):
            nranks = row[header_idx["nranks"]].strip()
        if "nsteps" in header_idx and header_idx["nsteps"] < len(row):
            nsteps = row[header_idx["nsteps"]].strip()
        if "omp_threads" in header_idx and header_idx["omp_threads"] < len(row):
            threads = row[header_idx["omp_threads"]].strip()
        elif "threads" in header_idx and header_idx["threads"] < len(row):
            threads = row[header_idx["threads"]].strip()
        block_size = parse_block_size(row, header_idx)

        if "time_s" in header_idx and header_idx["time_s"] < len(row):
            val = row[header_idx["time_s"]].strip()
            if val:
                try:
                    t = float(val)
                except ValueError:
                    t = None
        elif "wall_time_s" in header_idx and header_idx["wall_time_s"] < len(row):
            val = row[header_idx["wall_time_s"]].strip()
            if val:
                try:
                    t = float(val)
                except ValueError:
                    t = None

        # Backward compatibility for old/partial files:
        # executable stdout rows are: version,nx,ny,nranks,nsteps,wall_time_s
        if (not variant or t is None) and len(row) >= 6:
            if not variant:
                variant = row[0].strip()
            if not nx:
                nx = row[1].strip()
            if not ny:
                ny = row[2].strip()
            if not nranks:
                nranks = row[3].strip()
            if not nsteps:
                nsteps = row[4].strip()
            try:
                t = float(row[5].strip())
            except ValueError:
                t = None

        if variant and t is not None:
            parsed_runs.append(
                {
                    "variant": variant,
                    "nx": parse_int(nx),
                    "ny": parse_int(ny),
                    "nranks": parse_int(nranks),
                    "nsteps": parse_int(nsteps),
                    "threads": parse_int(threads),
                    "block_size": block_size,
                    "wall_time_s": t,
                }
            )

    return parsed_runs


def summarize_scaling_runs(parsed_runs: list, csv_path: str) -> None:
    """
    Print sorted aggregated table from scaling runs.

    The output is CSV-style to remain easy to parse in downstream scripts.
    """
    if not parsed_runs:
        print(f"No usable rows found in {csv_path}")
        return

    header, rows = build_scaling_table_rows(parsed_runs)
    print(",".join(header))
    for row in rows:
        print(",".join(row))


def build_scaling_table_rows(parsed_runs: list) -> tuple:
    """
    Build sorted aggregated scaling table rows for text/UI display.

    Returns (header, rows), where each row is a list of strings.
    """
    grouped = defaultdict(list)
    for run in parsed_runs:
        key = (
            run["variant"],
            run["nx"],
            run["ny"],
            run["nranks"],
            run["nsteps"],
            max(1, run["threads"]),
            run.get("block_size", 0),
        )
        grouped[key].append(run["wall_time_s"])

    variant_rank = {"serial": 0, "omp": 1, "cuda": 2, "hybrid": 3}

    def sort_key(item: tuple) -> tuple:
        (variant, nx, ny, nranks, nsteps, threads, block_size), _times = item
        return (
            nx if nx > 0 else 10**9,
            ny if ny > 0 else 10**9,
            variant_rank.get(variant, 99),
            variant,
            nranks if nranks > 0 else 10**9,
            threads if threads > 0 else 10**9,
            block_size if block_size > 0 else 10**9,
            nsteps if nsteps > 0 else 10**9,
        )

    serial_baseline = {}
    for key, times in grouped.items():
        variant, nx, ny, _nranks, nsteps, _threads, _block_size = key
        if variant != "serial":
            continue
        problem_key = (nx, ny, nsteps)
        mean_time = float(np.mean(times))
        if problem_key not in serial_baseline:
            serial_baseline[problem_key] = mean_time
        else:
            serial_baseline[problem_key] = min(serial_baseline[problem_key], mean_time)

    header = [
        "variant",
        "nx",
        "ny",
        "nranks",
        "nsteps",
        "omp_threads",
        "block_size",
        "samples",
        "wall_time_s",
        "speedup_vs_serial",
    ]
    rows = []
    for key, times in sorted(grouped.items(), key=sort_key):
        variant, nx, ny, nranks, nsteps, threads, block_size = key
        mean_t = float(np.mean(times))

        base = serial_baseline.get((nx, ny, nsteps))
        speedup = ""
        if base is not None and mean_t > 0.0:
            speedup = f"{(base / mean_t):.6f}"

        rows.append(
            [
                variant,
                str(nx),
                str(ny),
                str(nranks),
                str(nsteps),
                str(threads),
                str(block_size),
                str(len(times)),
                f"{mean_t:.6f}",
                speedup,
            ]
        )

    return header, rows


def show_scaling_ui_table(parsed_runs: list, title: str) -> None:
    """
    Show a visual UI table of sorted scaling metrics using matplotlib.
    """
    if not parsed_runs:
        print("No scaling rows available for UI table.")
        return

    header, rows = build_scaling_table_rows(parsed_runs)
    if not rows:
        print("No scaling rows available for UI table.")
        return

    nrows = len(rows)
    fig_height = max(5.5, min(0.42 * nrows + 2.4, 24.0))
    fig = plt.figure(figsize=(20, fig_height))
    ax = fig.add_subplot(111)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.55)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")

    fig.suptitle(title, fontsize=14, y=0.985)
    plt.tight_layout()
    plt.show()


def plot_scaling_comparison(parsed_runs: list, save_prefix: str) -> None:
    """
    Plot grouped scaling comparisons:
    - Top: runtime vs nx for each variant (mean + min/max band)
    - Bottom: speedup over serial at matching nx
    """
    points = defaultdict(lambda: defaultdict(list))
    for run in parsed_runs:
        nx = run["nx"]
        if nx <= 0:
            continue
        threads = run["threads"]
        if threads <= 0:
            threads = 1
        label = series_label(run["variant"], threads)
        points[label][nx].append(run["wall_time_s"])

    if not points:
        print("No plotting points with valid nx values were found.")
        return

    variants = sorted(points.keys())
    serial_key = "serial" if "serial" in points else None

    thread_colors = {
        1: "tab:gray",
        2: "tab:brown",
        4: "tab:blue",
        8: "tab:orange",
        12: "tab:olive",
        16: "tab:green",
        24: "tab:cyan",
        32: "tab:purple",
        64: "tab:pink",
    }

    def style_for_series(label: str) -> dict:
        base_variant = label.split("-", 1)[0]
        if base_variant == "serial":
            return {
                "color": "black",
                "linestyle": "--",
                "marker": "s",
                "linewidth": 2.6,
            }
        if base_variant == "cuda":
            return {
                "color": "tab:red",
                "linestyle": "-",
                "marker": "o",
                "linewidth": 2.6,
            }

        match = re.search(r"-t(\d+)$", label)
        threads = int(match.group(1)) if match else 0
        color = thread_colors.get(threads, "tab:blue")

        if base_variant == "omp":
            return {
                "color": color,
                "linestyle": "-",
                "marker": "o",
                "linewidth": 2.0,
            }
        if base_variant == "hybrid":
            return {
                "color": color,
                "linestyle": "--",
                "marker": "D",
                "linewidth": 2.0,
            }
        return {
            "color": color,
            "linestyle": "-.",
            "marker": "o",
            "linewidth": 2.0,
        }

    stats = {}
    for variant in variants:
        nx_vals = sorted(points[variant].keys())
        means = np.array([float(np.mean(points[variant][nx])) for nx in nx_vals], dtype=float)
        mins = np.array([float(np.min(points[variant][nx])) for nx in nx_vals], dtype=float)
        maxs = np.array([float(np.max(points[variant][nx])) for nx in nx_vals], dtype=float)
        stats[variant] = {
            "nx": np.array(nx_vals, dtype=float),
            "mean": means,
            "min": mins,
            "max": maxs,
        }

    fig, (ax_runtime, ax_speedup) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for variant in variants:
        st = stats[variant]
        style = style_for_series(variant)
        ax_runtime.plot(
            st["nx"],
            st["mean"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=variant,
        )
        ax_runtime.fill_between(
            st["nx"],
            st["min"],
            st["max"],
            color=style["color"],
            alpha=0.12,
        )

    ax_runtime.set_xscale("log", base=2)
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Wall time (s)")
    ax_runtime.set_title("Scaling Comparison by Version / Thread Count")
    ax_runtime.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_runtime.legend()

    if serial_key is not None:
        serial_means = {
            int(nx): float(mean)
            for nx, mean in zip(stats[serial_key]["nx"], stats[serial_key]["mean"])
        }
        for variant in variants:
            st = stats[variant]
            common_nx = [int(nx) for nx in st["nx"] if int(nx) in serial_means]
            if not common_nx:
                continue
            style = style_for_series(variant)
            speedups = []
            for nx in common_nx:
                variant_mean = float(np.mean(points[variant][nx]))
                speedups.append(serial_means[nx] / max(variant_mean, 1e-12))
            ax_speedup.plot(
                np.array(common_nx, dtype=float),
                np.array(speedups, dtype=float),
                marker=style["marker"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=variant,
            )
        ax_speedup.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax_speedup.set_ylabel("Speedup vs serial")
        ax_speedup.set_title("Relative Speedup vs serial (higher is better)")
        ax_speedup.grid(True, which="both", linestyle="--", alpha=0.4)
        ax_speedup.legend()
    else:
        ax_speedup.text(
            0.5,
            0.5,
            "Serial baseline not present in CSV.",
            transform=ax_speedup.transAxes,
            ha="center",
            va="center",
        )
        ax_speedup.set_axis_off()

    all_nx = sorted({int(nx) for variant in variants for nx in stats[variant]["nx"]})
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xlabel("Grid size nx (= ny)")
    ax_speedup.set_xticks(all_nx)
    ax_speedup.set_xticklabels([str(nx) for nx in all_nx])

    plt.tight_layout()

    if save_prefix:
        png_path = f"{save_prefix}.png"
        pdf_path = f"{save_prefix}.pdf"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {png_path} and {pdf_path}")

    plt.show()


def parse_metadata(metadata_path: str) -> dict:
    """Parse key=value metadata file into a dict."""
    data = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def discover_snapshots(outdir: str) -> dict:
    """
    Build mapping:
      snapshots[step][rank] = file_path
    """
    snapshots = defaultdict(dict)
    pattern = os.path.join(outdir, "rank_*_step_*.bin")
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = SNAPSHOT_RE.match(name)
        if not match:
            continue
        rank = int(match.group(1))
        step = int(match.group(2))
        snapshots[step][rank] = path
    return snapshots


def load_rank_snapshot(path: str, nx: int) -> np.ndarray:
    """Load one rank file and infer local_ny from file size."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % nx != 0:
        raise ValueError(f"Snapshot size is not divisible by nx: {path}")
    local_ny = raw.size // nx
    return raw.reshape((local_ny, nx))


def assemble_global_field(step_files: dict, nx: int, nranks: int) -> np.ndarray:
    """Stack rank slabs in rank order to reconstruct a full 2D field."""
    slabs = []
    for rank in range(nranks):
        if rank not in step_files:
            raise ValueError(f"Missing rank {rank} snapshot for selected step.")
        slabs.append(load_rank_snapshot(step_files[rank], nx))
    return np.vstack(slabs)


def load_fields_by_step(snapshots: dict, nx: int, nranks: int) -> tuple:
    """Load and cache all reconstructed fields keyed by simulation step."""
    steps = sorted(snapshots.keys())
    fields = {}
    vmin = np.inf
    vmax = -np.inf

    for step in steps:
        if len(snapshots[step]) < nranks:
            raise ValueError(f"Step {step} is missing rank files (expected {nranks}).")

        field = assemble_global_field(snapshots[step], nx, nranks)
        fields[step] = field
        vmin = min(vmin, float(np.min(field)))
        vmax = max(vmax, float(np.max(field)))

    return steps, fields, vmin, vmax


def show_heatmap_slider(outdir: str, cmap: str) -> None:
    """
    Interactive heatmap viewer:
    - Reads metadata and snapshot files from outdir
    - Reconstructs global field by stacking rank files
    - Uses a slider to move across saved simulation steps
    """
    metadata_path = os.path.join(outdir, "metadata.txt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    meta = parse_metadata(metadata_path)
    nx = int(meta.get("nx", "0"))
    ny = int(meta.get("ny", "0"))
    nranks = int(meta.get("nranks", "1"))
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid nx/ny in metadata.txt")

    snapshots = discover_snapshots(outdir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot files found in: {outdir}")

    steps, fields, vmin, vmax = load_fields_by_step(snapshots, nx, nranks)
    first = fields[steps[0]]
    if first.shape[0] != ny:
        print(
            f"Warning: reconstructed ny={first.shape[0]} differs from metadata ny={ny}. "
            "Using reconstructed size."
        )
        ny = first.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.20)

    image = ax.imshow(
        first,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Temperature (K)")
    title = ax.set_title(f"Heat distribution | step={steps[0]}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    slider_ax = fig.add_axes([0.15, 0.08, 0.70, 0.04])
    step_slider = Slider(
        ax=slider_ax,
        label="Snapshot index",
        valmin=0,
        valmax=len(steps) - 1,
        valinit=0,
        valstep=1,
    )

    def on_slider_change(val: float) -> None:
        idx = int(val)
        step = steps[idx]
        field = fields[step]
        image.set_data(field)
        title.set_text(f"Heat distribution | step={step}")
        fig.canvas.draw_idle()

    step_slider.on_changed(on_slider_change)
    plt.show()


def show_surface_slider(
    outdir: str,
    cmap: str,
    elev: float,
    azim: float,
    surface_stride: int,
    save_prefix: str,
) -> None:
    """
    Interactive 3D surface viewer inspired by p07 MATLAB plotting style:
    - Smooth shaded surface with fixed color limits across steps
    - Time-step slider
    - Camera controls via --elev/--azim and native matplotlib interaction
    """
    if surface_stride <= 0:
        raise ValueError("surface_stride must be >= 1")

    metadata_path = os.path.join(outdir, "metadata.txt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    meta = parse_metadata(metadata_path)
    nx = int(meta.get("nx", "0"))
    ny = int(meta.get("ny", "0"))
    nranks = int(meta.get("nranks", "1"))
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid nx/ny in metadata.txt")

    snapshots = discover_snapshots(outdir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot files found in: {outdir}")

    steps, fields, vmin, vmax = load_fields_by_step(snapshots, nx, nranks)
    first = fields[steps[0]]
    if first.shape[0] != ny:
        print(
            f"Warning: reconstructed ny={first.shape[0]} differs from metadata ny={ny}. "
            "Using reconstructed size."
        )
        ny = first.shape[0]

    y_idx = np.arange(0, ny, surface_stride)
    x_idx = np.arange(0, nx, surface_stride)
    xx, yy = np.meshgrid(x_idx, y_idx)

    sampled_fields = {
        step: fields[step][::surface_stride, ::surface_stride] for step in steps
    }

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.20)

    surface = ax.plot_surface(
        xx,
        yy,
        sampled_fields[steps[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
    )

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label("Temperature (K)")
    title = ax.set_title(f"Heat distribution surface | step={steps[0]}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_zlabel("Temperature (K)")

    x_range = max(1, nx - 1)
    y_range = max(1, ny - 1)
    z_range = max(vmax - vmin, 1e-12)
    target_visible_z = 0.35 * max(x_range, y_range)
    z_aspect = target_visible_z / z_range
    ax.set_box_aspect((x_range, y_range, z_range * z_aspect))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)

    slider_ax = fig.add_axes([0.15, 0.08, 0.70, 0.04])
    step_slider = Slider(
        ax=slider_ax,
        label="Snapshot index",
        valmin=0,
        valmax=len(steps) - 1,
        valinit=0,
        valstep=1,
    )

    def on_slider_change(val: float) -> None:
        nonlocal surface
        idx = int(val)
        step = steps[idx]
        surface.remove()
        surface = ax.plot_surface(
            xx,
            yy,
            sampled_fields[step],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
        )
        title.set_text(f"Heat distribution surface | step={step}")
        fig.canvas.draw_idle()

    step_slider.on_changed(on_slider_change)

    if save_prefix:
        png_path = f"{save_prefix}.png"
        pdf_path = f"{save_prefix}.pdf"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {png_path} and {pdf_path}")

    plt.show()


def resolve_step(steps: list, requested_step: int) -> int:
    """Resolve selected simulation step (-1 means latest available)."""
    if requested_step < 0:
        return steps[-1]
    if requested_step not in steps:
        raise ValueError(
            f"Requested step {requested_step} not found. "
            f"Available range: {steps[0]}..{steps[-1]}"
        )
    return requested_step


def export_snapshot_text(
    outdir: str,
    requested_step: int,
    export_format: str,
    export_path: str,
) -> None:
    """
    Export one reconstructed global field to plain text.

    Formats:
    - csv: comma-separated matrix (ny rows x nx columns)
    - dat/out: whitespace-separated matrix (ny rows x nx columns)
    """
    metadata_path = os.path.join(outdir, "metadata.txt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    meta = parse_metadata(metadata_path)
    nx = int(meta.get("nx", "0"))
    ny = int(meta.get("ny", "0"))
    nranks = int(meta.get("nranks", "1"))
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid nx/ny in metadata.txt")

    snapshots = discover_snapshots(outdir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshot files found in: {outdir}")

    steps, fields, _, _ = load_fields_by_step(snapshots, nx, nranks)
    step = resolve_step(steps, requested_step)
    field = fields[step]

    if not export_path:
        export_path = os.path.join(outdir, f"field_step_{step:06d}.{export_format}")

    if export_format == "csv":
        np.savetxt(export_path, field, delimiter=",", fmt="%.6f")
    elif export_format in ("dat", "out"):
        np.savetxt(export_path, field, delimiter=" ", fmt="%.6f")
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    print(f"Saved {export_path}")
    print(f"shape={field.shape[0]}x{field.shape[1]}, step={step}")


def resolve_heatmap_outdir(outdir: str, ask_outdir: bool) -> str:
    """
    Resolve a valid heatmap output folder.
    A valid folder must contain metadata.txt and rank snapshot files.
    """
    candidate = outdir

    while True:
        metadata_path = os.path.join(candidate, "metadata.txt")
        has_metadata = os.path.exists(metadata_path)
        has_snapshots = bool(glob.glob(os.path.join(candidate, "rank_*_step_*.bin")))
        if has_metadata and has_snapshots:
            return candidate

        should_prompt = ask_outdir or sys.stdin.isatty()
        if not should_prompt:
            return candidate

        print(
            f"Folder '{candidate}' is missing metadata.txt and/or snapshots.\n"
            "Enter a folder path with run outputs, or press Enter to keep current."
        )
        user_input = input(f"Outdir [{candidate}]: ").strip()
        if not user_input:
            return candidate
        candidate = user_input


def resolve_scaling_csv(csv_path: str, ask_csv: bool) -> str:
    """Resolve scaling CSV path, optionally prompting the user."""
    candidate = csv_path

    if ask_csv or not os.path.exists(candidate):
        if sys.stdin.isatty():
            print("Enter path to scaling CSV file.")
            user_input = input(f"CSV [{candidate}]: ").strip()
            if user_input:
                candidate = user_input

    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Project visualization helper")
    parser.add_argument(
        "--mode",
        default="scaling",
        choices=["scaling", "heatmap", "surface", "export"],
    )
    parser.add_argument("--csv", default="results/scaling.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument(
        "--no-scaling-plot",
        action="store_true",
        help="Print scaling summary only (do not open plotting figure).",
    )
    parser.add_argument(
        "--scaling-save-prefix",
        default="",
        help="If set, save scaling figure to <prefix>.png and <prefix>.pdf.",
    )
    parser.add_argument(
        "--show-scaling-table",
        action="store_true",
        help="Open a matplotlib UI table of sorted scaling metrics.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Snapshot step to export (-1 = latest available).",
    )
    parser.add_argument(
        "--export-format",
        default="csv",
        choices=["csv", "dat", "out"],
        help="Text format for --mode export.",
    )
    parser.add_argument(
        "--export-path",
        default="",
        help="Output path for --mode export (default: outdir/field_step_xxxxxx.ext).",
    )
    parser.add_argument("--elev", type=float, default=35.0, help="Surface plot elevation angle.")
    parser.add_argument("--azim", type=float, default=45.0, help="Surface plot azimuth angle.")
    parser.add_argument(
        "--surface-stride",
        type=int,
        default=8,
        help="Subsample stride for 3D surface plotting (1 = full resolution).",
    )
    parser.add_argument(
        "--save-prefix",
        default="",
        help="If set, save surface figure to <prefix>.png and <prefix>.pdf.",
    )
    parser.add_argument(
        "--ask-outdir",
        action="store_true",
        help="Prompt for heatmap output directory at runtime.",
    )
    parser.add_argument(
        "--ask-csv",
        action="store_true",
        help="Prompt for scaling CSV path at runtime.",
    )
    args = parser.parse_args()

    if args.mode == "scaling":
        csv_path = resolve_scaling_csv(args.csv, args.ask_csv)
        runs = load_scaling_runs(csv_path)
        summarize_scaling_runs(runs, csv_path)
        if args.show_scaling_table:
            show_scaling_ui_table(runs, "Scaling metrics table")
        if not args.no_scaling_plot:
            plot_scaling_comparison(runs, args.scaling_save_prefix)
    elif args.mode == "heatmap":
        outdir = resolve_heatmap_outdir(args.outdir, args.ask_outdir)
        show_heatmap_slider(outdir, args.cmap)
    elif args.mode == "surface":
        outdir = resolve_heatmap_outdir(args.outdir, args.ask_outdir)
        show_surface_slider(
            outdir=outdir,
            cmap=args.cmap,
            elev=args.elev,
            azim=args.azim,
            surface_stride=args.surface_stride,
            save_prefix=args.save_prefix,
        )
    elif args.mode == "export":
        outdir = resolve_heatmap_outdir(args.outdir, args.ask_outdir)
        export_snapshot_text(
            outdir=outdir,
            requested_step=args.step,
            export_format=args.export_format,
            export_path=args.export_path,
        )


if __name__ == "__main__":
    main()
