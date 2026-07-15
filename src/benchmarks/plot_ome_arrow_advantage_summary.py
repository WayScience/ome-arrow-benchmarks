"""OME-Arrow benchmark advantage summary figure.

This script derives a concise summary figure from existing benchmark summary
tables. It does not rerun benchmarks; it normalizes the saved results so the
OME-Arrow advantages are visible at summary scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")
PLOT_PNG = FIGURES_DIR / "ome_arrow_advantage_summary.png"

ALL_IMAGES_SUMMARY = DATA_DIR / "compare_ome_iris_all_images_summary.parquet"
JOINS_SUMMARY = DATA_DIR / "compare_ome_iris_joins_summary.parquet"

DATASET_LABELS = {
    "CP_tutorial_3D_noise_nuclei_segmentation": "3D nuclei",
    "JUMP_plate_BR00117006": "JUMP",
    "NF1_cellpainting_data_shrunken": "NF1",
    "pediatric_cancer_atlas_profiling": "Ped. atlas",
}
DATASET_ORDER = tuple(DATASET_LABELS)

GREEN = "#1F7A4D"
BLUE = "#2F6FDB"
GOLD = "#B7791F"
BROWN = "#6F4E37"
GRAY = "#6B7280"
INK = "#111827"
GRID = "#D1D5DB"
SHADE = "#E5E7EB"


def benchmark_value(
    summary: pd.DataFrame, dataset: str, method: str, column: str
) -> float:
    """Return one scalar benchmark value from the all-image summary table."""
    rows = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
    if rows.empty:
        raise KeyError(f"Missing {method!r} for {dataset!r} in {column!r}")
    return float(rows.iloc[0][column])


def operation_row(summary: pd.DataFrame, operation: str) -> pd.Series:
    """Return one operation row from the join benchmark summary table."""
    rows = summary[summary["operation"] == operation]
    if rows.empty:
        raise KeyError(f"Missing operation {operation!r}")
    return rows.iloc[0]


def style_axis(ax: plt.Axes) -> None:
    """Apply common visual styling for summary figure panels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.18) -> None:
    """Place panel labels outside the plotting area to avoid overlap."""
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=INK,
        va="top",
        clip_on=False,
    )


def labeled_bars(
    ax: plt.Axes,
    values: list[float],
    color: str,
    fmt: str,
    y_offset: float,
) -> None:
    """Draw vertical bars with bold value labels."""
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=color, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontweight="bold",
            color=INK,
        )


def plot_advantage(summary: pd.DataFrame, joins: pd.DataFrame) -> None:
    """Build and save the benchmark advantage summary figure."""
    labels = [DATASET_LABELS[dataset] for dataset in DATASET_ORDER]

    read_speedup = [
        benchmark_value(summary, dataset, "Zarr", "read_all_avg_s")
        / benchmark_value(summary, dataset, "OA NT Vortex", "read_all_avg_s")
        for dataset in DATASET_ORDER
    ]
    write_speedup = [
        benchmark_value(summary, dataset, "Zarr", "write_avg_s")
        / benchmark_value(summary, dataset, "OA NT Parquet", "write_avg_s")
        for dataset in DATASET_ORDER
    ]
    storage_ratio = [
        benchmark_value(summary, dataset, "OA NT Parquet", "size_mb")
        / benchmark_value(summary, dataset, "Zarr", "size_mb")
        for dataset in DATASET_ORDER
    ]

    path_join = operation_row(joins, "Path Parquet join, observation scale")
    oa_join = operation_row(joins, "OA NT Parquet metadata join, observation scale")
    converted_oa_meta = operation_row(joins, "Converted OA NT Parquet metadata scan")
    tiff_meta = operation_row(joins, "OME-TIFF metadata scan")
    zarr_meta = operation_row(joins, "OME-Zarr metadata scan")

    join_ms = [path_join["avg_seconds"] * 1000, oa_join["avg_seconds"] * 1000]
    scan_ms = [
        converted_oa_meta["avg_seconds"] * 1000,
        tiff_meta["avg_seconds"] * 1000,
        zarr_meta["avg_seconds"] * 1000,
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig = plt.figure(figsize=(14, 8.6), facecolor="white")
    grid = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.48)

    ax_read = fig.add_subplot(grid[0, 0])
    ax_write = fig.add_subplot(grid[0, 1])
    ax_storage = fig.add_subplot(grid[0, 2])
    ax_join = fig.add_subplot(grid[1, 0:2])
    ax_scan = fig.add_subplot(grid[1, 2])

    x = np.arange(len(labels))

    labeled_bars(
        ax_read,
        read_speedup,
        GREEN,
        "{:.1f}x",
        max(read_speedup) * 0.035,
    )
    ax_read.axhline(1, color=GRAY, lw=1.2, ls="--")
    ax_read.set_xticks(x, labels)
    ax_read.set_ylabel("OME-Arrow speedup vs. OME-Zarr")
    ax_read.set_title("Bulk image reads")
    ax_read.set_ylim(0, max(read_speedup) * 1.23)
    panel_label(ax_read, "A")
    style_axis(ax_read)

    labeled_bars(
        ax_write,
        write_speedup,
        BLUE,
        "{:.1f}x",
        max(write_speedup) * 0.04,
    )
    ax_write.axhline(1, color=GRAY, lw=1.2, ls="--")
    ax_write.set_xticks(x, labels)
    ax_write.set_ylabel("OME-Arrow speedup vs. OME-Zarr")
    ax_write.set_title("Image writes")
    ax_write.set_ylim(0, max(write_speedup) * 1.30)
    panel_label(ax_write, "B")
    style_axis(ax_write)

    labeled_bars(
        ax_storage,
        storage_ratio,
        GOLD,
        "{:.2f}x",
        max(storage_ratio) * 0.04,
    )
    ax_storage.axhline(1, color=GRAY, lw=1.2, ls="--")
    ax_storage.fill_between(
        [-0.5, len(labels) - 0.5],
        0.8,
        1.3,
        color=SHADE,
        alpha=0.45,
        zorder=0,
    )
    ax_storage.set_xticks(x, labels)
    ax_storage.set_ylabel("storage / OME-Zarr")
    ax_storage.set_title("Storage footprint")
    ax_storage.set_ylim(0, max(storage_ratio) * 1.35)
    panel_label(ax_storage, "C")
    style_axis(ax_storage)

    join_labels = [
        "Parquet path\njoin",
        "OME-Arrow\nmetadata join",
        "OME-Zarr native\nprofile join",
    ]
    join_colors = [GRAY, BROWN]
    join_x = np.arange(len(join_labels))
    bars = ax_join.bar(join_x[:2], join_ms, color=join_colors, width=0.48)
    for bar, value in zip(bars, join_ms, strict=True):
        ax_join.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(join_ms) * 0.07,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=INK,
        )
    ax_join.text(
        join_x[2],
        max(join_ms) * 0.28,
        "not native",
        ha="center",
        va="center",
        color=GRAY,
        fontweight="bold",
        fontsize=13,
    )
    ax_join.set_xticks(join_x, join_labels)
    ax_join.set_xlim(-0.45, len(join_labels) - 0.55)
    ax_join.set_ylabel("milliseconds")
    ax_join.set_title("Profile-to-image metadata joins")
    ax_join.set_ylim(0, max(join_ms) * 1.38)
    panel_label(ax_join, "D", x=-0.07)
    style_axis(ax_join)

    scan_labels = ["OME-\nArrow", "OME-\nTIFF", "OME-\nZarr"]
    scan_colors = [BROWN, GOLD, GRAY]
    scan_x = np.arange(len(scan_labels))
    bars = ax_scan.bar(scan_x, scan_ms, color=scan_colors, width=0.56)
    for bar, value in zip(bars, scan_ms, strict=True):
        ax_scan.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(scan_ms) * 0.05,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=INK,
        )
    ax_scan.set_xticks(scan_x, scan_labels)
    ax_scan.set_ylabel("milliseconds")
    ax_scan.set_title("Metadata scans")
    ax_scan.set_ylim(0, max(scan_ms) * 1.35)
    panel_label(ax_scan, "E")
    style_axis(ax_scan)

    fig.suptitle(
        "OME-Arrow benchmark advantage summary",
        fontsize=19,
        fontweight="bold",
        y=0.97,
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.07, right=0.98)
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load saved benchmark summaries and regenerate the summary figure."""
    all_images = pd.read_parquet(ALL_IMAGES_SUMMARY)
    joins = pd.read_parquet(JOINS_SUMMARY)
    plot_advantage(all_images, joins)
    print(f"Wrote {PLOT_PNG}")


if __name__ == "__main__":
    main()
