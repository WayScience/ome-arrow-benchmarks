"""OME-IRIS OME-Arrow nested byte leaf compression benchmark.

This benchmark focuses on the PR #77 nested byte table path and sweeps
leaf-level chunk compression against Parquet container compression. It measures
write-all, read-table, read-and-decode-to-NumPy, random decode, and file size
across the local OME-IRIS image bundles.
"""

from __future__ import annotations

import gc
import importlib.metadata as importlib_metadata
import importlib.resources as resources
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from ome_arrow import from_numpy, to_numpy
from ome_arrow.meta import OME_ARROW_BYTE_STRUCT

import OME_IRIS
from OME_IRIS.fetch import fetch_datasets

DATA_DIR = Path("data")
OME_IRIS_DATA_DIR = DATA_DIR / "ome_iris_leaf_compression"
BENCH_DIR = DATA_DIR / "ome_iris_leaf_compression_benchmark"
FIGURES_DIR = Path("figures")
SUMMARY_PARQUET = DATA_DIR / "compare_ome_iris_leaf_compression_summary.parquet"
RUNS_PARQUET = DATA_DIR / "compare_ome_iris_leaf_compression_runs.parquet"
PLOT_PATH = FIGURES_DIR / "compare_ome_iris_leaf_compression_summary.png"
OME_ARROW_VERSION = importlib_metadata.version("ome-arrow")
BENCHMARK_VERSION = f"ome_iris_leaf_compression_v1:{OME_ARROW_VERSION}"
TIMING_REPEATS = 5
RANDOM_IMAGE_COUNT = 3
SOURCE_NOTE = "Datasets: OME-IRIS catalog entries."

FIGURES_DIR.mkdir(exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)


COMPRESSION_CONFIGS: list[dict[str, Any]] = [
    {
        "label": "Leaf none / PQ zstd",
        "slug": "leaf_none_pq_zstd",
        "leaf_compression": None,
        "leaf_level": None,
        "parquet_compression": "zstd",
    },
    {
        "label": "Leaf none / PQ none",
        "slug": "leaf_none_pq_none",
        "leaf_compression": None,
        "leaf_level": None,
        "parquet_compression": None,
    },
    {
        "label": "Leaf zstd1 / PQ none",
        "slug": "leaf_zstd1_pq_none",
        "leaf_compression": "zstd",
        "leaf_level": 1,
        "parquet_compression": None,
    },
    {
        "label": "Leaf zstd1 / PQ zstd",
        "slug": "leaf_zstd1_pq_zstd",
        "leaf_compression": "zstd",
        "leaf_level": 1,
        "parquet_compression": "zstd",
    },
    {
        "label": "Leaf zstd3 / PQ none",
        "slug": "leaf_zstd3_pq_none",
        "leaf_compression": "zstd",
        "leaf_level": 3,
        "parquet_compression": None,
    },
    {
        "label": "Leaf zstd6 / PQ none",
        "slug": "leaf_zstd6_pq_none",
        "leaf_compression": "zstd",
        "leaf_level": 6,
        "parquet_compression": None,
    },
    {
        "label": "Leaf lz4 / PQ none",
        "slug": "leaf_lz4_pq_none",
        "leaf_compression": "lz4",
        "leaf_level": None,
        "parquet_compression": None,
    },
]


def cache_is_current() -> bool:
    if not SUMMARY_PARQUET.exists() or not RUNS_PARQUET.exists():
        return False
    try:
        summary = pd.read_parquet(SUMMARY_PARQUET, columns=["benchmark_version"])
    except Exception:
        return False
    return bool((summary["benchmark_version"] == BENCHMARK_VERSION).all())


RUN_BENCHMARKS = not cache_is_current()


def fetch_all_datasets() -> list[Path]:
    manifests_dir = resources.files(OME_IRIS) / "data" / "datasets"
    result = fetch_datasets(
        manifests_dir=Path(str(manifests_dir)),
        data_dir=OME_IRIS_DATA_DIR,
        silent=True,
    )
    if result.failed:
        raise RuntimeError(f"OME-IRIS fetch failed: {result.failed}")
    return sorted(path for path in OME_IRIS_DATA_DIR.iterdir() if path.is_dir())


def discover_images(dataset_dir: Path) -> list[Path]:
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        return []
    return [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.suffix.lower() in {".tif", ".tiff"}
        and "outlines" not in {part.lower() for part in path.parts}
    ]


def to_tczyx(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        return arr.reshape((1, 1, 1, *arr.shape))
    if arr.ndim == 3:
        return arr.reshape((1, 1, *arr.shape))
    if arr.ndim == 5:
        return arr
    raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def drop_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def build_nested_bytes_table(
    paths: list[Path], *, leaf_compression: str | None, leaf_level: int | None
) -> pa.Table:
    rows = []
    for path in paths:
        arr = to_tczyx(path)
        scalar = from_numpy(
            arr,
            dim_order="TCZYX",
            image_id=path.name,
            name=path.name,
            clamp_to_uint16=False,
            chunk_shape=(arr.shape[2], arr.shape[3], arr.shape[4]),
            chunk_encoding="bytes",
            chunk_compression=leaf_compression,
            chunk_compression_level=leaf_level,
        )
        rows.append({"image_key": path.name, "ome_image": scalar.as_py()})
    return pa.table(
        {
            "image_key": pa.array([row["image_key"] for row in rows]),
            "ome_image": pa.array(
                [row["ome_image"] for row in rows], type=OME_ARROW_BYTE_STRUCT
            ),
        }
    )


def write_nested_bytes(
    paths: list[Path],
    output_path: Path,
    *,
    leaf_compression: str | None,
    leaf_level: int | None,
    parquet_compression: str | None,
) -> None:
    table = build_nested_bytes_table(
        paths, leaf_compression=leaf_compression, leaf_level=leaf_level
    )
    pq.write_table(table, output_path, compression=parquet_compression)


def read_table(path: Path) -> pa.Table:
    return pq.read_table(path, columns=["image_key", "ome_image"])


def decode_table(path: Path, indices: list[int] | None = None) -> list[np.ndarray]:
    table = read_table(path)
    images = table.column("ome_image").combine_chunks()
    if indices is None:
        selected = range(len(images))
    else:
        selected = indices
    arrays = []
    for idx in selected:
        scalar = images[idx]
        record = scalar.as_py()
        chunks = record.get("chunks") or []
        dtype_name = chunks[0].get("dtype", "uint16") if chunks else "uint16"
        arrays.append(to_numpy(scalar, dtype=np.dtype(dtype_name)))
    return arrays


def timed(func) -> list[float]:
    times = []
    for _ in range(TIMING_REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return times


def benchmark_dataset(dataset_dir: Path, image_paths: list[Path]):
    dataset_slug = dataset_dir.name
    dataset_bench_dir = BENCH_DIR / dataset_slug
    dataset_bench_dir.mkdir(parents=True, exist_ok=True)
    random_indices = list(range(min(RANDOM_IMAGE_COUNT, len(image_paths))))
    summaries = []
    runs = []

    for cfg in COMPRESSION_CONFIGS:
        output_path = dataset_bench_dir / f"{cfg['slug']}.parquet"
        print(f"[compression start] {dataset_slug}: {cfg['label']}", flush=True)
        write_times = []
        for run_idx in range(TIMING_REPEATS):
            drop_path(output_path)
            gc.collect()
            t0 = time.perf_counter()
            write_nested_bytes(
                image_paths,
                output_path,
                leaf_compression=cfg["leaf_compression"],
                leaf_level=cfg["leaf_level"],
                parquet_compression=cfg["parquet_compression"],
            )
            seconds = time.perf_counter() - t0
            write_times.append(seconds)
            runs.append(
                {
                    "dataset": dataset_slug,
                    "label": cfg["label"],
                    "kind": "write",
                    "run": run_idx,
                    "seconds": seconds,
                }
            )
        size_mb = path_size_bytes(output_path) / (1024 * 1024)
        read_times = timed(lambda: read_table(output_path))
        decode_times = timed(lambda: decode_table(output_path))
        random_decode_times = timed(lambda: decode_table(output_path, random_indices))
        for kind, values in (
            ("read_table", read_times),
            ("decode_all", decode_times),
            ("decode_random", random_decode_times),
        ):
            for run_idx, seconds in enumerate(values):
                runs.append(
                    {
                        "dataset": dataset_slug,
                        "label": cfg["label"],
                        "kind": kind,
                        "run": run_idx,
                        "seconds": seconds,
                    }
                )

        summaries.append(
            {
                "dataset": dataset_slug,
                "label": cfg["label"],
                "benchmark_version": BENCHMARK_VERSION,
                "leaf_compression": cfg["leaf_compression"] or "none",
                "leaf_level": cfg["leaf_level"],
                "parquet_compression": cfg["parquet_compression"] or "none",
                "image_count": len(image_paths),
                "random_images": len(random_indices),
                "write_avg_s": float(np.mean(write_times)),
                "write_std_s": float(np.std(write_times)),
                "read_table_avg_s": float(np.mean(read_times)),
                "read_table_std_s": float(np.std(read_times)),
                "decode_all_avg_s": float(np.mean(decode_times)),
                "decode_all_std_s": float(np.std(decode_times)),
                "decode_random_avg_s": float(np.mean(random_decode_times)),
                "decode_random_std_s": float(np.std(random_decode_times)),
                "size_mb": float(size_mb),
                "version": (f"ome-arrow {OME_ARROW_VERSION}; pyarrow {pa.__version__}"),
            }
        )
        print(
            f"[compression end] {dataset_slug}: {cfg['label']} "
            f"write={np.mean(write_times):.4f}s "
            f"read={np.mean(read_times):.4f}s "
            f"decode={np.mean(decode_times):.4f}s "
            f"size={size_mb:.2f}MB",
            flush=True,
        )
    return summaries, runs


def run_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dirs = fetch_all_datasets()
    summaries = []
    runs = []
    for dataset_dir in dataset_dirs:
        image_paths = discover_images(dataset_dir)
        if not image_paths:
            continue
        dataset_summaries, dataset_runs = benchmark_dataset(dataset_dir, image_paths)
        summaries.extend(dataset_summaries)
        runs.extend(dataset_runs)
    summary_df = pd.DataFrame(summaries)
    runs_df = pd.DataFrame(runs)
    summary_df.to_parquet(SUMMARY_PARQUET, index=False)
    runs_df.to_parquet(RUNS_PARQUET, index=False)
    return summary_df, runs_df


def label_bars(ax, bars) -> None:
    axis_top = ax.get_ylim()[1]
    for bar in bars:
        value = bar.get_height()
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + axis_top * 0.01,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=8,
            rotation=90,
            clip_on=False,
        )


def plot_results(summary: pd.DataFrame) -> None:
    metrics = [
        ("write_avg_s", "Write all avg (s)"),
        ("read_table_avg_s", "Read table avg (s)"),
        ("decode_all_avg_s", "Decode all to NumPy avg (s)"),
        ("decode_random_avg_s", "Decode random avg (s)"),
        ("size_mb", "Size (MB)"),
    ]
    dataset_labels = {
        "CP_tutorial_3D_noise_nuclei_segmentation": "CP Tutorial 3D",
        "JUMP_plate_BR00117006": "JUMP",
        "NF1_cellpainting_data_shrunken": "NF1",
        "pediatric_cancer_atlas_profiling": "Ped Atlas",
    }
    labels = [cfg["label"] for cfg in COMPRESSION_CONFIGS]
    colors = {
        "Leaf none / PQ zstd": "#6F4E37",
        "Leaf none / PQ none": "#B23B3B",
        "Leaf zstd1 / PQ none": "#2E7D4F",
        "Leaf zstd1 / PQ zstd": "#3D7FBF",
        "Leaf zstd3 / PQ none": "#5A6B3A",
        "Leaf zstd6 / PQ none": "#8A5A9E",
        "Leaf lz4 / PQ none": "#C86A1B",
    }
    legend_labels = {
        "Leaf none / PQ zstd": "PQ zstd",
        "Leaf none / PQ none": "Raw",
        "Leaf zstd1 / PQ none": "Zstd1",
        "Leaf zstd1 / PQ zstd": "Zstd1+PQ",
        "Leaf zstd3 / PQ none": "Zstd3",
        "Leaf zstd6 / PQ none": "Zstd6",
        "Leaf lz4 / PQ none": "LZ4",
    }
    datasets = list(dict.fromkeys(summary["dataset"]))
    x = np.arange(len(datasets))
    width = 0.10

    plt.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 5 * len(metrics)))
    # All panels share the same subject: each compares the same 7 compression
    # configurations (bar colors) across the same 4 OME-IRIS datasets (x-axis groups).
    # The panels differ only in what is being measured on the y-axis.
    fig.suptitle(
        "OME-Arrow chunk-level and Parquet container compression benchmark\n"
        "Each panel shows the same 7 compression configurations (bar colors) × 4 OME-IRIS datasets (x-axis groups)\n"
        "measured on a different metric (y-axis)",
        fontsize=16,
    )
    for ax, (metric, title) in zip(axes, metrics):
        y_max = 0.0
        for offset, label in enumerate(labels):
            values = []
            for dataset in datasets:
                row = summary[
                    (summary["dataset"] == dataset) & (summary["label"] == label)
                ]
                values.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            y_max = max(y_max, max(values))
            bars = ax.bar(
                x + (offset - (len(labels) - 1) / 2) * width,
                values,
                width,
                label=legend_labels[label],
                color=colors[label],
            )
            if y_max > 0:
                ax.set_ylim(0, y_max * 1.18)
            label_bars(ax, bars)
        ax.set_title(title, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [dataset_labels.get(name, name) for name in datasets], fontsize=16
        )
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        fontsize=15,
        title="PQ = Parquet container compression; leaf = OME-Arrow chunk byte compression",
        title_fontsize=14,
    )
    fig.text(0.5, 0.0, SOURCE_NOTE, ha="center", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if RUN_BENCHMARKS:
    summary_df, runs_df = run_benchmarks()
else:
    summary_df = pd.read_parquet(SUMMARY_PARQUET)
    runs_df = pd.read_parquet(RUNS_PARQUET)

plot_results(summary_df)
print(summary_df)
