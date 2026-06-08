"""OME-IRIS all-dataset image payload benchmark.

Fetch every dataset in the local OME-IRIS catalog, discover all TIFF images under
each dataset's image bundle, and benchmark simple full-image payload access
across OME-Arrow byte nested tables, OMEArrowDataset, OME-Zarr, and source TIFF.

This complements the NF1 join benchmark. It intentionally does not benchmark
profile joins because OME-IRIS datasets use different profile/image
relationship schemas.
"""

from __future__ import annotations

import gc
import importlib.metadata as importlib_metadata
import importlib.resources as resources
import shutil
import time
from pathlib import Path

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import vortex
import vortex.io as vxio
import zarr
from ome_arrow import OMEArrowDataset, from_numpy, write_ome_arrow_dataset
from ome_arrow.meta import OME_ARROW_BYTE_STRUCT
from ome_zarr.writer import write_image

import OME_IRIS
from OME_IRIS.fetch import fetch_datasets

DATA_DIR = Path("data")
OME_IRIS_DATA_DIR = DATA_DIR / "ome_iris_all"
BENCH_DIR = DATA_DIR / "ome_iris_all_images_benchmark"
FIGURES_DIR = Path("figures")
SUMMARY_PARQUET = DATA_DIR / "compare_ome_iris_all_images_summary.parquet"
RUNS_PARQUET = DATA_DIR / "compare_ome_iris_all_images_runs.parquet"
PLOT_PATH = FIGURES_DIR / "compare_ome_iris_all_images_summary.png"
OME_ARROW_VERSION = importlib_metadata.version("ome-arrow")
BENCHMARK_VERSION = f"ome_iris_all_images_v3:{OME_ARROW_VERSION}"
TIMING_REPEATS = 5
OA_KEY_TEXT = (
    "OA NT = OME-Arrow nested table; OA DS = OMEArrowDataset; "
    "Parquet/Vortex/Lance = table backend"
)
SOURCE_NOTE = "Datasets: OME-IRIS catalog entries backed by cytomining/CytoDataFrame test data (CC-BY-4.0)."

FIGURES_DIR.mkdir(exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)


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
    images = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.suffix.lower() in {".tif", ".tiff"}
        and "outlines" not in {part.lower() for part in path.parts}
    ]
    return images


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


def build_nested_bytes_table(paths: list[Path]) -> pa.Table:
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
        )
        rows.append({"image_key": path.name, "ome_image": scalar.as_py()})
    table = pa.table(
        {
            "image_key": pa.array([row["image_key"] for row in rows]),
            "ome_image": pa.array(
                [row["ome_image"] for row in rows], type=OME_ARROW_BYTE_STRUCT
            ),
        }
    )
    return table


def write_nested_bytes(paths: list[Path], output_path: Path) -> None:
    pq.write_table(build_nested_bytes_table(paths), output_path, compression="zstd")


def write_nested_vortex(paths: list[Path], output_path: Path) -> None:
    drop_path(output_path)
    vxio.write(build_nested_bytes_table(paths), str(output_path))


def write_nested_lance(paths: list[Path], output_path: Path) -> None:
    drop_path(output_path)
    db = lancedb.connect(output_path)
    db.create_table("images", build_nested_bytes_table(paths), mode="overwrite")


def write_oa_dataset(paths: list[Path], output_path: Path) -> None:
    arrays = [to_tczyx(path) for path in paths]
    write_ome_arrow_dataset(
        arrays,
        output_path,
        layout="auto",
        access_pattern="balanced",
        compression="zstd",
        build_physical_index=True,
        chunk_rows_per_row_group=1,
    )


def write_zarr_images(paths: list[Path], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for path in paths:
        arr = to_tczyx(path)
        group = zarr.group(
            store=zarr.storage.LocalStore(str(output_path / f"{path.stem}.zarr")),
            overwrite=True,
        )
        write_image(arr, group, axes="tczyx", scale_factors=[], method=None)


def read_nested_bytes(path: Path) -> pa.Table:
    return pq.read_table(path, columns=["image_key", "ome_image"])


def read_nested_vortex(path: Path) -> pa.Table:
    return (
        vortex.open(str(path)).to_arrow().read_all().select(["image_key", "ome_image"])
    )


def read_nested_lance(path: Path) -> pa.Table:
    db = lancedb.connect(path)
    return db.open_table("images").to_arrow().select(["image_key", "ome_image"])


def read_oa_dataset(path: Path):
    dataset = OMEArrowDataset(path)
    return dataset.read_many(dataset.image_ids)


def read_zarr_images(path: Path) -> list[np.ndarray]:
    return [
        zarr.open_group(str(store), mode="r")["s0"][...]
        for store in path.glob("*.zarr")
    ]


def read_tiff_images(paths: list[Path]) -> list[np.ndarray]:
    return [tifffile.imread(path) for path in paths]


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
    nested_path = dataset_bench_dir / "oa_nested_bytes.parquet"
    nested_vortex_path = dataset_bench_dir / "oa_nested_bytes.vortex"
    nested_lance_path = dataset_bench_dir / "oa_nested_bytes_lance"
    oa_dataset_path = dataset_bench_dir / "oa_dataset"
    zarr_path = dataset_bench_dir / "zarr"

    configs = [
        {
            "method": "OA NT Parquet",
            "path": nested_path,
            "write": lambda: write_nested_bytes(image_paths, nested_path),
            "read": lambda: read_nested_bytes(nested_path),
        },
        {
            "method": "OA NT Vortex",
            "path": nested_vortex_path,
            "write": lambda: write_nested_vortex(image_paths, nested_vortex_path),
            "read": lambda: read_nested_vortex(nested_vortex_path),
        },
        {
            "method": "OA NT Lance",
            "path": nested_lance_path,
            "write": lambda: write_nested_lance(image_paths, nested_lance_path),
            "read": lambda: read_nested_lance(nested_lance_path),
        },
        {
            "method": "OA DS Parquet",
            "path": oa_dataset_path,
            "write": lambda: write_oa_dataset(image_paths, oa_dataset_path),
            "read": lambda: read_oa_dataset(oa_dataset_path),
        },
        {
            "method": "Zarr",
            "path": zarr_path,
            "write": lambda: write_zarr_images(image_paths, zarr_path),
            "read": lambda: read_zarr_images(zarr_path),
        },
        {
            "method": "TIFF source",
            "path": None,
            "write": None,
            "read": lambda: read_tiff_images(image_paths),
        },
    ]

    runs = []
    summaries = []
    dataset_bench_dir.mkdir(parents=True, exist_ok=True)
    first_shape = "x".join(str(v) for v in to_tczyx(image_paths[0]).shape)
    for cfg in configs:
        if cfg["path"] is not None:
            write_times = []
            for run_idx in range(TIMING_REPEATS):
                drop_path(cfg["path"])
                seconds = timed(cfg["write"])[0]
                write_times.append(seconds)
                runs.append(
                    {
                        "dataset": dataset_slug,
                        "method": cfg["method"],
                        "kind": "write",
                        "run": run_idx,
                        "seconds": seconds,
                    }
                )
            size_mb = path_size_bytes(cfg["path"]) / (1024 * 1024)
        else:
            write_times = [0.0]
            size_mb = sum(path.stat().st_size for path in image_paths) / (1024 * 1024)

        read_times = timed(cfg["read"])
        for run_idx, seconds in enumerate(read_times):
            runs.append(
                {
                    "dataset": dataset_slug,
                    "method": cfg["method"],
                    "kind": "read_all",
                    "run": run_idx,
                    "seconds": seconds,
                }
            )
        summaries.append(
            {
                "dataset": dataset_slug,
                "method": cfg["method"],
                "benchmark_version": BENCHMARK_VERSION,
                "image_count": len(image_paths),
                "shape": f"TCZYX={first_shape}",
                "write_avg_s": float(np.mean(write_times)),
                "read_all_avg_s": float(np.mean(read_times)),
                "size_mb": float(size_mb),
                "version": f"ome-arrow {OME_ARROW_VERSION}",
            }
        )
        print(
            dataset_slug,
            cfg["method"],
            f"write={np.mean(write_times):.4f}s",
            f"read={np.mean(read_times):.4f}s",
            f"size={size_mb:.2f}MB",
            flush=True,
        )
    return summaries, runs


def run_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_summaries = []
    all_runs = []
    for dataset_dir in fetch_all_datasets():
        image_paths = discover_images(dataset_dir)
        if not image_paths:
            print(f"Skipping {dataset_dir.name}: no TIFF images found", flush=True)
            continue
        summaries, runs = benchmark_dataset(dataset_dir, image_paths)
        all_summaries.extend(summaries)
        all_runs.extend(runs)
    summary = pd.DataFrame(all_summaries)
    runs_df = pd.DataFrame(all_runs)
    summary.to_parquet(SUMMARY_PARQUET, index=False)
    runs_df.to_parquet(RUNS_PARQUET, index=False)
    return summary, runs_df


def label_bars(ax, bars) -> None:
    values = [bar.get_height() for bar in bars]
    if not values:
        return
    y_max = max(values)
    min_labeled_value = y_max * 0.04
    for bar, value in zip(bars, values):
        if value <= 0 or value < min_labeled_value:
            continue
        inset = max(value * 0.08, y_max * 0.015)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - inset,
            f"{value:.3f}" if value < 1 else f"{value:.2f}",
            ha="center",
            va="top",
            color="white",
            fontsize=6,
        )


def plot_results(summary: pd.DataFrame) -> None:
    metrics = [
        ("write_avg_s", "Write all avg (s)"),
        ("read_all_avg_s", "Read all avg (s)"),
        ("size_mb", "Size (MB)"),
    ]
    methods = [
        "OA NT Parquet",
        "OA NT Vortex",
        "OA NT Lance",
        "OA DS Parquet",
        "Zarr",
        "TIFF source",
    ]
    dataset_labels = {
        "CP_tutorial_3D_noise_nuclei_segmentation": "CP 3D",
        "JUMP_plate_BR00117006": "JUMP",
        "NF1_cellpainting_data_shrunken": "NF1",
        "pediatric_cancer_atlas_profiling": "Ped atlas",
    }
    colors = {
        "OA NT Parquet": "#6F4E37",
        "OA NT Vortex": "#2E7D4F",
        "OA NT Lance": "#8A5A9E",
        "OA DS Parquet": "#3D7FBF",
        "Zarr": "#6B7280",
        "TIFF source": "#C86A1B",
    }
    method_labels = {
        "OA NT Parquet": "OA NT Parquet",
        "OA NT Vortex": "OA NT Vortex",
        "OA NT Lance": "OA NT Lance",
        "OA DS Parquet": "OA DS Parquet",
        "Zarr": "OME-Zarr",
        "TIFF source": "TIFF source",
    }
    datasets = list(dict.fromkeys(summary["dataset"]))
    x = np.arange(len(datasets))
    width = 0.12

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    fig.suptitle("OME-IRIS all-dataset image payload benchmark")
    for ax, (metric, title) in zip(axes, metrics):
        y_max = 0.0
        for offset, method in enumerate(methods):
            values = []
            for dataset in datasets:
                row = summary[
                    (summary["dataset"] == dataset) & (summary["method"] == method)
                ]
                values.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            y_max = max(y_max, max(values))
            bars = ax.bar(
                x + (offset - (len(methods) - 1) / 2) * width,
                values,
                width,
                label=method_labels[method],
                color=colors[method],
            )
            label_bars(ax, bars)
        ax.set_title(title)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_labels.get(name, name) for name in datasets])
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(methods),
        title=OA_KEY_TEXT,
    )
    fig.text(0.5, 0.01, SOURCE_NOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.20, 1, 0.93))
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


if RUN_BENCHMARKS:
    summary_df, runs = run_benchmarks()
else:
    summary_df = pd.read_parquet(SUMMARY_PARQUET)
    runs = pd.read_parquet(RUNS_PARQUET)
    print("Loaded existing OME-IRIS all-image benchmark data")

print(summary_df)
print(runs)
plot_results(summary_df)
