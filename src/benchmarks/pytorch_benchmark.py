"""PyTorch-focused benchmarking for image-based profiling file format

Benchmarks PyTorch Dataset and DataLoader performance with OME-Arrow format.
Measures: __getitem__ latency, DataLoader throughput, end-to-end model loop.

Requires `torch`, `torchvision`, `ome-arrow`, `pyarrow`, `lancedb` (installed via uv).

Setup:
- Artifacts are written under `data/` (git-ignored).
- Results are saved to `data/pytorch_benchmark_*.parquet`.
- Plots are saved to `images/pytorch_benchmark_*.png`.
"""

import gc
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import lancedb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ome_arrow import OMEArrow

pd.set_option("display.precision", 4)
PLOT_TITLE = f"{(__doc__ or 'PyTorch benchmark').splitlines()[0]} (lower is better)"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Benchmark configuration
N_ROWS = 1_000  # Number of images for benchmarking
OME_SHAPE = (100, 100)  # Image dimensions (H, W)
OME_DTYPE = np.uint8
SEED = 42

# Track 1: Dataset microbenchmark config
TRACK1_REPEATS = 3
TRACK1_WARMUP_SAMPLES = 10
TRACK1_BENCHMARK_SAMPLES = 100
TRACK1_RANDOM_ACCESS_COUNT = 50

# Track 2: DataLoader config
TRACK2_REPEATS = 3
TRACK2_NUM_BATCHES = 50
TRACK2_BATCH_SIZES = [1, 4, 16]
TRACK2_NUM_WORKERS_OPTIONS = [0, 2, 4]

# Track 3: End-to-end config
TRACK3_REPEATS = 3
TRACK3_STEPS = 50
TRACK3_BATCH_SIZE = 16

# Paths
LANCE_PATH = DATA_DIR / "pytorch_bench_lance"
PARQUET_PATH = DATA_DIR / "pytorch_bench.parquet"
RESULTS_TRACK1_PATH = DATA_DIR / "pytorch_benchmark_track1.parquet"
RESULTS_TRACK2_PATH = DATA_DIR / "pytorch_benchmark_track2.parquet"
RESULTS_TRACK3_PATH = DATA_DIR / "pytorch_benchmark_track3.parquet"
SUMMARY_PATH = DATA_DIR / "pytorch_benchmark_summary.json"

RUN_BENCHMARKS = not (
    RESULTS_TRACK1_PATH.exists()
    and RESULTS_TRACK2_PATH.exists()
    and RESULTS_TRACK3_PATH.exists()
)

# Resolve versions
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def _pkg_version(name: str, default: str = "missing") -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return default


VERSIONS = {
    "torch": torch.__version__,
    "numpy": np.__version__,
    "pyarrow": pa.__version__,
    "lancedb": getattr(lancedb, "__version__", "unknown"),
    "ome-arrow": getattr(__import__("ome_arrow"), "__version__", "unknown"),
}

HARDWARE_INFO = {
    "platform": platform.system(),
    "python_version": platform.python_version(),
    "cpu_count": torch.get_num_threads(),
}

print("PyTorch Benchmark Configuration")
print(f"N_ROWS: {N_ROWS}, OME_SHAPE: {OME_SHAPE}")
print(f"Versions: {VERSIONS}")
print(f"Hardware: {HARDWARE_INFO}")


def drop_path(path: Path) -> None:
    """Remove a file or directory."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


class OMEArrowDataset(Dataset):
    """PyTorch Dataset for OME-Arrow images stored in Parquet/Lance.
    
    Args:
        data_path: Path to Parquet file or Lance database
        format_type: "parquet" or "lance"
        transform: Optional torchvision transform
        access_pattern: "random" or "sequential" or "grouped"
        crop_size: Optional crop size (H, W)
    """

    def __init__(
        self,
        data_path: Path,
        format_type: str = "parquet",
        transform: Optional[Any] = None,
        access_pattern: str = "sequential",
        crop_size: Optional[Tuple[int, int]] = None,
    ):
        self.data_path = data_path
        self.format_type = format_type
        self.transform = transform
        self.access_pattern = access_pattern
        self.crop_size = crop_size

        # Load the table
        if format_type == "parquet":
            self.table = pq.read_table(data_path)
        elif format_type == "lance":
            db = lancedb.connect(data_path)
            self.table = db.open_table("bench").to_arrow()
        else:
            raise ValueError(f"Unknown format: {format_type}")

        self._length = self.table.num_rows

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get an image at the given index.
        
        Returns:
            torch.Tensor: Image tensor of shape (C, H, W) as float32
        """
        # Extract OME-Arrow struct from table
        row = self.table.slice(idx, 1)
        ome_struct = row["ome_image"][0].as_py()
        
        # Reconstruct numpy array from OME-Arrow chunks
        # OME-Arrow stores data in 'chunks' field with multiple chunks possible
        chunks = ome_struct.get("chunks", [])
        if not chunks:
            raise ValueError(f"No chunks found in OME-Arrow struct at index {idx}")
        
        # Reconstruct the full image from chunks
        # For now, we assume a single chunk (which is common for small images)
        # TODO: Handle multi-chunk images by stitching
        chunk = chunks[0]
        pixels = chunk["pixels"]
        shape_x = chunk["shape_x"]
        shape_y = chunk["shape_y"]
        shape_z = chunk["shape_z"]
        
        # Reconstruct the array (Z, Y, X)
        img_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
        
        # Get all chunks for all channels and time points
        # Organize by T, C to create TCZYX structure
        pixels_meta = ome_struct.get("pixels_meta", {})
        size_t = pixels_meta.get("size_t", 1)
        size_c = pixels_meta.get("size_c", 1)
        
        # For simplicity, if we have multiple chunks, collect them
        if len(chunks) > 1:
            # Build a full TCZYX array
            all_chunks = []
            for chunk in sorted(chunks, key=lambda c: (c['t'], c['c'])):
                pixels = chunk["pixels"]
                shape_x = chunk["shape_x"]
                shape_y = chunk["shape_y"]
                shape_z = chunk["shape_z"]
                chunk_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
                all_chunks.append(chunk_array)
            
            # Stack into (T, C, Z, Y, X)
            # This is a simplified approach; actual implementation may need more sophisticated stitching
            img_array = np.stack(all_chunks, axis=0)
            if size_t > 1 and size_c > 1:
                img_array = img_array.reshape(size_t, size_c, shape_z, shape_y, shape_x)
        
        # Squeeze singleton T and Z dimensions to get (C, H, W)
        # From (T=1, C, Z=1, Y, X) to (C, Y, X)
        while img_array.ndim > 3 and img_array.shape[0] == 1:
            img_array = img_array.squeeze(0)
        if img_array.ndim > 3 and img_array.shape[-3] == 1:  # Squeeze Z
            img_array = np.squeeze(img_array, axis=-3)
        
        # Ensure we have (C, H, W) format
        if img_array.ndim == 2:
            img_array = img_array[np.newaxis, :, :]  # Add channel dimension
        
        # Crop if requested
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            h, w = img_array.shape[-2:]
            if h >= crop_h and w >= crop_w:
                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                img_array = img_array[..., start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        # Normalize to [0, 1] range
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Apply transforms if any
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor


class SimpleEmbeddingModel(nn.Module):
    """Simple CNN for embedding extraction in Track 3."""

    def __init__(self, input_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def generate_test_data() -> pa.Table:
    """Generate synthetic OME-Arrow dataset for benchmarking."""
    rng = np.random.default_rng(SEED)
    row_ids = pa.array(np.arange(N_ROWS, dtype=np.int64))

    # Build OME-Arrow column with random images
    ome_pylist = []
    for _ in range(N_ROWS):
        img = rng.integers(0, 256, size=(1, 1, 1, *OME_SHAPE), dtype=OME_DTYPE)
        ome_scalar = OMEArrow(data=img).data.as_py()
        ome_scalar.pop("masks", None)  # drop Null field
        ome_pylist.append(ome_scalar)
    
    ome_column = pa.array(ome_pylist)
    table = pa.Table.from_arrays([row_ids, ome_column], names=["row_id", "ome_image"])
    return table


def write_test_data(table: pa.Table) -> None:
    """Write test data to both Parquet and Lance formats."""
    # Write Parquet
    drop_path(PARQUET_PATH)
    pq.write_table(table, PARQUET_PATH, compression="zstd")
    print(f"Wrote Parquet to {PARQUET_PATH}")

    # Write Lance
    drop_path(LANCE_PATH)
    db = lancedb.connect(LANCE_PATH)
    db.create_table("bench", table, mode="overwrite")
    print(f"Wrote Lance to {LANCE_PATH}")


def percentile_stats(values: List[float]) -> Dict[str, float]:
    """Calculate percentile statistics."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ============================================================================
# Track 1: Dataset / __getitem__ Microbenchmark
# ============================================================================

def benchmark_dataset_getitem(
    dataset: Dataset,
    access_pattern: str,
    num_samples: int,
    warmup: int = 10,
) -> Dict[str, Any]:
    """Benchmark Dataset.__getitem__ performance.
    
    Args:
        dataset: The dataset to benchmark
        access_pattern: "sequential" or "random"
        num_samples: Number of samples to benchmark
        warmup: Number of warmup samples
        
    Returns:
        Dict with latency statistics
    """
    indices = list(range(len(dataset)))
    if access_pattern == "random":
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(dataset), size=warmup + num_samples, replace=False).tolist()
    else:
        indices = indices[:warmup + num_samples]
    
    # Warmup
    for idx in indices[:warmup]:
        _ = dataset[idx]
    
    # Benchmark
    latencies = []
    for idx in indices[warmup:]:
        t0 = time.perf_counter()
        _ = dataset[idx]
        latencies.append(time.perf_counter() - t0)
    
    stats = percentile_stats(latencies)
    stats["samples_per_sec"] = 1.0 / stats["mean"] if stats["mean"] > 0 else 0
    stats["access_pattern"] = access_pattern
    stats["num_samples"] = num_samples
    
    return stats


def run_track1() -> pd.DataFrame:
    """Run Track 1: Dataset __getitem__ microbenchmark."""
    print("\n" + "=" * 80)
    print("Track 1: Dataset __getitem__ Microbenchmark")
    print("=" * 80)
    
    results = []
    
    for format_type in ["parquet", "lance"]:
        data_path = PARQUET_PATH if format_type == "parquet" else LANCE_PATH
        
        for access_pattern in ["sequential", "random"]:
            for transform_type in ["none", "normalize", "augment"]:
                # Define transforms
                if transform_type == "none":
                    transform = None
                elif transform_type == "normalize":
                    transform = transforms.Normalize(mean=[0.5], std=[0.5])
                else:  # augment
                    transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ])
                
                dataset = OMEArrowDataset(
                    data_path=data_path,
                    format_type=format_type,
                    transform=transform,
                    access_pattern=access_pattern,
                )
                
                print(f"\n[Track 1] format={format_type}, pattern={access_pattern}, transform={transform_type}")
                
                for run_idx in range(TRACK1_REPEATS):
                    gc.collect()
                    stats = benchmark_dataset_getitem(
                        dataset,
                        access_pattern,
                        TRACK1_BENCHMARK_SAMPLES,
                        warmup=TRACK1_WARMUP_SAMPLES,
                    )
                    
                    result = {
                        "format": format_type,
                        "access_pattern": access_pattern,
                        "transform": transform_type,
                        "run": run_idx,
                        **stats,
                    }
                    results.append(result)
                    
                    print(f"  Run {run_idx + 1}/{TRACK1_REPEATS}: "
                          f"p50={stats['p50']*1000:.2f}ms, "
                          f"p95={stats['p95']*1000:.2f}ms, "
                          f"p99={stats['p99']*1000:.2f}ms, "
                          f"samples/s={stats['samples_per_sec']:.1f}")
    
    return pd.DataFrame(results)


# ============================================================================
# Track 2: DataLoader Throughput
# ============================================================================

def benchmark_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Dict[str, Any]:
    """Benchmark DataLoader throughput.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        num_batches: Number of batches to benchmark
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        
    Returns:
        Dict with throughput statistics
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        shuffle=False,
    )
    
    batch_times = []
    first_batch_time = None
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        t0 = time.perf_counter()
        # Simulate minimal processing
        _ = batch.shape
        elapsed = time.perf_counter() - t0
        
        if batch_idx == 0:
            first_batch_time = elapsed
        else:
            batch_times.append(elapsed)
    
    if not batch_times:
        batch_times = [first_batch_time] if first_batch_time else [0.0]
    
    stats = percentile_stats(batch_times)
    stats["first_batch_time"] = first_batch_time or 0.0
    stats["samples_per_sec"] = batch_size / stats["mean"] if stats["mean"] > 0 else 0
    stats["batch_size"] = batch_size
    stats["num_workers"] = num_workers
    
    return stats


def run_track2() -> pd.DataFrame:
    """Run Track 2: DataLoader throughput benchmark."""
    print("\n" + "=" * 80)
    print("Track 2: DataLoader Throughput")
    print("=" * 80)
    
    results = []
    
    for format_type in ["parquet", "lance"]:
        data_path = PARQUET_PATH if format_type == "parquet" else LANCE_PATH
        
        for batch_size in TRACK2_BATCH_SIZES:
            for num_workers in TRACK2_NUM_WORKERS_OPTIONS:
                dataset = OMEArrowDataset(
                    data_path=data_path,
                    format_type=format_type,
                    transform=None,
                )
                
                print(f"\n[Track 2] format={format_type}, batch_size={batch_size}, num_workers={num_workers}")
                
                for run_idx in range(TRACK2_REPEATS):
                    gc.collect()
                    
                    stats = benchmark_dataloader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        num_batches=TRACK2_NUM_BATCHES,
                        pin_memory=False,
                        persistent_workers=False,
                    )
                    
                    result = {
                        "format": format_type,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "run": run_idx,
                        **stats,
                    }
                    results.append(result)
                    
                    print(f"  Run {run_idx + 1}/{TRACK2_REPEATS}: "
                          f"p50={stats['p50']*1000:.2f}ms/batch, "
                          f"first_batch={stats['first_batch_time']*1000:.2f}ms, "
                          f"samples/s={stats['samples_per_sec']:.1f}")
    
    return pd.DataFrame(results)


# ============================================================================
# Track 3: End-to-End Model Loop
# ============================================================================

def benchmark_model_loop(
    dataset: Dataset,
    model: nn.Module,
    batch_size: int,
    num_steps: int,
    num_workers: int = 2,
) -> Dict[str, Any]:
    """Benchmark end-to-end model training/inference loop.
    
    Args:
        dataset: Dataset to use
        model: Model to run
        batch_size: Batch size
        num_steps: Number of training steps
        num_workers: Number of DataLoader workers
        
    Returns:
        Dict with step time statistics
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=False,
    )
    
    step_times = []
    data_times = []
    model_times = []
    
    data_iter = iter(dataloader)
    for step_idx in range(num_steps):
        # Measure data loading time
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_time = time.perf_counter() - t0
        
        # Measure model forward time
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        model_time = time.perf_counter() - t0
        
        total_time = data_time + model_time
        
        step_times.append(total_time)
        data_times.append(data_time)
        model_times.append(model_time)
    
    step_stats = percentile_stats(step_times)
    data_stats = percentile_stats(data_times)
    
    return {
        "step_time_mean": step_stats["mean"],
        "step_time_p50": step_stats["p50"],
        "step_time_p95": step_stats["p95"],
        "data_wait_fraction": np.mean(data_times) / np.mean(step_times) if np.mean(step_times) > 0 else 0,
        "images_per_sec": batch_size / step_stats["mean"] if step_stats["mean"] > 0 else 0,
        "num_steps": num_steps,
    }


def run_track3() -> pd.DataFrame:
    """Run Track 3: End-to-end model loop benchmark."""
    print("\n" + "=" * 80)
    print("Track 3: End-to-End Model Loop")
    print("=" * 80)
    
    results = []
    
    # Create a simple model
    model = SimpleEmbeddingModel(input_channels=1, embedding_dim=128)
    model.eval()
    
    for format_type in ["parquet", "lance"]:
        data_path = PARQUET_PATH if format_type == "parquet" else LANCE_PATH
        
        dataset = OMEArrowDataset(
            data_path=data_path,
            format_type=format_type,
            transform=transforms.Normalize(mean=[0.5], std=[0.5]),
        )
        
        print(f"\n[Track 3] format={format_type}, batch_size={TRACK3_BATCH_SIZE}")
        
        for run_idx in range(TRACK3_REPEATS):
            gc.collect()
            
            stats = benchmark_model_loop(
                dataset,
                model,
                batch_size=TRACK3_BATCH_SIZE,
                num_steps=TRACK3_STEPS,
                num_workers=2,
            )
            
            result = {
                "format": format_type,
                "batch_size": TRACK3_BATCH_SIZE,
                "run": run_idx,
                **stats,
            }
            results.append(result)
            
            print(f"  Run {run_idx + 1}/{TRACK3_REPEATS}: "
                  f"step_time={stats['step_time_p50']*1000:.2f}ms, "
                  f"data_wait={stats['data_wait_fraction']*100:.1f}%, "
                  f"images/s={stats['images_per_sec']:.1f}")
    
    return pd.DataFrame(results)


# ============================================================================
# Plotting and Summary
# ============================================================================

def plot_track1_results(df: pd.DataFrame) -> None:
    """Plot Track 1 results: Dataset __getitem__ latency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Track 1: Dataset __getitem__ Latency (lower is better)")
    
    # Plot 1: p95 latency by format and access pattern
    ax = axes[0]
    summary = df.groupby(["format", "access_pattern", "transform"])["p95"].mean().reset_index()
    
    x_labels = []
    x_pos = []
    colors = []
    values = []
    
    color_map = {"parquet": "#2F5C8A", "lance": "#C86A1B"}
    
    pos = 0
    for _, row in summary.iterrows():
        label = f"{row['format']}\n{row['access_pattern']}\n{row['transform']}"
        x_labels.append(label)
        x_pos.append(pos)
        colors.append(color_map.get(row['format'], "#BAB0AC"))
        values.append(row['p95'] * 1000)  # Convert to ms
        pos += 1
    
    bars = ax.bar(x_pos, values, color=colors)
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("p95 __getitem__ Latency")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Plot 2: Samples per second
    ax = axes[1]
    summary = df.groupby(["format", "access_pattern", "transform"])["samples_per_sec"].mean().reset_index()
    
    values = summary["samples_per_sec"].tolist()
    bars = ax.bar(x_pos, values, color=colors)
    ax.set_ylabel("Samples per Second")
    ax.set_title("Throughput (higher is better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    fig.savefig(IMAGES_DIR / "pytorch_benchmark_track1.png", dpi=150)
    plt.close(fig)
    print(f"Saved Track 1 plot to {IMAGES_DIR / 'pytorch_benchmark_track1.png'}")


def plot_track2_results(df: pd.DataFrame) -> None:
    """Plot Track 2 results: DataLoader throughput."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Track 2: DataLoader Throughput")
    
    color_map = {"parquet": "#2F5C8A", "lance": "#C86A1B"}
    
    # Plot 1: Samples per second by num_workers and batch_size
    ax = axes[0]
    summary = df.groupby(["format", "num_workers", "batch_size"])["samples_per_sec"].mean().reset_index()
    
    for format_type in summary["format"].unique():
        format_data = summary[summary["format"] == format_type]
        for batch_size in sorted(format_data["batch_size"].unique()):
            batch_data = format_data[format_data["batch_size"] == batch_size]
            ax.plot(
                batch_data["num_workers"],
                batch_data["samples_per_sec"],
                marker="o",
                label=f"{format_type} bs={batch_size}",
                color=color_map.get(format_type, "#BAB0AC"),
                linestyle="-" if batch_size == min(TRACK2_BATCH_SIZES) else "--",
            )
    
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Samples per Second")
    ax.set_title("DataLoader Throughput (higher is better)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Plot 2: First batch latency
    ax = axes[1]
    summary = df.groupby(["format", "num_workers", "batch_size"])["first_batch_time"].mean().reset_index()
    
    x_labels = []
    x_pos = []
    colors = []
    values = []
    
    pos = 0
    for _, row in summary.iterrows():
        label = f"{row['format']}\nw={row['num_workers']}\nbs={row['batch_size']}"
        x_labels.append(label)
        x_pos.append(pos)
        colors.append(color_map.get(row['format'], "#BAB0AC"))
        values.append(row['first_batch_time'] * 1000)  # Convert to ms
        pos += 1
    
    bars = ax.bar(x_pos, values, color=colors)
    ax.set_ylabel("First Batch Time (ms)")
    ax.set_title("Worker Startup Overhead (lower is better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    fig.savefig(IMAGES_DIR / "pytorch_benchmark_track2.png", dpi=150)
    plt.close(fig)
    print(f"Saved Track 2 plot to {IMAGES_DIR / 'pytorch_benchmark_track2.png'}")


def plot_track3_results(df: pd.DataFrame) -> None:
    """Plot Track 3 results: End-to-end model loop."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Track 3: End-to-End Model Loop")
    
    color_map = {"parquet": "#2F5C8A", "lance": "#C86A1B"}
    
    # Plot 1: Step time p50
    ax = axes[0]
    summary = df.groupby("format")["step_time_p50"].mean().reset_index()
    
    bars = ax.bar(
        summary["format"],
        summary["step_time_p50"] * 1000,  # Convert to ms
        color=[color_map.get(f, "#BAB0AC") for f in summary["format"]],
    )
    ax.set_ylabel("Step Time p50 (ms)")
    ax.set_title("Step Latency (lower is better)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Plot 2: Data wait fraction
    ax = axes[1]
    summary = df.groupby("format")["data_wait_fraction"].mean().reset_index()
    
    bars = ax.bar(
        summary["format"],
        summary["data_wait_fraction"] * 100,  # Convert to percentage
        color=[color_map.get(f, "#BAB0AC") for f in summary["format"]],
    )
    ax.set_ylabel("Data Wait Fraction (%)")
    ax.set_title("Time Waiting on Data (lower is better)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    fig.savefig(IMAGES_DIR / "pytorch_benchmark_track3.png", dpi=150)
    plt.close(fig)
    print(f"Saved Track 3 plot to {IMAGES_DIR / 'pytorch_benchmark_track3.png'}")


def save_summary(
    track1_df: pd.DataFrame,
    track2_df: pd.DataFrame,
    track3_df: pd.DataFrame,
) -> None:
    """Save a JSON summary of all benchmark results."""
    
    # Convert grouped results to list of dicts for JSON serialization
    track1_summary = []
    for (fmt, pattern, transform), group in track1_df.groupby(["format", "access_pattern", "transform"]):
        track1_summary.append({
            "format": fmt,
            "access_pattern": pattern,
            "transform": transform,
            "p50_mean": float(group["p50"].mean()),
            "p95_mean": float(group["p95"].mean()),
            "p99_mean": float(group["p99"].mean()),
            "samples_per_sec_mean": float(group["samples_per_sec"].mean()),
        })
    
    track2_summary = []
    for (fmt, workers, batch), group in track2_df.groupby(["format", "num_workers", "batch_size"]):
        track2_summary.append({
            "format": fmt,
            "num_workers": int(workers),
            "batch_size": int(batch),
            "p50_mean": float(group["p50"].mean()),
            "samples_per_sec_mean": float(group["samples_per_sec"].mean()),
            "first_batch_time_mean": float(group["first_batch_time"].mean()),
        })
    
    track3_summary = []
    for fmt, group in track3_df.groupby("format"):
        track3_summary.append({
            "format": fmt,
            "step_time_p50_mean": float(group["step_time_p50"].mean()),
            "data_wait_fraction_mean": float(group["data_wait_fraction"].mean()),
            "images_per_sec_mean": float(group["images_per_sec"].mean()),
        })
    
    summary = {
        "metadata": {
            "versions": VERSIONS,
            "hardware": HARDWARE_INFO,
            "config": {
                "n_rows": N_ROWS,
                "ome_shape": list(OME_SHAPE),
                "seed": SEED,
            },
        },
        "track1": {
            "description": "Dataset __getitem__ microbenchmark",
            "results": track1_summary,
        },
        "track2": {
            "description": "DataLoader throughput",
            "results": track2_summary,
        },
        "track3": {
            "description": "End-to-end model loop",
            "results": track3_summary,
        },
    }
    
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to {SUMMARY_PATH}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if RUN_BENCHMARKS:
        print("Generating test data...")
        table = generate_test_data()
        write_test_data(table)
        
        # Run all tracks
        track1_df = run_track1()
        track1_df.to_parquet(RESULTS_TRACK1_PATH, index=False)
        print(f"\nTrack 1 results saved to {RESULTS_TRACK1_PATH}")
        
        track2_df = run_track2()
        track2_df.to_parquet(RESULTS_TRACK2_PATH, index=False)
        print(f"Track 2 results saved to {RESULTS_TRACK2_PATH}")
        
        track3_df = run_track3()
        track3_df.to_parquet(RESULTS_TRACK3_PATH, index=False)
        print(f"Track 3 results saved to {RESULTS_TRACK3_PATH}")
    else:
        print("Loading existing benchmark results...")
        track1_df = pd.read_parquet(RESULTS_TRACK1_PATH)
        track2_df = pd.read_parquet(RESULTS_TRACK2_PATH)
        track3_df = pd.read_parquet(RESULTS_TRACK3_PATH)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_track1_results(track1_df)
    plot_track2_results(track2_df)
    plot_track3_results(track3_df)
    
    # Save summary
    save_summary(track1_df, track2_df, track3_df)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print("\nTrack 1: Dataset __getitem__ (mean across runs)")
    print(track1_df.groupby(["format", "access_pattern", "transform"])[["p50", "p95", "samples_per_sec"]].mean())
    
    print("\nTrack 2: DataLoader Throughput (mean across runs)")
    print(track2_df.groupby(["format", "num_workers", "batch_size"])[["p50", "samples_per_sec", "first_batch_time"]].mean())
    
    print("\nTrack 3: End-to-End Model Loop (mean across runs)")
    print(track3_df.groupby("format")[["step_time_p50", "data_wait_fraction", "images_per_sec"]].mean())
    
    print("\nBenchmark complete!")
