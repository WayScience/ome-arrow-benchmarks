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
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ome_arrow import OMEArrow

# Optional imports
try:
    import vortex
    import vortex.io as vxio
    VORTEX_AVAILABLE = True
except ImportError:
    VORTEX_AVAILABLE = False
    print("Warning: vortex not available, skipping Vortex format")

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
PARQUET_DUCK_PATH = DATA_DIR / "pytorch_bench_duck.parquet"
VORTEX_PATH = DATA_DIR / "pytorch_bench.vortex"
DUCK_PATH = DATA_DIR / "pytorch_bench.duckdb"
TIFF_DIR = DATA_DIR / "pytorch_bench_tiff"
OME_ZARR_DIR = DATA_DIR / "pytorch_bench_ome_zarr"
LANCE_TABLE = "bench"
DUCK_TABLE = "bench"

RESULTS_TRACK1_PATH = DATA_DIR / "pytorch_benchmark_track1.parquet"
RESULTS_TRACK2_PATH = DATA_DIR / "pytorch_benchmark_track2.parquet"
RESULTS_TRACK3_PATH = DATA_DIR / "pytorch_benchmark_track3.parquet"
SUMMARY_PATH = DATA_DIR / "pytorch_benchmark_summary.json"

# Check if we need to run benchmarks (results don't exist)
RUN_BENCHMARKS = not (
    RESULTS_TRACK1_PATH.exists()
    and RESULTS_TRACK2_PATH.exists()
    and RESULTS_TRACK3_PATH.exists()
)

# Check if we need to generate test data (any format data is missing)
GENERATE_DATA = not (
    PARQUET_PATH.exists()
    and LANCE_PATH.exists()
    and PARQUET_DUCK_PATH.exists()
    and DUCK_PATH.exists()
)
# Also regenerate if TIFF or OME-Zarr directories are missing
try:
    import tifffile
    if not TIFF_DIR.exists() or not any(TIFF_DIR.glob("*.tiff")):
        GENERATE_DATA = True
except ImportError:
    pass

try:
    import zarr
    from ome_zarr.io import parse_url
    if not OME_ZARR_DIR.exists() or not any(OME_ZARR_DIR.glob("*.zarr")):
        GENERATE_DATA = True
except ImportError:
    pass

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
    "duckdb": duckdb.__version__,
    "vortex": getattr(vortex, "__version__", "unavailable") if VORTEX_AVAILABLE else "unavailable",
    "ome-arrow": getattr(__import__("ome_arrow"), "__version__", "unknown"),
    "tifffile": _pkg_version("tifffile"),
    "ome-zarr": _pkg_version("ome-zarr"),
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
    """PyTorch Dataset for OME-Arrow images stored in various formats.
    
    Args:
        data_path: Path to data file/directory
        format_type: "parquet", "parquet_duck", "lance", "vortex", "duckdb", "tiff", or "ome_zarr"
        transform: Optional torchvision transform
        access_pattern: "random" or "sequential" or "grouped"
        crop_size: Optional crop size (H, W)
        lance_table: Table name for Lance format (default: "bench")
        duck_table: Table name for DuckDB format (default: "bench")
    """

    def __init__(
        self,
        data_path: Path,
        format_type: str = "parquet",
        transform: Optional[Any] = None,
        access_pattern: str = "sequential",
        crop_size: Optional[Tuple[int, int]] = None,
        lance_table: str = "bench",
        duck_table: str = "bench",
    ):
        self.data_path = data_path
        self.format_type = format_type
        self.transform = transform
        self.access_pattern = access_pattern
        self.crop_size = crop_size
        self.lance_table = lance_table
        self.duck_table = duck_table
        self.table = None
        self.pixel_cache = None  # Cache for pre-extracted pixels

        # Load the table or prepare for directory-based formats
        if format_type == "parquet":
            self.table = pq.read_table(data_path)
            self._length = self.table.num_rows
            # Pre-extract pixel data for better DataLoader performance
            self._preload_pixels()
        elif format_type == "parquet_duck":
            with duckdb.connect() as con:
                self.table = con.execute(f"SELECT * FROM read_parquet('{data_path}')").fetch_arrow_table()
            self._length = self.table.num_rows
            self._preload_pixels()
        elif format_type == "lance":
            db = lancedb.connect(data_path)
            self.table = db.open_table(lance_table).to_arrow()
            self._length = self.table.num_rows
            self._preload_pixels()
        elif format_type == "vortex":
            if not VORTEX_AVAILABLE:
                raise RuntimeError("Vortex format requires vortex-data package")
            self.table = vortex.open(str(data_path)).to_arrow().read_all()
            self._length = self.table.num_rows
            self._preload_pixels()
        elif format_type == "duckdb":
            with duckdb.connect(str(data_path)) as con:
                self.table = con.execute(f"SELECT * FROM {duck_table}").fetch_arrow_table()
            self._length = self.table.num_rows
            self._preload_pixels()
        elif format_type == "tiff":
            # Directory-based TIFF format
            import tifffile
            self.image_paths = sorted(data_path.glob("*.tiff"))
            self._length = len(self.image_paths)
        elif format_type == "ome_zarr":
            # Directory-based OME-Zarr format
            import zarr
            self.zarr_paths = sorted(data_path.glob("*.zarr"))
            self._length = len(self.zarr_paths)
        else:
            raise ValueError(f"Unknown format: {format_type}")

    def _preload_pixels(self):
        """Pre-extract pixel data from OME-Arrow structures for faster access.
        
        This converts Arrow data to numpy arrays once during initialization,
        avoiding expensive Python conversions on every __getitem__ call.
        """
        if self.table is None:
            return
        
        print(f"Pre-loading {self._length} images for faster DataLoader access...")
        self.pixel_cache = []
        
        for i in range(self._length):
            # Direct column access is faster than slicing
            ome_struct = self.table["ome_image"][i].as_py()
            
            # Extract pixel data from chunks
            chunks = ome_struct.get("chunks", [])
            if not chunks:
                raise ValueError(f"No chunks found in OME-Arrow struct at index {i}")
            
            # Currently only single-chunk images are supported
            chunk = chunks[0]
            pixels = chunk["pixels"]
            shape_x = chunk["shape_x"]
            shape_y = chunk["shape_y"]
            shape_z = chunk["shape_z"]
            
            # Reconstruct the array (Z, Y, X) and store as numpy
            img_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
            
            # Handle multi-chunk if needed
            if len(chunks) > 1:
                pixels_meta = ome_struct.get("pixels_meta", {})
                size_t = pixels_meta.get("size_t", 1)
                size_c = pixels_meta.get("size_c", 1)
                
                all_chunks = []
                for chunk in sorted(chunks, key=lambda c: (c['t'], c['c'])):
                    pixels = chunk["pixels"]
                    shape_x = chunk["shape_x"]
                    shape_y = chunk["shape_y"]
                    shape_z = chunk["shape_z"]
                    chunk_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
                    all_chunks.append(chunk_array)
                
                img_array = np.stack(all_chunks, axis=0)
                if size_t > 1 and size_c > 1:
                    img_array = img_array.reshape(size_t, size_c, shape_z, shape_y, shape_x)
            
            # Squeeze singleton T and Z dimensions
            while img_array.ndim > 3 and img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            if img_array.ndim > 3 and img_array.shape[-3] == 1:
                img_array = np.squeeze(img_array, axis=-3)
            
            # Ensure (C, H, W) format
            if img_array.ndim == 2:
                img_array = img_array[np.newaxis, :, :]
            
            self.pixel_cache.append(img_array)
        
        print(f"Pre-loading complete. Images cached in memory.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get an image at the given index.
        
        Returns:
            torch.Tensor: Image tensor of shape (C, H, W) as float32
        """
        # Handle directory-based formats (TIFF, OME-Zarr)
        if self.format_type == "tiff":
            import tifffile
            img_array = tifffile.imread(str(self.image_paths[idx]))
            # TIFF typically stores as (Z, Y, X) or (T, C, Z, Y, X)
            # Squeeze singleton dimensions
            while img_array.ndim > 3 and img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            if img_array.ndim > 3 and img_array.shape[-3] == 1:
                img_array = np.squeeze(img_array, axis=-3)
            if img_array.ndim == 2:
                img_array = img_array[np.newaxis, :, :]
        elif self.format_type == "ome_zarr":
            import zarr
            zarr_dir = self.zarr_paths[idx]
            grp = zarr.open_group(str(zarr_dir), mode="r")
            if "0" in grp:
                img_array = grp["0"][...]
            else:
                raise ValueError(f"No '0' group found in OME-Zarr at {zarr_dir}")
            # Squeeze singleton dimensions
            while img_array.ndim > 3 and img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            if img_array.ndim > 3 and img_array.shape[-3] == 1:
                img_array = np.squeeze(img_array, axis=-3)
            if img_array.ndim == 2:
                img_array = img_array[np.newaxis, :, :]
        else:
            # Table-based formats - use pre-loaded pixel cache
            if self.pixel_cache is not None:
                # Fast path: use pre-loaded numpy arrays
                img_array = self.pixel_cache[idx].copy()  # Copy to avoid mutation
            else:
                # Fallback: extract from Arrow table (slower)
                # This shouldn't happen if _preload_pixels() was called
                ome_struct = self.table["ome_image"][idx].as_py()
                
                chunks = ome_struct.get("chunks", [])
                if not chunks:
                    raise ValueError(f"No chunks found in OME-Arrow struct at index {idx}")
                
                chunk = chunks[0]
                pixels = chunk["pixels"]
                shape_x = chunk["shape_x"]
                shape_y = chunk["shape_y"]
                shape_z = chunk["shape_z"]
                
                img_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
                
                # Handle multi-chunk
                if len(chunks) > 1:
                    pixels_meta = ome_struct.get("pixels_meta", {})
                    size_t = pixels_meta.get("size_t", 1)
                    size_c = pixels_meta.get("size_c", 1)
                    
                    all_chunks = []
                    for chunk in sorted(chunks, key=lambda c: (c['t'], c['c'])):
                        pixels = chunk["pixels"]
                        shape_x = chunk["shape_x"]
                        shape_y = chunk["shape_y"]
                        shape_z = chunk["shape_z"]
                        chunk_array = np.array(pixels, dtype=np.uint8).reshape(shape_z, shape_y, shape_x)
                        all_chunks.append(chunk_array)
                    
                    img_array = np.stack(all_chunks, axis=0)
                    if size_t > 1 and size_c > 1:
                        img_array = img_array.reshape(size_t, size_c, shape_z, shape_y, shape_x)
                
                # Squeeze singleton dimensions
                while img_array.ndim > 3 and img_array.shape[0] == 1:
                    img_array = img_array.squeeze(0)
                if img_array.ndim > 3 and img_array.shape[-3] == 1:
                    img_array = np.squeeze(img_array, axis=-3)
                
                # Ensure (C, H, W) format
                if img_array.ndim == 2:
                    img_array = img_array[np.newaxis, :, :]
        
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


def generate_test_data() -> Tuple[pa.Table, List[np.ndarray]]:
    """Generate synthetic OME-Arrow dataset for benchmarking.
    
    Returns:
        Tuple of (Arrow table, list of numpy arrays for directory-based formats)
    """
    rng = np.random.default_rng(SEED)
    row_ids = pa.array(np.arange(N_ROWS, dtype=np.int64))

    # Build OME-Arrow column with random images
    ome_pylist = []
    ome_arrays = []
    for _ in range(N_ROWS):
        img = rng.integers(0, 256, size=(1, 1, 1, *OME_SHAPE), dtype=OME_DTYPE)
        ome_arrays.append(img)
        ome_scalar = OMEArrow(data=img).data.as_py()
        ome_scalar.pop("masks", None)  # drop Null field
        ome_pylist.append(ome_scalar)
    
    ome_column = pa.array(ome_pylist)
    table = pa.Table.from_arrays([row_ids, ome_column], names=["row_id", "ome_image"])
    return table, ome_arrays


def write_test_data(table: pa.Table, ome_arrays: List[np.ndarray]) -> None:
    """Write test data to all supported formats."""
    # Write Parquet (pyarrow)
    drop_path(PARQUET_PATH)
    pq.write_table(table, PARQUET_PATH, compression="zstd")
    print(f"Wrote Parquet to {PARQUET_PATH}")

    # Write Parquet (duckdb)
    drop_path(PARQUET_DUCK_PATH)
    con = duckdb.connect()
    con.register("tmp_tbl", table)
    con.execute(
        f"COPY (SELECT * FROM tmp_tbl) TO '{PARQUET_DUCK_PATH}' WITH (FORMAT 'PARQUET', COMPRESSION 'ZSTD')"
    )
    con.close()
    print(f"Wrote Parquet (DuckDB) to {PARQUET_DUCK_PATH}")

    # Write Lance
    drop_path(LANCE_PATH)
    db = lancedb.connect(LANCE_PATH)
    db.create_table(LANCE_TABLE, table, mode="overwrite")
    print(f"Wrote Lance to {LANCE_PATH}")

    # Write Vortex
    if VORTEX_AVAILABLE:
        drop_path(VORTEX_PATH)
        vxio.write(table, str(VORTEX_PATH))
        print(f"Wrote Vortex to {VORTEX_PATH}")
    else:
        print("Skipping Vortex (not available)")

    # Write DuckDB
    drop_path(DUCK_PATH)
    con = duckdb.connect(str(DUCK_PATH))
    con.register("tmp_tbl", table)
    con.execute(f"CREATE TABLE {DUCK_TABLE} AS SELECT * FROM tmp_tbl")
    con.close()
    print(f"Wrote DuckDB to {DUCK_PATH}")

    # Write TIFF (directory-based)
    try:
        import tifffile
        drop_path(TIFF_DIR)
        TIFF_DIR.mkdir(parents=True, exist_ok=True)
        for idx, arr in enumerate(ome_arrays):
            out_path = TIFF_DIR / f"img_{idx:05d}.tiff"
            tifffile.imwrite(str(out_path), arr)
        print(f"Wrote TIFF to {TIFF_DIR}")
    except ImportError:
        print("Warning: tifffile not available, skipping TIFF format")

    # Write OME-Zarr (directory-based)
    try:
        from ome_zarr.io import parse_url
        from ome_zarr.writer import write_image
        import zarr
        
        drop_path(OME_ZARR_DIR)
        OME_ZARR_DIR.mkdir(parents=True, exist_ok=True)
        for idx, arr in enumerate(ome_arrays):
            out_dir = OME_ZARR_DIR / f"img_{idx:05d}.zarr"
            store = parse_url(str(out_dir), mode="w").store
            root = zarr.group(store=store)
            write_image(arr, root, axes="tczyx")
        print(f"Wrote OME-Zarr to {OME_ZARR_DIR}")
    except ImportError:
        print("Warning: ome-zarr not available, skipping OME-Zarr format")


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


def get_available_formats() -> List[Dict[str, Any]]:
    """Get list of available formats with their configurations.
    
    Returns:
        List of format configurations
    """
    formats = [
        {"name": "Parquet", "type": "parquet", "path": PARQUET_PATH},
        {"name": "Parquet (DuckDB)", "type": "parquet_duck", "path": PARQUET_DUCK_PATH},
        {"name": "Lance", "type": "lance", "path": LANCE_PATH},
        {"name": "DuckDB", "type": "duckdb", "path": DUCK_PATH},
    ]
    
    # Add Vortex if available
    if VORTEX_AVAILABLE and VORTEX_PATH.exists():
        formats.append({"name": "Vortex", "type": "vortex", "path": VORTEX_PATH})
    
    # Add TIFF if available
    try:
        import tifffile
        if TIFF_DIR.exists():
            formats.append({"name": "TIFF", "type": "tiff", "path": TIFF_DIR})
    except ImportError:
        pass
    
    # Add OME-Zarr if available
    try:
        import zarr
        from ome_zarr.io import parse_url
        if OME_ZARR_DIR.exists():
            formats.append({"name": "OME-Zarr", "type": "ome_zarr", "path": OME_ZARR_DIR})
    except ImportError:
        pass
    
    return formats


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
    formats = get_available_formats()
    
    for fmt in formats:
        for access_pattern in ["sequential", "random"]:
            for transform_type in ["none", "normalize"]:  # Reduced transforms
                # Define transforms
                if transform_type == "none":
                    transform = None
                elif transform_type == "normalize":
                    transform = transforms.Normalize(mean=[0.5], std=[0.5])
                
                dataset = OMEArrowDataset(
                    data_path=fmt["path"],
                    format_type=fmt["type"],
                    transform=transform,
                    access_pattern=access_pattern,
                    lance_table=LANCE_TABLE,
                    duck_table=DUCK_TABLE,
                )
                
                print(f"\n[Track 1] format={fmt['name']}, pattern={access_pattern}, transform={transform_type}")
                
                for run_idx in range(TRACK1_REPEATS):
                    gc.collect()
                    stats = benchmark_dataset_getitem(
                        dataset,
                        access_pattern,
                        TRACK1_BENCHMARK_SAMPLES,
                        warmup=TRACK1_WARMUP_SAMPLES,
                    )
                    
                    result = {
                        "format": fmt["name"],
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
    formats = get_available_formats()
    
    # Use only a subset of formats for Track 2 (table-based only) to reduce runtime
    table_formats = [fmt for fmt in formats if fmt["type"] not in ["tiff", "ome_zarr"]]
    
    for fmt in table_formats:
        for batch_size in TRACK2_BATCH_SIZES:
            for num_workers in TRACK2_NUM_WORKERS_OPTIONS:
                dataset = OMEArrowDataset(
                    data_path=fmt["path"],
                    format_type=fmt["type"],
                    transform=None,
                    lance_table=LANCE_TABLE,
                    duck_table=DUCK_TABLE,
                )
                
                print(f"\n[Track 2] format={fmt['name']}, batch_size={batch_size}, num_workers={num_workers}")
                
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
                        "format": fmt["name"],
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
    
    formats = get_available_formats()
    # Use only table-based formats for Track 3 to reduce runtime
    table_formats = [fmt for fmt in formats if fmt["type"] not in ["tiff", "ome_zarr"]]
    
    for fmt in table_formats:
        dataset = OMEArrowDataset(
            data_path=fmt["path"],
            format_type=fmt["type"],
            transform=transforms.Normalize(mean=[0.5], std=[0.5]),
            lance_table=LANCE_TABLE,
            duck_table=DUCK_TABLE,
        )
        
        print(f"\n[Track 3] format={fmt['name']}, batch_size={TRACK3_BATCH_SIZE}")
        
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
                "format": fmt["name"],
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Track 1: Dataset __getitem__ Latency", fontsize=14)
    
    # Color map for all formats
    color_map = {
        "Parquet": "#2F5C8A",
        "Parquet (DuckDB)": "#3D7FBF",
        "Lance": "#C86A1B",
        "Vortex": "#2E7D4F",
        "DuckDB": "#B23B3B",
        "TIFF": "#5A6B3A",
        "OME-Zarr": "#7A5A3C",
    }
    
    # Plot 1: p95 latency by format and access pattern
    ax = axes[0]
    summary = df.groupby(["format", "access_pattern"])["p95"].mean().reset_index()
    
    x_labels = []
    x_pos = []
    colors = []
    values = []
    
    # Create abbreviated labels
    pattern_abbrev = {"sequential": "seq", "random": "rnd"}
    
    pos = 0
    for _, row in summary.iterrows():
        # Shorten label: format name + pattern abbreviation
        label = f"{row['format'][:8]}\n{pattern_abbrev.get(row['access_pattern'], row['access_pattern'])}"
        x_labels.append(label)
        x_pos.append(pos)
        colors.append(color_map.get(row['format'], "#BAB0AC"))
        values.append(row['p95'] * 1000)  # Convert to ms
        pos += 1
    
    bars = ax.bar(x_pos, values, color=colors, width=0.8)
    ax.set_ylabel("p95 Latency (ms)", fontsize=11)
    ax.set_title("p95 __getitem__ Latency (lower is better)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Plot 2: Samples per second
    ax = axes[1]
    summary = df.groupby(["format", "access_pattern"])["samples_per_sec"].mean().reset_index()
    
    values = summary["samples_per_sec"].tolist()
    bars = ax.bar(x_pos, values, color=colors, width=0.8)
    ax.set_ylabel("Samples per Second", fontsize=11)
    ax.set_title("Throughput (higher is better)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    fig.savefig(IMAGES_DIR / "pytorch_benchmark_track1.png", dpi=150)
    plt.close(fig)
    print(f"Saved Track 1 plot to {IMAGES_DIR / 'pytorch_benchmark_track1.png'}")


def plot_track2_results(df: pd.DataFrame) -> None:
    """Plot Track 2 results: DataLoader throughput."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Track 2: DataLoader Throughput", fontsize=14)
    
    color_map = {
        "Parquet": "#2F5C8A",
        "Parquet (DuckDB)": "#3D7FBF",
        "Lance": "#C86A1B",
        "Vortex": "#2E7D4F",
        "DuckDB": "#B23B3B",
    }
    
    # Plot 1: Samples per second by num_workers and batch_size
    ax = axes[0]
    summary = df.groupby(["format", "num_workers", "batch_size"])["samples_per_sec"].mean().reset_index()
    
    # Shorten format names for legend
    format_abbrev = {
        "Parquet": "Pq",
        "Parquet (DuckDB)": "PqD",
        "Lance": "Ln",
        "Vortex": "Vx",
        "DuckDB": "Dk",
    }
    
    for format_type in summary["format"].unique():
        format_data = summary[summary["format"] == format_type]
        for batch_size in sorted(format_data["batch_size"].unique()):
            batch_data = format_data[format_data["batch_size"] == batch_size]
            fmt_abbr = format_abbrev.get(format_type, format_type[:4])
            ax.plot(
                batch_data["num_workers"],
                batch_data["samples_per_sec"],
                marker="o",
                label=f"{fmt_abbr} bs={batch_size}",
                color=color_map.get(format_type, "#BAB0AC"),
                linestyle="-" if batch_size == min(TRACK2_BATCH_SIZES) else "--",
            )
    
    ax.set_xlabel("Number of Workers", fontsize=11)
    ax.set_ylabel("Samples per Second", fontsize=11)
    ax.set_title("DataLoader Throughput (higher is better)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Plot 2: First batch latency - show only batch_size=16 for clarity
    ax = axes[1]
    summary = df[df["batch_size"] == 16].groupby(["format", "num_workers"])["first_batch_time"].mean().reset_index()
    
    x_labels = []
    x_pos = []
    colors = []
    values = []
    
    pos = 0
    for _, row in summary.iterrows():
        fmt_abbr = format_abbrev.get(row['format'], row['format'][:6])
        label = f"{fmt_abbr}\nw={row['num_workers']}"
        x_labels.append(label)
        x_pos.append(pos)
        colors.append(color_map.get(row['format'], "#BAB0AC"))
        values.append(row['first_batch_time'] * 1000)  # Convert to ms
        pos += 1
    
    bars = ax.bar(x_pos, values, color=colors, width=0.7)
    ax.set_ylabel("First Batch Time (ms)", fontsize=11)
    ax.set_title("Worker Startup (bs=16, lower is better)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    fig.savefig(IMAGES_DIR / "pytorch_benchmark_track2.png", dpi=150)
    plt.close(fig)
    print(f"Saved Track 2 plot to {IMAGES_DIR / 'pytorch_benchmark_track2.png'}")


def plot_track3_results(df: pd.DataFrame) -> None:
    """Plot Track 3 results: End-to-end model loop."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Track 3: End-to-End Model Loop", fontsize=14)
    
    color_map = {
        "Parquet": "#2F5C8A",
        "Parquet (DuckDB)": "#3D7FBF",
        "Lance": "#C86A1B",
        "Vortex": "#2E7D4F",
        "DuckDB": "#B23B3B",
    }
    
    # Plot 1: Step time p50
    ax = axes[0]
    summary = df.groupby("format")["step_time_p50"].mean().reset_index()
    
    bars = ax.bar(
        summary["format"],
        summary["step_time_p50"] * 1000,  # Convert to ms
        color=[color_map.get(f, "#BAB0AC") for f in summary["format"]],
        width=0.7,
    )
    ax.set_ylabel("Step Time p50 (ms)", fontsize=11)
    ax.set_title("Step Latency (lower is better)", fontsize=12)
    ax.set_xticklabels(summary["format"], rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Plot 2: Data wait fraction
    ax = axes[1]
    summary = df.groupby("format")["data_wait_fraction"].mean().reset_index()
    
    bars = ax.bar(
        summary["format"],
        summary["data_wait_fraction"] * 100,  # Convert to percentage
        color=[color_map.get(f, "#BAB0AC") for f in summary["format"]],
        width=0.7,
    )
    ax.set_ylabel("Data Wait Fraction (%)", fontsize=11)
    ax.set_title("Time Waiting on Data (lower is better)", fontsize=12)
    ax.set_xticklabels(summary["format"], rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
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
    for (fmt, pattern), group in track1_df.groupby(["format", "access_pattern"]):
        track1_summary.append({
            "format": fmt,
            "access_pattern": pattern,
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
    # Generate test data if needed (even if benchmarks already run)
    if GENERATE_DATA or RUN_BENCHMARKS:
        print("Generating test data...")
        table, ome_arrays = generate_test_data()
        write_test_data(table, ome_arrays)
    
    if RUN_BENCHMARKS:
        
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
    print(track1_df.groupby(["format", "access_pattern"])[["p50", "p95", "samples_per_sec"]].mean())
    
    print("\nTrack 2: DataLoader Throughput (mean across runs)")
    print(track2_df.groupby(["format", "num_workers", "batch_size"])[["p50", "samples_per_sec", "first_batch_time"]].mean())
    
    print("\nTrack 3: End-to-End Model Loop (mean across runs)")
    print(track3_df.groupby("format")[["step_time_p50", "data_wait_fraction", "images_per_sec"]].mean())
    
    print("\nBenchmark complete!")
