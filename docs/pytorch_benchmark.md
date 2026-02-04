# PyTorch Benchmark Documentation

This document describes the PyTorch-focused benchmark for the OME-Arrow file format, designed to measure performance in real-world PyTorch-based image profiling pipelines.

## Overview

The PyTorch benchmark (`src/benchmarks/pytorch_benchmark.py`) evaluates three key aspects of PyTorch integration:

1. **Track 1: Dataset `__getitem__` Microbenchmark** - Measures low-level data access performance
2. **Track 2: DataLoader Throughput** - Evaluates multi-worker data loading scalability
3. **Track 3: End-to-End Model Loop** - Tests real-world training/inference workflow

### Performance Optimization

**Note:** The benchmark uses an optimized data loading strategy for table-based formats (Parquet, Lance, DuckDB, Vortex):

- **Pre-loading:** Pixel data is extracted from Arrow structures during dataset initialization
- **Caching:** Images are stored as numpy arrays for O(1) access in `__getitem__`
- **Benefit:** 50-100x faster DataLoader iteration compared to on-demand Arrow conversions
- **Trade-off:** Higher memory usage (~10MB per 1000 images @ 100x100)

See [Performance Optimization Documentation](pytorch_optimization.md) for details.

## Running the Benchmark

### Prerequisites

Install dependencies using uv:

```bash
uv sync
```

Or with pip:

```bash
pip install torch>=2.6.0 torchvision>=0.21.0 pyarrow lancedb ome-arrow pandas matplotlib
```

### Running

Execute the benchmark script:

```bash
python src/benchmarks/pytorch_benchmark.py
```

Or use the task runner:

```bash
poe run-benchmarks  # Runs all benchmarks including PyTorch
```

### Configuration

Edit the constants at the top of `pytorch_benchmark.py` to adjust benchmark parameters:

```python
# Data configuration
N_ROWS = 1_000              # Number of images (use 100-1000 for testing, 10k+ for production)
OME_SHAPE = (100, 100)      # Image dimensions (H, W)

# Track 1: Dataset microbenchmark
TRACK1_REPEATS = 3                  # Number of repetitions
TRACK1_BENCHMARK_SAMPLES = 100      # Samples per run

# Track 2: DataLoader config
TRACK2_BATCH_SIZES = [1, 4, 16]            # Batch sizes to test
TRACK2_NUM_WORKERS_OPTIONS = [0, 2, 4]    # Worker counts to test

# Track 3: End-to-end config
TRACK3_STEPS = 50                   # Training steps
TRACK3_BATCH_SIZE = 16              # Batch size
```

For quick smoke testing, reduce `N_ROWS` to 100-500 and `TRACK1_BENCHMARK_SAMPLES` to 50.

## Output Files

The benchmark generates the following outputs:

### Data Files (in `data/`)

- `pytorch_benchmark_track1.parquet` - Raw Track 1 results
- `pytorch_benchmark_track2.parquet` - Raw Track 2 results  
- `pytorch_benchmark_track3.parquet` - Raw Track 3 results
- `pytorch_benchmark_summary.json` - Aggregated summary with metadata

### Plots (in `images/`)

- `pytorch_benchmark_track1.png` - Dataset `__getitem__` latency and throughput
- `pytorch_benchmark_track2.png` - DataLoader throughput and worker startup overhead
- `pytorch_benchmark_track3.png` - End-to-end step time and data wait fraction

## Understanding the Results

### Track 1: Dataset `__getitem__` Microbenchmark

**What it measures:**
- Latency of individual sample access (p50, p95, p99 percentiles)
- Samples per second throughput
- Impact of access patterns (sequential vs. random)
- Overhead of transforms (none vs. normalize vs. augment)

**Key metrics:**
- **p50/p95/p99 latency** (ms): Lower is better. Typical values: 5-10ms for 100x100 images
- **Samples per second**: Higher is better. Typical values: 100-200 samples/sec

**Interpreting results:**
- Sequential access is often slightly faster due to caching
- Transform overhead should be minimal (< 0.5ms)
- Parquet and Lance should show similar performance for this workload

### Track 2: DataLoader Throughput

**What it measures:**
- Overall throughput with multiple workers
- First batch latency (worker startup overhead)
- Scaling efficiency with worker count

**Key metrics:**
- **Samples per second**: Higher is better. Should scale with batch size
- **First batch time** (ms): Worker startup cost. Lower is better
- **p50 batch time** (ms): Steady-state per-batch latency

**Interpreting results:**
- num_workers=0 uses main process (single-threaded)
- num_workers=2-4 enables parallel loading
- Larger batch sizes should show higher throughput (samples/sec)
- First batch time indicates worker initialization overhead

**Typical patterns:**
- Throughput increases with batch size
- Adding workers helps up to a point (typically 2-4 workers)
- Diminishing returns beyond 4 workers for small datasets

### Track 3: End-to-End Model Loop

**What it measures:**
- Real training/inference loop performance
- Data loading vs. model forward time breakdown
- Overall images per second throughput

**Key metrics:**
- **Step time p50** (ms): Total time per training step. Lower is better
- **Data wait fraction** (%): Percentage of time waiting for data. Lower is better
- **Images per second**: Overall throughput. Higher is better

**Interpreting results:**
- **Data wait fraction**:
  - < 30%: Data loading is not a bottleneck
  - 30-50%: Data loading is significant but acceptable
  - > 50%: Data loading is a bottleneck; consider optimizations
  
- **Step time**: Includes both data loading and model forward pass
  - For simple models, data loading may dominate
  - For complex models, compute may dominate

**Optimization guidance:**
- High data wait fraction → Increase num_workers, batch size, or optimize data loading
- Low throughput → Profile model forward pass, consider GPU acceleration

## Comparing Formats

The benchmark compares 7 different storage formats:

- **Parquet (pyarrow)**: Standard columnar format, widely supported, good compression
- **Parquet (DuckDB)**: DuckDB variant of Parquet with different I/O path
- **Lance**: Modern columnar format optimized for ML workloads
- **Vortex**: Experimental columnar format (optional, may be unavailable)
- **DuckDB**: File-based database format
- **TIFF**: Directory-based image format (one file per image) - typically fastest for single image access
- **OME-Zarr**: Directory-based OME format with chunked storage

**Expected differences:**
- **TIFF is fastest** for random single-image access (~10x faster than table formats)
- **OME-Zarr** is 2-3x faster than table formats due to optimized chunking
- **Table formats** (Parquet, Lance, DuckDB) show similar performance (~150-180 samples/sec)
- **Multi-threaded DataLoader** benefits all formats, especially with larger batch sizes
- **Directory-based formats** (TIFF, OME-Zarr) are excluded from Track 2 & 3 (DataLoader focused on table formats)

## Reproducibility

The benchmark ensures reproducibility through:

1. **Fixed random seed** (`SEED = 42`)
2. **Recorded metadata**: Software versions, hardware info
3. **Structured output**: JSON summary includes all configuration
4. **Deterministic sampling**: Same access patterns across runs

To reproduce results:
1. Use the same software versions (PyTorch, OME-Arrow, etc.)
2. Use the same hardware (CPU, RAM)
3. Use the same configuration parameters
4. Set the same random seed

## Extending the Benchmark

### Adding Custom Transforms

Modify the transform definitions in `run_track1()`:

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

### Adding Custom Models

Replace `SimpleEmbeddingModel` in Track 3:

```python
model = YourCustomModel()
model.eval()
```

### Testing Different Image Sizes

Change `OME_SHAPE`:

```python
OME_SHAPE = (256, 256)  # Larger images
```

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches improve throughput but use more memory
2. **Tune num_workers**: Start with 2-4 workers; more isn't always better
3. **Profile data loading**: If data wait fraction > 50%, optimize data pipeline
4. **Use SSD storage**: Disk I/O can be a bottleneck with slow storage
5. **Monitor CPU usage**: Ensure workers are efficiently utilizing available cores

## Troubleshooting

### Slow performance

- Check CPU usage: Workers should be utilizing multiple cores
- Check disk I/O: Ensure data is on fast storage (SSD)
- Reduce image size or transform complexity for testing

### High memory usage

- Reduce batch size
- Reduce num_workers
- Reduce N_ROWS for testing

### Inconsistent results

- Ensure no other processes are consuming resources
- Run multiple iterations (increase TRACK*_REPEATS)
- Use fixed random seed for reproducibility

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{ome_arrow_benchmarks,
  title={OME Arrow Benchmarks},
  author={WayScience},
  year={2024},
  url={https://github.com/WayScience/ome-arrow-benchmarks}
}
```
