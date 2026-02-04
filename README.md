# OME Arrow benchmarks

[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.18234383.svg)](https://doi.org/10.5281/zenodo.18234383)

Benchmarking [OME Arrow](https://github.com/WayScience/ome-arrow) through Parquet, Vortex, LanceDB, and more.

## Available Benchmarks

### File Format Benchmarks

1. **compare_parquet_vortex_lance.py** - Wide dataset benchmark (~100k rows × 4k columns)
2. **compare_parquet_vortex_lance_ome.py** - OME-Arrow variant with image column
3. **compare_ome_arrow_only.py** - OME-Arrow-only + OME-Zarr + TIFF comparison

### PyTorch Integration Benchmark

4. **pytorch_benchmark.py** - PyTorch-focused performance testing
   - Track 1: Dataset `__getitem__` microbenchmark
   - Track 2: DataLoader throughput
   - Track 3: End-to-end model training loop
   
   See [PyTorch Benchmark Documentation](docs/pytorch_benchmark.md) for details.

## Running benchmarks

1. Create and sync a uv environment (includes parquet, lancedb, vortex-data, pytorch):

```bash
uv venv
uv sync
```

2. Run individual benchmarks:

```bash
# File format benchmarks
uv run python src/benchmarks/compare_parquet_vortex_lance.py
uv run python src/benchmarks/compare_parquet_vortex_lance_ome.py
uv run python src/benchmarks/compare_ome_arrow_only.py

# PyTorch benchmark
uv run python src/benchmarks/pytorch_benchmark.py
```

Or run all benchmarks at once:

```bash
poe run-benchmarks
```

## Configuration

The benchmarks default to ~100,000 rows × ~4,000 columns of `float64` data and ~50 columns of `string` data. Lower `N_ROWS`/`N_COLS` in the config section if you hit memory pressure.

For PyTorch benchmarks, adjust `N_ROWS` (default: 1,000 images) and other parameters in `src/benchmarks/pytorch_benchmark.py`. See the [documentation](docs/pytorch_benchmark.md) for configuration details.

## Output

Benchmarks generate:
- **Data files**: Results in Parquet and JSON format (`data/` directory)
- **Plots**: Visualizations of benchmark results (`images/` directory)
