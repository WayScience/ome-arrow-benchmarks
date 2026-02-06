# Track 4: Numpy vs Torch Benchmark - Quick Reference

## What It Does

Measures the performance impact of loading data into numpy arrays versus torch tensors by separating:
1. **Numpy loading time** - Getting data as numpy arrays
2. **Tensor conversion time** - Converting numpy to torch
3. **Total time** - End-to-end torch tensor retrieval

## Why It Matters

Helps you understand:
- Is tensor conversion a bottleneck? (10-20% typical, 30%+ is high)
- Should I optimize I/O or conversion?
- What's the overhead of using torch vs staying in numpy?

## Quick Start

### Run the benchmark:
```bash
python src/benchmarks/pytorch_benchmark.py
```

### Test the implementation:
```bash
python test_track4.py
```

### View results:
```bash
# Results data
cat data/pytorch_benchmark_track4.parquet

# Summary with all tracks
cat data/pytorch_benchmark_summary.json

# Visualization
open images/pytorch_benchmark_track4.png
```

## Understanding Output

### Example Output:
```
[Track 4] format=Parquet
  Run 1/3:
    Numpy:      p50=0.042ms, throughput=23809.5 samples/s
    Conversion: p50=0.008ms, overhead=16.0%
    Torch:      p50=0.050ms, throughput=20000.0 samples/s
```

### What This Means:
- **Numpy time (0.042ms)**: Time to load and prepare numpy array
- **Conversion time (0.008ms)**: Time for torch.from_numpy().float()
- **Overhead (16%)**: Conversion is 16% of total time
- **Torch time (0.050ms)**: Total time for torch tensor

### Interpretation:
- **<10% overhead**: Conversion is negligible, focus on I/O
- **10-30% overhead**: Conversion is measurable but acceptable
- **>30% overhead**: Conversion is significant, consider optimization

## Key Metrics

| Metric | Description | Good/Bad |
|--------|-------------|----------|
| `numpy_p50` | Median numpy loading time | Lower is better |
| `conversion_p50` | Median conversion time | Lower is better |
| `torch_p50` | Median total time | Lower is better |
| `conversion_overhead_pct` | Conversion as % of total | <10% good, >30% high |
| `numpy_samples_per_sec` | Numpy throughput | Higher is better |
| `torch_samples_per_sec` | Torch throughput | Higher is better |

## Optimization Tips

### If overhead is HIGH (>30%):
```python
# Instead of:
tensor = torch.from_numpy(arr).float()  # Creates copy

# Try:
arr = arr.astype(np.float32)  # Convert in numpy first
tensor = torch.from_numpy(arr)  # Zero-copy (shares memory)
```

### If overhead is LOW (<10%):
Focus on I/O instead:
- Use faster storage (SSD)
- Increase DataLoader workers
- Pre-load more data
- Choose optimized formats

## Files & Documentation

- **Implementation**: `src/benchmarks/pytorch_benchmark.py`
  - `OMEArrowDatasetNumpy` class (returns numpy)
  - `benchmark_numpy_vs_torch()` function
  - `run_track4()` orchestration
  - `plot_track4_results()` visualization

- **Documentation**:
  - Quick guide: `docs/pytorch_benchmark.md` (Track 4 section)
  - Detailed guide: `docs/track4_implementation.md`
  - This file: `docs/TRACK4_README.md`

- **Testing**: `test_track4.py`

- **Outputs**:
  - Data: `data/pytorch_benchmark_track4.parquet`
  - Plot: `images/pytorch_benchmark_track4.png`
  - Summary: `data/pytorch_benchmark_summary.json`

## Common Questions

**Q: Why only table formats?**  
A: Directory formats (TIFF, OME-Zarr) don't use OME-Arrow structures, so conversion overhead is different.

**Q: Why is conversion overhead higher for small images?**  
A: Conversion time is relatively fixed, so it's a larger percentage of total time when I/O is fast.

**Q: Should I always avoid tensor conversion?**  
A: No! Only optimize if overhead is high (>30%). Most of the time, I/O is the bottleneck.

**Q: Can I disable Track 4?**  
A: Yes, delete `RESULTS_TRACK4_PATH.exists()` from `RUN_BENCHMARKS` check, or just ignore the results.

## Integration

Track 4 is fully integrated:
- ✅ Runs automatically with main benchmark
- ✅ Results saved alongside other tracks
- ✅ Included in summary JSON
- ✅ Plots generated automatically
- ✅ Uses same configuration as Track 1

## See Also

- Main documentation: `docs/pytorch_benchmark.md`
- Implementation details: `docs/track4_implementation.md`
- Performance optimization: `docs/pytorch_optimization.md`
- Test script: `test_track4.py`
