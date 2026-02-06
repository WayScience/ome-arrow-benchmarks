#!/usr/bin/env python3
"""Quick test script for Track 4 numpy vs torch benchmark."""

import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from ome_arrow import OMEArrow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "benchmarks"))

from pytorch_benchmark import (
    OMEArrowDataset,
    OMEArrowDatasetNumpy,
    benchmark_numpy_vs_torch,
    DATA_DIR,
    LANCE_TABLE,
    DUCK_TABLE,
)

def create_test_data(n_rows=10, shape=(50, 50)):
    """Create minimal test data."""
    print(f"Creating test data: {n_rows} images of shape {shape}")
    
    DATA_DIR.mkdir(exist_ok=True)
    test_path = DATA_DIR / "test_track4.parquet"
    
    rng = np.random.default_rng(42)
    ome_pylist = []
    
    for _ in range(n_rows):
        img = rng.integers(0, 256, size=(1, 1, 1, *shape), dtype=np.uint8)
        ome_scalar = OMEArrow(data=img).data.as_py()
        ome_scalar.pop("masks", None)
        ome_pylist.append(ome_scalar)
    
    row_ids = pa.array(np.arange(n_rows, dtype=np.int64))
    ome_column = pa.array(ome_pylist)
    table = pa.Table.from_arrays([row_ids, ome_column], names=["row_id", "ome_image"])
    
    pq.write_table(table, test_path, compression="zstd")
    print(f"Created {test_path}")
    
    return test_path

def test_numpy_dataset():
    """Test OMEArrowDatasetNumpy returns numpy arrays."""
    print("\n" + "="*60)
    print("Test 1: OMEArrowDatasetNumpy returns numpy arrays")
    print("="*60)
    
    test_path = create_test_data(n_rows=5)
    
    dataset = OMEArrowDatasetNumpy(
        data_path=test_path,
        format_type="parquet",
    )
    
    # Test getting an item
    item = dataset[0]
    
    assert isinstance(item, np.ndarray), f"Expected numpy array, got {type(item)}"
    assert item.dtype == np.float32, f"Expected float32, got {item.dtype}"
    assert item.ndim == 3, f"Expected 3D array (C, H, W), got {item.ndim}D"
    
    print(f"✓ Item type: {type(item)}")
    print(f"✓ Item shape: {item.shape}")
    print(f"✓ Item dtype: {item.dtype}")
    print(f"✓ Item range: [{item.min():.3f}, {item.max():.3f}]")
    print("PASS: OMEArrowDatasetNumpy works correctly")

def test_torch_dataset():
    """Test OMEArrowDataset returns torch tensors."""
    print("\n" + "="*60)
    print("Test 2: OMEArrowDataset returns torch tensors")
    print("="*60)
    
    test_path = create_test_data(n_rows=5)
    
    dataset = OMEArrowDataset(
        data_path=test_path,
        format_type="parquet",
    )
    
    # Test getting an item
    item = dataset[0]
    
    assert isinstance(item, torch.Tensor), f"Expected torch tensor, got {type(item)}"
    assert item.dtype == torch.float32, f"Expected float32, got {item.dtype}"
    assert item.ndim == 3, f"Expected 3D tensor (C, H, W), got {item.ndim}D"
    
    print(f"✓ Item type: {type(item)}")
    print(f"✓ Item shape: {item.shape}")
    print(f"✓ Item dtype: {item.dtype}")
    print(f"✓ Item range: [{item.min():.3f}, {item.max():.3f}]")
    print("PASS: OMEArrowDataset works correctly")

def test_benchmark():
    """Test benchmark_numpy_vs_torch function."""
    print("\n" + "="*60)
    print("Test 3: benchmark_numpy_vs_torch")
    print("="*60)
    
    test_path = create_test_data(n_rows=20)
    
    format_config = {
        "name": "Test-Parquet",
        "type": "parquet",
        "path": test_path,
    }
    
    # Run benchmark with minimal samples
    results = benchmark_numpy_vs_torch(
        format_config,
        num_samples=10,
        warmup=3,
    )
    
    # Verify expected keys
    expected_keys = [
        "format", "num_samples",
        "numpy_mean", "numpy_p50", "numpy_p95", "numpy_p99",
        "conversion_mean", "conversion_p50", "conversion_p95", "conversion_p99",
        "torch_mean", "torch_p50", "torch_p95", "torch_p99",
        "conversion_overhead_pct", "numpy_samples_per_sec", "torch_samples_per_sec",
    ]
    
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"
    
    print(f"✓ All {len(expected_keys)} expected keys present")
    print(f"\nResults:")
    print(f"  Numpy p50:      {results['numpy_p50']*1000:.3f} ms")
    print(f"  Conversion p50: {results['conversion_p50']*1000:.3f} ms")
    print(f"  Torch p50:      {results['torch_p50']*1000:.3f} ms")
    print(f"  Overhead:       {results['conversion_overhead_pct']:.1f}%")
    print(f"  Numpy rate:     {results['numpy_samples_per_sec']:.1f} samples/s")
    print(f"  Torch rate:     {results['torch_samples_per_sec']:.1f} samples/s")
    
    # Sanity checks
    assert results['torch_mean'] >= results['numpy_mean'], \
        "Torch time should be >= numpy time"
    assert 0 <= results['conversion_overhead_pct'] <= 100, \
        "Conversion overhead should be 0-100%"
    
    print("\nPASS: benchmark_numpy_vs_torch works correctly")

def main():
    """Run all tests."""
    print("Testing Track 4: Numpy vs Torch Benchmark")
    print("="*60)
    
    try:
        test_numpy_dataset()
        test_torch_dataset()
        test_benchmark()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED ✗")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
