"""OME-IRIS join benchmarks for OME-Arrow and path-based image tables.

Fetch the OME-IRIS NF1 Cell Painting shrunken dataset, convert its profile-image
relationships into OME-Arrow, OME-TIFF, and OME-Zarr artifacts, and benchmark
simple, observable joins over real profile rows and image files. The path-based
cases represent the common current pattern: feature/profile Parquet joined to
image metadata/path Parquet with DuckDB. The converted image cases use the same
images represented in OME-Arrow, OME-TIFF, or OME-Zarr.

Artifacts are written under `data/` and plots under `figures/`.
"""

from __future__ import annotations

import gc
import importlib.metadata as importlib_metadata
import importlib.resources as resources
import shutil
import time
from pathlib import Path

import duckdb
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
from matplotlib.patches import Patch
from ome_zarr.writer import write_image
from ome_arrow import OMEArrow, OMEArrowDataset, from_numpy, write_ome_arrow_dataset
from ome_arrow.meta import OME_ARROW_BYTE_STRUCT
from ome_arrow.meta import OME_ARROW_STRUCT

import OME_IRIS
from OME_IRIS.fetch import fetch_datasets

pd.set_option("display.precision", 4)

DATASET_ID = "nf1-cellpainting-shrunken"
SOURCE_IDENTIFIER = "NF1_cellpainting_data_shrunken"
DATASET_3D_ID = "nuclei-3d"
SOURCE_3D_IDENTIFIER = "CP_tutorial_3D_noise_nuclei_segmentation"
DATA_DIR = Path("data")
OME_IRIS_DATA_DIR = DATA_DIR / "ome_iris"
BENCH_DIR = DATA_DIR / "ome_iris_join_benchmark"
BENCH_3D_DIR = DATA_DIR / "ome_iris_3d_benchmark"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)
BENCH_3D_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PARQUET = DATA_DIR / "compare_ome_iris_joins_summary.parquet"
RUNS_PARQUET = DATA_DIR / "compare_ome_iris_joins_runs.parquet"
TABLE_SUMMARY_PARQUET = DATA_DIR / "compare_ome_iris_joins_tables.parquet"
FORMAT_SUMMARY_PARQUET = DATA_DIR / "compare_ome_iris_formats_summary.parquet"
FORMAT_RUNS_PARQUET = DATA_DIR / "compare_ome_iris_formats_runs.parquet"
SUMMARY_3D_PARQUET = DATA_DIR / "compare_ome_iris_3d_summary.parquet"
RUNS_3D_PARQUET = DATA_DIR / "compare_ome_iris_3d_runs.parquet"
OME_ARROW_VERSION = importlib_metadata.version("ome-arrow")
BENCHMARK_VERSION = f"profile_ome_arrow_ome_tiff_ome_zarr_v14:{OME_ARROW_VERSION}"

OBSERVATION_REPEATS = 1_000
TIMING_REPEATS = 9
FORMAT_WRITE_REPEATS = 5
FORMAT_READ_REPEATS = 9
RANDOM_IMAGE_COUNT = 3
N_FEATURE_COLUMNS = 24
CHANNELS = ("DAPI", "GFP", "RFP")

OBSERVATIONS_PATH = BENCH_DIR / "observations.parquet"
IMAGE_PATH_INDEX_PATH = BENCH_DIR / "image_path_index.parquet"
IMAGE_OME_INDEX_PATH = BENCH_DIR / "image_ome_arrow_index.parquet"
IMAGE_OME_BYTE_INDEX_PATH = BENCH_DIR / "image_ome_arrow_byte_index.parquet"
IMAGE_OME_BYTE_VORTEX_PATH = BENCH_DIR / "image_ome_arrow_byte_index.vortex"
IMAGE_OME_BYTE_LANCE_PATH = BENCH_DIR / "image_ome_arrow_byte_index_lance"
CONVERTED_OME_DATASET_PATH = BENCH_DIR / "nf1_profile_images_ome_arrow.parquet"
CONVERTED_OME_BYTE_DATASET_PATH = (
    BENCH_DIR / "nf1_profile_images_ome_arrow_bytes.parquet"
)
OME_ARROW_DATASET_DIR = BENCH_DIR / "ome_arrow_typed_chunk_dataset"
OME_ARROW_DATASET_NONE_DIR = BENCH_DIR / "ome_arrow_typed_chunk_dataset_none"
OME_TIFF_DIR = BENCH_DIR / "ome_tiff_images"
OME_ZARR_DIR = BENCH_DIR / "ome_zarr_images"
OME_ARROW_3D_ZPLANE_DIR = BENCH_3D_DIR / "ome_arrow_zplane"
OME_ARROW_3D_BLOCK_DIR = BENCH_3D_DIR / "ome_arrow_block"
OME_ARROW_3D_BLOCK_NONE_DIR = BENCH_3D_DIR / "ome_arrow_block_none"
OME_ZARR_3D_WHOLE_DIR = BENCH_3D_DIR / "ome_zarr_whole"
OME_ZARR_3D_BLOCK_DIR = BENCH_3D_DIR / "ome_zarr_block"

PLOT_TITLE = "OME-IRIS NF1 joins and converted OME image formats"
OA_KEY_TEXT = (
    "OA NT = OME-Arrow nested table; OA DS = OMEArrowDataset; "
    "Parquet/Vortex/Lance = table backend"
)
SOURCE_NOTE = "Datasets: OME-IRIS catalog entries."


def cache_is_current() -> bool:
    if not (
        SUMMARY_PARQUET.exists()
        and RUNS_PARQUET.exists()
        and TABLE_SUMMARY_PARQUET.exists()
        and FORMAT_SUMMARY_PARQUET.exists()
        and FORMAT_RUNS_PARQUET.exists()
        and SUMMARY_3D_PARQUET.exists()
        and RUNS_3D_PARQUET.exists()
        and CONVERTED_OME_DATASET_PATH.exists()
        and OME_ARROW_DATASET_DIR.exists()
        and OME_ARROW_DATASET_NONE_DIR.exists()
        and OME_TIFF_DIR.exists()
        and OME_ZARR_DIR.exists()
        and CONVERTED_OME_BYTE_DATASET_PATH.exists()
        and IMAGE_OME_BYTE_VORTEX_PATH.exists()
        and IMAGE_OME_BYTE_LANCE_PATH.exists()
        and OME_ARROW_3D_ZPLANE_DIR.exists()
        and OME_ARROW_3D_BLOCK_DIR.exists()
        and OME_ARROW_3D_BLOCK_NONE_DIR.exists()
        and OME_ZARR_3D_WHOLE_DIR.exists()
        and OME_ZARR_3D_BLOCK_DIR.exists()
    ):
        return False
    try:
        summary = pd.read_parquet(SUMMARY_PARQUET, columns=["benchmark_version"])
        format_summary = pd.read_parquet(
            FORMAT_SUMMARY_PARQUET, columns=["benchmark_version"]
        )
        summary_3d = pd.read_parquet(SUMMARY_3D_PARQUET, columns=["benchmark_version"])
    except Exception:
        return False
    return bool(
        (summary["benchmark_version"] == BENCHMARK_VERSION).all()
        and (format_summary["benchmark_version"] == BENCHMARK_VERSION).all()
        and (summary_3d["benchmark_version"] == BENCHMARK_VERSION).all()
    )


RUN_BENCHMARKS = not cache_is_current()


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def drop_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def fetch_dataset(
    dataset_id: str = DATASET_ID, source_identifier: str = SOURCE_IDENTIFIER
) -> Path:
    manifests_dir = resources.files(OME_IRIS) / "data" / "datasets"
    result = fetch_datasets(
        manifests_dir=Path(str(manifests_dir)),
        data_dir=OME_IRIS_DATA_DIR,
        dataset_id=dataset_id,
        silent=True,
    )
    if result.failed:
        raise RuntimeError(f"OME-IRIS fetch failed: {result.failed}")
    dataset_dir = OME_IRIS_DATA_DIR / source_identifier
    images_path = dataset_dir / "images"
    if not images_path.exists():
        raise FileNotFoundError(f"Expected OME-IRIS dataset under {dataset_dir}")
    return dataset_dir


def build_image_records(dataset_dir: Path) -> list[dict[str, object]]:
    images_path = dataset_dir / "images"
    records = []
    for image_path in sorted(images_path.glob("*.tif")):
        parts = image_path.name.split("_")
        channel = parts[3] if len(parts) >= 4 else "unknown"
        arr = tifffile.imread(image_path)
        records.append(
            {
                "image_key": image_path.name,
                "channel": channel,
                "file_name": image_path.name,
                "image_path": str(image_path),
                "size_bytes": image_path.stat().st_size,
                "height": int(arr.shape[-2]),
                "width": int(arr.shape[-1]),
                "dtype": str(arr.dtype),
            }
        )
    return records


def build_observations(profile_path: Path, image_records: list[dict[str, object]]):
    profile = pq.read_table(profile_path).to_pandas()
    numeric_cols = [
        col
        for col in profile.columns
        if pd.api.types.is_numeric_dtype(profile[col])
        and not col.startswith("__index_level_")
    ][:N_FEATURE_COLUMNS]
    image_keys = {str(rec["image_key"]) for rec in image_records}
    base_rows = []
    for source_row_id, row in profile.reset_index(drop=True).iterrows():
        for channel in CHANNELS:
            image_key = str(row[f"Image_FileName_{channel}"])
            if image_key not in image_keys:
                continue
            base = {
                "source_row_id": int(source_row_id),
                "image_key": image_key,
                "channel": channel,
                "metadata_well": str(row.get("Image_Metadata_Well_y", "")),
                "metadata_site": str(row.get("Image_Metadata_Site_y", "")),
            }
            for col in numeric_cols:
                base[col] = row[col]
            base_rows.append(base)

    repeated = []
    for repeat_id in range(OBSERVATION_REPEATS):
        for base_idx, base in enumerate(base_rows):
            repeated.append(
                {
                    "obs_id": repeat_id * len(base_rows) + base_idx,
                    "repeat_id": repeat_id,
                    **base,
                }
            )
    return pd.DataFrame(repeated), pd.DataFrame(base_rows), numeric_cols


def build_ome_arrow_table(image_records: list[dict[str, object]]) -> pa.Table:
    ome_pylist = []
    keys = []
    channels = []
    for rec in image_records:
        ome_scalar = OMEArrow(data=str(rec["image_path"])).data.as_py()
        ome_scalar["masks"] = None
        ome_pylist.append(ome_scalar)
        keys.append(rec["image_key"])
        channels.append(rec["channel"])
    return pa.table(
        {
            "image_key": pa.array(keys),
            "channel": pa.array(channels),
            "ome_image": pa.array(ome_pylist, type=OME_ARROW_STRUCT),
        }
    )


def build_ome_arrow_byte_table(image_records: list[dict[str, object]]) -> pa.Table:
    ome_pylist = []
    keys = []
    channels = []
    for rec in image_records:
        arr = image_to_tczyx(str(rec["image_path"]))
        ome_scalar = from_numpy(
            arr,
            dim_order="TCZYX",
            image_id=str(rec["image_key"]),
            name=Path(str(rec["image_path"])).name,
            channel_names=[str(rec["channel"])],
            clamp_to_uint16=False,
            chunk_shape=(1, arr.shape[-2], arr.shape[-1]),
            chunk_encoding="bytes",
        )
        ome_pylist.append(ome_scalar.as_py())
        keys.append(rec["image_key"])
        channels.append(rec["channel"])
    return pa.table(
        {
            "image_key": pa.array(keys),
            "channel": pa.array(channels),
            "ome_image": pa.array(ome_pylist, type=OME_ARROW_BYTE_STRUCT),
        }
    )


def image_to_tczyx(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    return arr.reshape((1, 1, 1, *arr.shape[-2:]))


def write_ome_tiff_images(image_records: list[dict[str, object]]) -> None:
    OME_TIFF_DIR.mkdir(parents=True, exist_ok=True)
    for rec in image_records:
        arr = image_to_tczyx(str(rec["image_path"]))
        out_path = OME_TIFF_DIR / f"{Path(str(rec['image_key'])).stem}.ome.tiff"
        tifffile.imwrite(
            out_path,
            arr,
            ome=True,
            metadata={"axes": "TCZYX", "Channel": {"Name": str(rec["channel"])}},
        )


def write_ome_zarr_images(image_records: list[dict[str, object]]) -> None:
    OME_ZARR_DIR.mkdir(parents=True, exist_ok=True)
    for rec in image_records:
        arr = image_to_tczyx(str(rec["image_path"]))
        out_path = OME_ZARR_DIR / f"{Path(str(rec['image_key'])).stem}.zarr"
        store = zarr.storage.LocalStore(str(out_path))
        root = zarr.group(store=store, overwrite=True)
        write_image(arr, root, axes="tczyx", scale_factors=[], method=None)


def write_ome_arrow_index(image_records: list[dict[str, object]], path: Path) -> None:
    pq.write_table(build_ome_arrow_table(image_records), path, compression="zstd")


def write_ome_arrow_byte_index(
    image_records: list[dict[str, object]], path: Path
) -> None:
    pq.write_table(build_ome_arrow_byte_table(image_records), path, compression="zstd")


def write_ome_arrow_typed_chunk_dataset(
    image_records: list[dict[str, object]], path: Path, compression: str | None = "zstd"
) -> None:
    arrays = [image_to_tczyx(str(rec["image_path"])) for rec in image_records]
    write_ome_arrow_dataset(
        arrays,
        path,
        layout="image",
        access_pattern="random_image",
        compression=compression,
        build_physical_index=True,
        chunk_rows_per_row_group=1,
    )


def converted_image_paths(base_dir: Path, suffix: str) -> list[Path]:
    return sorted(base_dir.glob(f"*{suffix}"))


def table_from_arrays(paths: list[Path], arrays: list[np.ndarray]) -> pa.Table:
    return pa.table(
        {
            "image_key": pa.array([path.name for path in paths]),
            "pixels": pa.array([arr.reshape(-1) for arr in arrays]),
        }
    )


def read_ome_tiff_metadata() -> pa.Table:
    records = []
    for path in converted_image_paths(OME_TIFF_DIR, ".ome.tiff"):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            records.append(
                {
                    "image_key": path.name,
                    "height": int(series.shape[-2]),
                    "width": int(series.shape[-1]),
                    "dtype": str(series.dtype),
                    "axes": series.axes,
                    "is_ome": bool(tif.ome_metadata),
                }
            )
    return pa.Table.from_pylist(records)


def read_ome_tiff_payload() -> pa.Table:
    paths = converted_image_paths(OME_TIFF_DIR, ".ome.tiff")
    arrays = [tifffile.imread(path) for path in paths]
    return table_from_arrays(paths, arrays)


def read_ome_tiff_random_payload(indices: list[int]) -> pa.Table:
    paths = [converted_image_paths(OME_TIFF_DIR, ".ome.tiff")[idx] for idx in indices]
    arrays = [tifffile.imread(path) for path in paths]
    return table_from_arrays(paths, arrays)


def read_ome_zarr_metadata() -> pa.Table:
    records = []
    for path in converted_image_paths(OME_ZARR_DIR, ".zarr"):
        group = zarr.open_group(str(path), mode="r")
        arr = group["s0"]
        records.append(
            {
                "image_key": path.name,
                "height": int(arr.shape[-2]),
                "width": int(arr.shape[-1]),
                "dtype": str(arr.dtype),
                "axes": "tczyx",
                "is_ome": "ome" in group.attrs.asdict(),
            }
        )
    return pa.Table.from_pylist(records)


def read_ome_zarr_payload() -> pa.Table:
    paths = converted_image_paths(OME_ZARR_DIR, ".zarr")
    arrays = [zarr.open_group(str(path), mode="r")["s0"][...] for path in paths]
    return table_from_arrays(paths, arrays)


def read_ome_zarr_random_payload(indices: list[int]) -> pa.Table:
    paths = [converted_image_paths(OME_ZARR_DIR, ".zarr")[idx] for idx in indices]
    arrays = [zarr.open_group(str(path), mode="r")["s0"][...] for path in paths]
    return table_from_arrays(paths, arrays)


def read_ome_arrow_payload(path: Path = IMAGE_OME_INDEX_PATH) -> pa.Table:
    return pq.read_table(path, columns=["image_key", "ome_image"])


def read_ome_arrow_random_payload(
    image_keys: list[str], path: Path = IMAGE_OME_INDEX_PATH
) -> pa.Table:
    quoted_keys = ", ".join(f"'{key}'" for key in image_keys)
    with duckdb.connect() as con:
        return con.execute(
            f"""
            SELECT image_key, ome_image
            FROM read_parquet('{path}')
            WHERE image_key IN ({quoted_keys})
            ORDER BY image_key
            """
        ).to_arrow_table()


def write_ome_arrow_byte_vortex(
    image_records: list[dict[str, object]], path: Path = IMAGE_OME_BYTE_VORTEX_PATH
) -> None:
    drop_path(path)
    vxio.write(build_ome_arrow_byte_table(image_records), str(path))


def read_ome_arrow_vortex_payload(path: Path = IMAGE_OME_BYTE_VORTEX_PATH) -> pa.Table:
    return (
        vortex.open(str(path)).to_arrow().read_all().select(["image_key", "ome_image"])
    )


def read_ome_arrow_vortex_random_payload(
    image_keys: list[str], path: Path = IMAGE_OME_BYTE_VORTEX_PATH
) -> pa.Table:
    table = read_ome_arrow_vortex_payload(path)
    wanted = set(image_keys)
    return pa.Table.from_pylist(
        [row for row in table.to_pylist() if row["image_key"] in wanted]
    )


def write_ome_arrow_byte_lance(
    image_records: list[dict[str, object]], path: Path = IMAGE_OME_BYTE_LANCE_PATH
) -> None:
    drop_path(path)
    db = lancedb.connect(path)
    db.create_table(
        "images", build_ome_arrow_byte_table(image_records), mode="overwrite"
    )


def read_ome_arrow_lance_payload(path: Path = IMAGE_OME_BYTE_LANCE_PATH) -> pa.Table:
    db = lancedb.connect(path)
    return db.open_table("images").to_arrow().select(["image_key", "ome_image"])


def read_ome_arrow_lance_random_payload(
    image_keys: list[str], path: Path = IMAGE_OME_BYTE_LANCE_PATH
) -> pa.Table:
    quoted_keys = ", ".join(f"'{key}'" for key in image_keys)
    db = lancedb.connect(path)
    try:
        return (
            db.open_table("images")
            .query()
            .where(f"image_key IN ({quoted_keys})")
            .to_arrow()
            .select(["image_key", "ome_image"])
        )
    except Exception:
        table = db.open_table("images").to_arrow().select(["image_key", "ome_image"])
        wanted = set(image_keys)
        indices = [
            idx
            for idx, key in enumerate(table.column("image_key").to_pylist())
            if key in wanted
        ]
        return table.take(pa.array(indices, type=pa.int64()))


def read_ome_arrow_typed_chunk_payload(path: Path = OME_ARROW_DATASET_DIR):
    dataset = OMEArrowDataset(path)
    return dataset.read_many(dataset.image_ids)


def read_ome_arrow_typed_chunk_payload_open(dataset: OMEArrowDataset):
    return dataset.read_many(dataset.image_ids)


def read_ome_arrow_typed_chunk_random_payload(
    indices: list[int], path: Path = OME_ARROW_DATASET_DIR
):
    dataset = OMEArrowDataset(path)
    image_ids = [dataset.image_ids[idx] for idx in indices]
    return dataset.read_many(image_ids)


def read_ome_arrow_typed_chunk_random_payload_open(
    dataset: OMEArrowDataset, indices: list[int]
):
    image_ids = [dataset.image_ids[idx] for idx in indices]
    return dataset.read_many(image_ids)


def read_ome_arrow_typed_chunk_rows_open(dataset: OMEArrowDataset):
    return [dataset.read_chunk_rows(image_id) for image_id in dataset.image_ids]


def read_ome_arrow_typed_chunk_random_rows_open(
    dataset: OMEArrowDataset, indices: list[int]
):
    image_ids = [dataset.image_ids[idx] for idx in indices]
    return [dataset.read_chunk_rows(image_id) for image_id in image_ids]


def write_benchmark_tables(dataset_dir: Path) -> pd.DataFrame:
    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    image_records = build_image_records(dataset_dir)
    observations, image_level_observations, feature_cols = build_observations(
        dataset_dir / "profiles.parquet", image_records
    )
    image_path_index = pd.DataFrame(image_records)
    image_level_path = BENCH_DIR / "image_level_observations.parquet"
    ome_arrow_table = build_ome_arrow_table(image_records)
    ome_arrow_byte_table = build_ome_arrow_byte_table(image_records)

    observations.to_parquet(OBSERVATIONS_PATH, index=False)
    image_level_observations.to_parquet(image_level_path, index=False)
    image_path_index.to_parquet(IMAGE_PATH_INDEX_PATH, index=False)
    pq.write_table(ome_arrow_table, IMAGE_OME_INDEX_PATH, compression="zstd")
    pq.write_table(ome_arrow_byte_table, IMAGE_OME_BYTE_INDEX_PATH, compression="zstd")
    write_ome_arrow_typed_chunk_dataset(image_records, OME_ARROW_DATASET_DIR)
    write_ome_arrow_typed_chunk_dataset(
        image_records, OME_ARROW_DATASET_NONE_DIR, compression=None
    )
    write_ome_tiff_images(image_records)
    write_ome_zarr_images(image_records)

    with duckdb.connect() as con:
        con.execute(
            f"""
            COPY (
                SELECT o.*, img.ome_image
                FROM read_parquet('{image_level_path}') AS o
                JOIN read_parquet('{IMAGE_OME_INDEX_PATH}') AS img
                USING (image_key)
                ORDER BY o.source_row_id, o.channel
            )
            TO '{CONVERTED_OME_DATASET_PATH}'
            WITH (FORMAT 'PARQUET', COMPRESSION 'ZSTD')
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT o.*, img.ome_image
                FROM read_parquet('{image_level_path}') AS o
                JOIN read_parquet('{IMAGE_OME_BYTE_INDEX_PATH}') AS img
                USING (image_key)
                ORDER BY o.source_row_id, o.channel
            )
            TO '{CONVERTED_OME_BYTE_DATASET_PATH}'
            WITH (FORMAT 'PARQUET', COMPRESSION 'ZSTD')
            """
        )

    table_summary = pd.DataFrame(
        [
            {
                "table": "OME-IRIS source dataset",
                "rows": np.nan,
                "columns": np.nan,
                "size_mb": path_size_bytes(dataset_dir) / (1024 * 1024),
            },
            {
                "table": "observations",
                "rows": len(observations),
                "columns": observations.shape[1],
                "size_mb": path_size_bytes(OBSERVATIONS_PATH) / (1024 * 1024),
            },
            {
                "table": "image path index",
                "rows": len(image_path_index),
                "columns": image_path_index.shape[1],
                "size_mb": path_size_bytes(IMAGE_PATH_INDEX_PATH) / (1024 * 1024),
            },
            {
                "table": "OA NT image index",
                "rows": len(image_records),
                "columns": 3,
                "size_mb": path_size_bytes(IMAGE_OME_BYTE_INDEX_PATH) / (1024 * 1024),
            },
            {
                "table": "OA DS typed chunk dataset",
                "rows": len(image_records),
                "columns": np.nan,
                "size_mb": path_size_bytes(OME_ARROW_DATASET_DIR) / (1024 * 1024),
            },
            {
                "table": "OA DS typed chunk dataset, uncompressed",
                "rows": len(image_records),
                "columns": np.nan,
                "size_mb": path_size_bytes(OME_ARROW_DATASET_NONE_DIR) / (1024 * 1024),
            },
            {
                "table": "converted NF1 OA NT profile-image table",
                "rows": len(image_level_observations),
                "columns": image_level_observations.shape[1] + 1,
                "size_mb": path_size_bytes(CONVERTED_OME_BYTE_DATASET_PATH)
                / (1024 * 1024),
            },
            {
                "table": "converted OME-TIFF images",
                "rows": len(image_records),
                "columns": np.nan,
                "size_mb": path_size_bytes(OME_TIFF_DIR) / (1024 * 1024),
            },
            {
                "table": "converted OME-Zarr images",
                "rows": len(image_records),
                "columns": np.nan,
                "size_mb": path_size_bytes(OME_ZARR_DIR) / (1024 * 1024),
            },
        ]
    )
    table_summary["feature_columns_used"] = len(feature_cols)
    table_summary.to_parquet(TABLE_SUMMARY_PARQUET, index=False)
    return table_summary


def run_format_benchmarks(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_records = build_image_records(dataset_dir)
    random_indices = list(range(min(RANDOM_IMAGE_COUNT, len(image_records))))
    random_image_keys = sorted(
        str(image_records[idx]["image_key"]) for idx in random_indices
    )
    configs = [
        {
            "format": "OME-Arrow nested bytes table, Parquet",
            "path": IMAGE_OME_BYTE_INDEX_PATH,
            "write": lambda: write_ome_arrow_byte_index(
                image_records, IMAGE_OME_BYTE_INDEX_PATH
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_arrow_payload(
                IMAGE_OME_BYTE_INDEX_PATH
            ),
            "read_random": lambda _state: read_ome_arrow_random_payload(
                random_image_keys, IMAGE_OME_BYTE_INDEX_PATH
            ),
            "cleanup": lambda: drop_path(IMAGE_OME_BYTE_INDEX_PATH),
            "version": f"ome-arrow {OME_ARROW_VERSION}; pyarrow {pa.__version__}",
        },
        {
            "format": "OME-Arrow nested bytes table, Vortex",
            "path": IMAGE_OME_BYTE_VORTEX_PATH,
            "write": lambda: write_ome_arrow_byte_vortex(
                image_records, IMAGE_OME_BYTE_VORTEX_PATH
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_arrow_vortex_payload(
                IMAGE_OME_BYTE_VORTEX_PATH
            ),
            "read_random": lambda _state: read_ome_arrow_vortex_random_payload(
                random_image_keys, IMAGE_OME_BYTE_VORTEX_PATH
            ),
            "cleanup": lambda: drop_path(IMAGE_OME_BYTE_VORTEX_PATH),
            "version": (
                f"ome-arrow {OME_ARROW_VERSION}; "
                f"vortex {getattr(vortex, '__version__', importlib_metadata.version('vortex-data'))}"
            ),
        },
        {
            "format": "OME-Arrow nested bytes table, Lance",
            "path": IMAGE_OME_BYTE_LANCE_PATH,
            "write": lambda: write_ome_arrow_byte_lance(
                image_records, IMAGE_OME_BYTE_LANCE_PATH
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_arrow_lance_payload(
                IMAGE_OME_BYTE_LANCE_PATH
            ),
            "read_random": lambda _state: read_ome_arrow_lance_random_payload(
                random_image_keys, IMAGE_OME_BYTE_LANCE_PATH
            ),
            "cleanup": lambda: drop_path(IMAGE_OME_BYTE_LANCE_PATH),
            "version": (
                f"ome-arrow {OME_ARROW_VERSION}; "
                f"lancedb {importlib_metadata.version('lancedb')}"
            ),
        },
        {
            "format": "OME-Arrow chunks zstd, NumPy, reopen",
            "path": OME_ARROW_DATASET_DIR,
            "write": lambda: write_ome_arrow_typed_chunk_dataset(
                image_records, OME_ARROW_DATASET_DIR
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_arrow_typed_chunk_payload(),
            "read_random": lambda _state: read_ome_arrow_typed_chunk_random_payload(
                random_indices
            ),
            "cleanup": lambda: drop_path(OME_ARROW_DATASET_DIR),
            "version": f"ome-arrow {OME_ARROW_VERSION}",
        },
        {
            "format": "OME-Arrow chunks zstd, NumPy, open",
            "path": OME_ARROW_DATASET_DIR,
            "write": lambda: write_ome_arrow_typed_chunk_dataset(
                image_records, OME_ARROW_DATASET_DIR
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_DATASET_DIR),
            "read_all": read_ome_arrow_typed_chunk_payload_open,
            "read_random": lambda dataset: (
                read_ome_arrow_typed_chunk_random_payload_open(dataset, random_indices)
            ),
            "cleanup": lambda: drop_path(OME_ARROW_DATASET_DIR),
            "version": f"ome-arrow {OME_ARROW_VERSION}",
        },
        {
            "format": "OME-Arrow chunks zstd, rows, open",
            "path": OME_ARROW_DATASET_DIR,
            "write": lambda: write_ome_arrow_typed_chunk_dataset(
                image_records, OME_ARROW_DATASET_DIR
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_DATASET_DIR),
            "read_all": read_ome_arrow_typed_chunk_rows_open,
            "read_random": lambda dataset: read_ome_arrow_typed_chunk_random_rows_open(
                dataset, random_indices
            ),
            "cleanup": lambda: drop_path(OME_ARROW_DATASET_DIR),
            "version": f"ome-arrow {OME_ARROW_VERSION}",
        },
        {
            "format": "OME-Arrow chunks none, NumPy, open",
            "path": OME_ARROW_DATASET_NONE_DIR,
            "write": lambda: write_ome_arrow_typed_chunk_dataset(
                image_records, OME_ARROW_DATASET_NONE_DIR, compression=None
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_DATASET_NONE_DIR),
            "read_all": read_ome_arrow_typed_chunk_payload_open,
            "read_random": lambda dataset: (
                read_ome_arrow_typed_chunk_random_payload_open(dataset, random_indices)
            ),
            "cleanup": lambda: drop_path(OME_ARROW_DATASET_NONE_DIR),
            "version": f"ome-arrow {OME_ARROW_VERSION}",
        },
        {
            "format": "OME-TIFF (dir-per-image)",
            "path": OME_TIFF_DIR,
            "write": lambda: write_ome_tiff_images(image_records),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_tiff_payload(),
            "read_random": lambda _state: read_ome_tiff_random_payload(random_indices),
            "cleanup": lambda: drop_path(OME_TIFF_DIR),
            "version": f"tifffile {importlib_metadata.version('tifffile')}",
        },
        {
            "format": "OME-Zarr (dir-per-image)",
            "path": OME_ZARR_DIR,
            "write": lambda: write_ome_zarr_images(image_records),
            "open": lambda: None,
            "read_all": lambda _state: read_ome_zarr_payload(),
            "read_random": lambda _state: read_ome_zarr_random_payload(random_indices),
            "cleanup": lambda: drop_path(OME_ZARR_DIR),
            "version": (
                f"ome-zarr {importlib_metadata.version('ome-zarr')}; "
                f"zarr {importlib_metadata.version('zarr')}; "
                f"numcodecs {importlib_metadata.version('numcodecs')}"
            ),
        },
    ]

    summaries = []
    runs = []
    for cfg in configs:
        write_times = []
        read_all_times = []
        read_random_times = []
        print(f"[format start] {cfg['format']}", flush=True)
        for run_idx in range(FORMAT_WRITE_REPEATS):
            cfg["cleanup"]()
            gc.collect()
            t0 = time.perf_counter()
            cfg["write"]()
            seconds = time.perf_counter() - t0
            write_times.append(seconds)
            print(
                f"[write] {cfg['format']} run {run_idx + 1}/{FORMAT_WRITE_REPEATS}: "
                f"{seconds:.4f}s",
                flush=True,
            )

        size_mb = path_size_bytes(cfg["path"]) / (1024 * 1024)
        read_state = cfg["open"]()

        for run_idx in range(FORMAT_READ_REPEATS):
            gc.collect()
            t0 = time.perf_counter()
            _ = cfg["read_all"](read_state)
            seconds = time.perf_counter() - t0
            read_all_times.append(seconds)
            print(
                f"[read ] {cfg['format']} run {run_idx + 1}/{FORMAT_READ_REPEATS}: "
                f"{seconds:.4f}s",
                flush=True,
            )

        for run_idx in range(FORMAT_READ_REPEATS):
            gc.collect()
            t0 = time.perf_counter()
            _ = cfg["read_random"](read_state)
            seconds = time.perf_counter() - t0
            read_random_times.append(seconds)
            print(
                f"[rnd  ] {cfg['format']} run {run_idx + 1}/{FORMAT_READ_REPEATS}: "
                f"{seconds:.4f}s",
                flush=True,
            )

        summaries.append(
            {
                "format": cfg["format"],
                "benchmark_version": BENCHMARK_VERSION,
                "write_avg_s": float(np.mean(write_times)),
                "write_std_s": float(np.std(write_times)),
                "read_all_avg_s": float(np.mean(read_all_times)),
                "read_all_std_s": float(np.std(read_all_times)),
                "read_random_avg_s": float(np.mean(read_random_times)),
                "read_random_std_s": float(np.std(read_random_times)),
                "size_mb": size_mb,
                "random_images": len(random_indices),
                "version": cfg["version"],
            }
        )
        for kind, values in (
            ("write", write_times),
            ("read", read_all_times),
            ("random_read", read_random_times),
        ):
            for run_idx, seconds in enumerate(values):
                runs.append(
                    {
                        "format": cfg["format"],
                        "kind": kind,
                        "run": run_idx,
                        "seconds": seconds,
                    }
                )
        print(f"[format end] {cfg['format']}", flush=True)

    format_summary = pd.DataFrame(summaries)
    format_runs = pd.DataFrame(runs)
    format_summary.to_parquet(FORMAT_SUMMARY_PARQUET, index=False)
    format_runs.to_parquet(FORMAT_RUNS_PARQUET, index=False)
    return format_summary, format_runs


def image_to_3d_tczyx(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    return arr.reshape((1, 1, *arr.shape[-3:]))


def build_3d_arrays(
    dataset_dir: Path,
) -> tuple[list[str], list[np.ndarray], list[Path]]:
    paths = sorted((dataset_dir / "images").glob("*.tif"))
    image_ids = [path.stem for path in paths]
    arrays = [image_to_3d_tczyx(path) for path in paths]
    return image_ids, arrays, paths


def write_ome_arrow_3d_dataset(
    arrays: list[np.ndarray],
    path: Path,
    *,
    layout: str,
    chunk_shape: tuple[int, int, int, int, int],
    compression: str | None = "zstd",
) -> None:
    write_ome_arrow_dataset(
        arrays,
        path,
        layout=layout,
        access_pattern="subvolume_3d",
        chunk_shape=chunk_shape,
        compression=compression,
        build_physical_index=True,
        chunk_rows_per_row_group=1,
    )


def write_zarr_3d_dataset(
    image_ids: list[str],
    arrays: list[np.ndarray],
    path: Path,
    *,
    chunks: tuple[int, int, int, int, int],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for image_id, arr in zip(image_ids, arrays):
        group = zarr.group(
            store=zarr.storage.LocalStore(str(path / f"{image_id}.zarr")),
            overwrite=True,
        )
        group.create_array("s0", data=arr, chunks=chunks)


def read_zarr_3d_all(path: Path) -> list[np.ndarray]:
    arrays = []
    for store_path in sorted(path.glob("*.zarr")):
        arrays.append(zarr.open_group(str(store_path), mode="r")["s0"][...])
    return arrays


def read_zarr_3d_plane(path: Path, index: int = 0, z_index: int = 10) -> np.ndarray:
    store_path = sorted(path.glob("*.zarr"))[index]
    return zarr.open_group(str(store_path), mode="r")["s0"][0, 0, z_index, :, :]


def read_zarr_3d_subvolume(path: Path, index: int = 0) -> np.ndarray:
    store_path = sorted(path.glob("*.zarr"))[index]
    return zarr.open_group(str(store_path), mode="r")["s0"][0, 0, 10:26, 64:192, 64:192]


def read_tiff_3d_all(paths: list[Path]) -> list[np.ndarray]:
    return [image_to_3d_tczyx(path) for path in paths]


def read_tiff_3d_plane(paths: list[Path], index: int = 0, z_index: int = 10):
    return image_to_3d_tczyx(paths[index])[0, 0, z_index]


def read_tiff_3d_subvolume(paths: list[Path], index: int = 0):
    return image_to_3d_tczyx(paths[index])[0, 0, 10:26, 64:192, 64:192]


def timed_callable(fn, repeats: int = FORMAT_READ_REPEATS) -> list[float]:
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        _ = fn()
        times.append(time.perf_counter() - t0)
    return times


def run_3d_benchmarks(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if BENCH_3D_DIR.exists():
        shutil.rmtree(BENCH_3D_DIR)
    BENCH_3D_DIR.mkdir(parents=True, exist_ok=True)

    image_ids, arrays, source_paths = build_3d_arrays(dataset_dir)
    _, _, size_z, size_y, size_x = arrays[0].shape
    zplane_shape = (1, 1, 1, size_y, size_x)
    block_shape = (1, 1, min(16, size_z), min(128, size_y), min(128, size_x))

    configs = [
        {
            "format": "OME-Arrow 3D z-plane",
            "path": OME_ARROW_3D_ZPLANE_DIR,
            "write": lambda: write_ome_arrow_3d_dataset(
                arrays,
                OME_ARROW_3D_ZPLANE_DIR,
                layout="z-plane",
                chunk_shape=zplane_shape,
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_3D_ZPLANE_DIR),
            "read_all": lambda dataset: dataset.read_many(dataset.image_ids),
            "read_plane": lambda dataset: dataset.read_plane(
                dataset.image_ids[0], z=10
            ),
            "read_subvolume": lambda dataset: dataset.read_region(
                dataset.image_ids[0],
                z=slice(10, 26),
                y=slice(64, 192),
                x=slice(64, 192),
            ),
        },
        {
            "format": "OME-Arrow 3D block",
            "path": OME_ARROW_3D_BLOCK_DIR,
            "write": lambda: write_ome_arrow_3d_dataset(
                arrays,
                OME_ARROW_3D_BLOCK_DIR,
                layout="block",
                chunk_shape=block_shape,
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_3D_BLOCK_DIR),
            "read_all": lambda dataset: dataset.read_many(dataset.image_ids),
            "read_plane": lambda dataset: dataset.read_plane(
                dataset.image_ids[0], z=10
            ),
            "read_subvolume": lambda dataset: dataset.read_region(
                dataset.image_ids[0],
                z=slice(10, 26),
                y=slice(64, 192),
                x=slice(64, 192),
            ),
        },
        {
            "format": "OME-Arrow 3D block none",
            "path": OME_ARROW_3D_BLOCK_NONE_DIR,
            "write": lambda: write_ome_arrow_3d_dataset(
                arrays,
                OME_ARROW_3D_BLOCK_NONE_DIR,
                layout="block",
                chunk_shape=block_shape,
                compression=None,
            ),
            "open": lambda: OMEArrowDataset(OME_ARROW_3D_BLOCK_NONE_DIR),
            "read_all": lambda dataset: dataset.read_many(dataset.image_ids),
            "read_plane": lambda dataset: dataset.read_plane(
                dataset.image_ids[0], z=10
            ),
            "read_subvolume": lambda dataset: dataset.read_region(
                dataset.image_ids[0],
                z=slice(10, 26),
                y=slice(64, 192),
                x=slice(64, 192),
            ),
        },
        {
            "format": "Zarr 3D whole chunks",
            "path": OME_ZARR_3D_WHOLE_DIR,
            "write": lambda: write_zarr_3d_dataset(
                image_ids, arrays, OME_ZARR_3D_WHOLE_DIR, chunks=arrays[0].shape
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_zarr_3d_all(OME_ZARR_3D_WHOLE_DIR),
            "read_plane": lambda _state: read_zarr_3d_plane(OME_ZARR_3D_WHOLE_DIR),
            "read_subvolume": lambda _state: read_zarr_3d_subvolume(
                OME_ZARR_3D_WHOLE_DIR
            ),
        },
        {
            "format": "Zarr 3D block chunks",
            "path": OME_ZARR_3D_BLOCK_DIR,
            "write": lambda: write_zarr_3d_dataset(
                image_ids, arrays, OME_ZARR_3D_BLOCK_DIR, chunks=block_shape
            ),
            "open": lambda: None,
            "read_all": lambda _state: read_zarr_3d_all(OME_ZARR_3D_BLOCK_DIR),
            "read_plane": lambda _state: read_zarr_3d_plane(OME_ZARR_3D_BLOCK_DIR),
            "read_subvolume": lambda _state: read_zarr_3d_subvolume(
                OME_ZARR_3D_BLOCK_DIR
            ),
        },
        {
            "format": "TIFF 3D source",
            "path": dataset_dir / "images",
            "write": None,
            "open": lambda: None,
            "read_all": lambda _state: read_tiff_3d_all(source_paths),
            "read_plane": lambda _state: read_tiff_3d_plane(source_paths),
            "read_subvolume": lambda _state: read_tiff_3d_subvolume(source_paths),
        },
    ]

    summaries = []
    runs = []
    for cfg in configs:
        write_times = []
        print(f"[3d format start] {cfg['format']}", flush=True)
        if cfg["write"] is not None:
            for run_idx in range(FORMAT_WRITE_REPEATS):
                drop_path(cfg["path"])
                gc.collect()
                t0 = time.perf_counter()
                cfg["write"]()
                seconds = time.perf_counter() - t0
                write_times.append(seconds)
                print(
                    f"[3d write] {cfg['format']} run {run_idx + 1}/"
                    f"{FORMAT_WRITE_REPEATS}: {seconds:.4f}s",
                    flush=True,
                )
        else:
            write_times = [0.0]

        size_mb = path_size_bytes(cfg["path"]) / (1024 * 1024)
        state = cfg["open"]()
        measurements = {
            "read_all": timed_callable(lambda: cfg["read_all"](state)),
            "read_plane": timed_callable(lambda: cfg["read_plane"](state)),
            "read_subvolume": timed_callable(lambda: cfg["read_subvolume"](state)),
        }
        for kind, values in {"write": write_times, **measurements}.items():
            for run_idx, seconds in enumerate(values):
                runs.append(
                    {
                        "format": cfg["format"],
                        "kind": kind,
                        "run": run_idx,
                        "seconds": seconds,
                    }
                )
        summaries.append(
            {
                "format": cfg["format"],
                "benchmark_version": BENCHMARK_VERSION,
                "write_avg_s": float(np.mean(write_times)),
                "read_all_avg_s": float(np.mean(measurements["read_all"])),
                "read_plane_avg_s": float(np.mean(measurements["read_plane"])),
                "read_subvolume_avg_s": float(np.mean(measurements["read_subvolume"])),
                "size_mb": size_mb,
                "image_count": len(arrays),
                "shape": "TCZYX=" + "x".join(str(v) for v in arrays[0].shape),
                "version": f"ome-arrow {OME_ARROW_VERSION}",
            }
        )
        print(f"[3d format end] {cfg['format']}", flush=True)

    summary = pd.DataFrame(summaries)
    runs_df = pd.DataFrame(runs)
    summary.to_parquet(SUMMARY_3D_PARQUET, index=False)
    runs_df.to_parquet(RUNS_3D_PARQUET, index=False)
    return summary, runs_df


def timed_query(sql: str, repeats: int = TIMING_REPEATS) -> list[float]:
    times = []
    for _ in range(repeats):
        gc.collect()
        with duckdb.connect() as con:
            t0 = time.perf_counter()
            _ = con.execute(sql).to_arrow_table()
            times.append(time.perf_counter() - t0)
    return times


def timed_reader(reader, repeats: int = TIMING_REPEATS) -> list[float]:
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        _ = reader()
        times.append(time.perf_counter() - t0)
    return times


def run_join_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_level_path = BENCH_DIR / "image_level_observations.parquet"
    operations = [
        {
            "operation": "Path Parquet join, observation scale",
            "mode": "without OME-Arrow",
            "description": "Join repeated feature rows to image file paths.",
            "sql": f"""
                SELECT o.obs_id, o.image_key, o.channel, o.metadata_well,
                       o.metadata_site, p.image_path, p.size_bytes
                FROM read_parquet('{OBSERVATIONS_PATH}') AS o
                JOIN read_parquet('{IMAGE_PATH_INDEX_PATH}') AS p
                USING (image_key)
                ORDER BY o.obs_id
            """,
        },
        {
            "operation": "OA NT Parquet metadata join, observation scale",
            "mode": "with OA NT Parquet",
            "description": "Join repeated feature rows to byte-backed nested OME pixel metadata.",
            "sql": f"""
                SELECT o.obs_id, o.image_key, o.channel, o.metadata_well,
                       img.ome_image.pixels_meta.size_x AS size_x,
                       img.ome_image.pixels_meta.size_y AS size_y,
                       img.ome_image.pixels_meta.type AS pixel_type
                FROM read_parquet('{OBSERVATIONS_PATH}') AS o
                JOIN read_parquet('{IMAGE_OME_BYTE_INDEX_PATH}') AS img
                USING (image_key)
                ORDER BY o.obs_id
            """,
        },
        {
            "operation": "Path Parquet join, image payload scale",
            "mode": "without OME-Arrow",
            "description": "Join one profile-image relationship per real image path.",
            "sql": f"""
                SELECT o.source_row_id, o.image_key, o.channel, p.image_path,
                       p.height, p.width, p.dtype
                FROM read_parquet('{image_level_path}') AS o
                JOIN read_parquet('{IMAGE_PATH_INDEX_PATH}') AS p
                USING (image_key)
                ORDER BY o.source_row_id, o.channel
            """,
        },
        {
            "operation": "OA NT Parquet payload join, image payload scale",
            "mode": "with OA NT Parquet",
            "description": (
                "Join one profile-image relationship per byte-backed nested OME image."
            ),
            "sql": f"""
                SELECT o.source_row_id, o.image_key, o.channel, img.ome_image
                FROM read_parquet('{image_level_path}') AS o
                JOIN read_parquet('{IMAGE_OME_BYTE_INDEX_PATH}') AS img
                USING (image_key)
                ORDER BY o.source_row_id, o.channel
            """,
        },
        {
            "operation": "Converted OA NT Parquet metadata scan",
            "mode": "converted OA NT Parquet",
            "description": (
                "Read nested OME metadata from byte-backed converted NF1 "
                "profile-image table."
            ),
            "sql": f"""
                SELECT source_row_id, image_key, channel, metadata_well,
                       ome_image.pixels_meta.size_x AS size_x,
                       ome_image.pixels_meta.size_y AS size_y,
                       ome_image.pixels_meta.type AS pixel_type
                FROM read_parquet('{CONVERTED_OME_BYTE_DATASET_PATH}')
                ORDER BY source_row_id, channel
            """,
        },
        {
            "operation": "OME-TIFF metadata scan",
            "mode": "converted OME-TIFF",
            "description": "Read OME metadata from converted OME-TIFF files.",
            "reader": read_ome_tiff_metadata,
        },
        {
            "operation": "OME-Zarr metadata scan",
            "mode": "converted OME-Zarr",
            "description": "Read OME metadata from converted OME-Zarr groups.",
            "reader": read_ome_zarr_metadata,
        },
        {
            "operation": "Converted OA NT Parquet payload scan",
            "mode": "converted OA NT Parquet",
            "description": (
                "Read byte-backed inline images from converted NF1 profile-image table."
            ),
            "sql": f"""
                SELECT source_row_id, image_key, channel, ome_image
                FROM read_parquet('{CONVERTED_OME_BYTE_DATASET_PATH}')
                ORDER BY source_row_id, channel
            """,
        },
        {
            "operation": "OME-TIFF payload scan",
            "mode": "converted OME-TIFF",
            "description": "Read pixel payloads from converted OME-TIFF files.",
            "reader": read_ome_tiff_payload,
        },
        {
            "operation": "OME-Zarr payload scan",
            "mode": "converted OME-Zarr",
            "description": "Read pixel payloads from converted OME-Zarr groups.",
            "reader": read_ome_zarr_payload,
        },
    ]

    summaries = []
    runs = []
    for op in operations:
        if "sql" in op:
            with duckdb.connect() as con:
                sample = con.execute(op["sql"]).to_arrow_table()
            times = timed_query(op["sql"])
        else:
            sample = op["reader"]()
            times = timed_reader(op["reader"])
        for run, seconds in enumerate(times):
            runs.append(
                {
                    "operation": op["operation"],
                    "mode": op["mode"],
                    "run": run,
                    "seconds": seconds,
                }
            )
        summaries.append(
            {
                "operation": op["operation"],
                "mode": op["mode"],
                "benchmark_version": BENCHMARK_VERSION,
                "description": op["description"],
                "rows_returned": sample.num_rows,
                "columns_returned": sample.num_columns,
                "result_size_mb": sample.nbytes / (1024 * 1024),
                "avg_seconds": float(np.mean(times)),
                "std_seconds": float(np.std(times)),
                "min_seconds": float(np.min(times)),
            }
        )
        print(
            f"{op['operation']}: {np.mean(times):.4f}s avg over {len(times)} runs",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    runs_df = pd.DataFrame(runs)
    summary_df.to_parquet(SUMMARY_PARQUET, index=False)
    runs_df.to_parquet(RUNS_PARQUET, index=False)
    return summary_df, runs_df


def label_bars(ax, bars, fmt="%.3f") -> None:
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    min_labeled_value = x_span * 0.07
    for bar in bars:
        value = bar.get_width()
        if value <= 0 or value < min_labeled_value:
            continue
        ax.text(
            value / 2,
            bar.get_y() + bar.get_height() / 2,
            fmt % value,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )


def plot_results(summary_df: pd.DataFrame, table_summary: pd.DataFrame) -> None:
    color_map = {
        "without OME-Arrow": "#3D7FBF",
        "with OA NT": "#6F4E37",
        "with OA NT Parquet": "#6F4E37",
        "converted OA NT": "#6F4E37",
        "converted OA NT Parquet": "#6F4E37",
        "converted OME-TIFF": "#C86A1B",
        "converted OME-Zarr": "#6B7280",
    }
    # Plain-english bar labels — no abbreviation key needed.
    operation_labels = {
        "Path Parquet join, observation scale": "Path join\nprofile rows",
        "OA NT Parquet metadata join, observation scale": "OA meta join\nprofile rows",
        "Path Parquet join, image payload scale": "Path join\nimage rows",
        "OA NT Parquet payload join, image payload scale": "OA pixel join\nimage rows",
        "Converted OA NT Parquet metadata scan": "OA meta read\n(converted)",
        "OME-TIFF metadata scan": "TIFF meta\nread",
        "OME-Zarr metadata scan": "Zarr meta\nread",
        "Converted OA NT Parquet payload scan": "OA pixel read\n(converted)",
        "OME-TIFF payload scan": "TIFF pixel\nread",
        "OME-Zarr payload scan": "Zarr pixel\nread",
    }
    table_labels = {
        "observations": "profile rows",
        "image path index": "path index",
        "OA NT image index": "OA NT index",
        "OA DS typed chunk dataset": "OA DS chunks",
        "OA DS typed chunk dataset, uncompressed": "OA DS raw",
        "converted NF1 OA NT profile-image table": "joined OA NT",
        "converted OME-TIFF images": "OME-TIFF",
        "converted OME-Zarr images": "OME-Zarr",
    }
    table_colors = {
        "observations": "#3D7FBF",
        "image path index": "#3D7FBF",
        "OA NT image index": "#6F4E37",
        "OA DS typed chunk dataset": "#2E7D4F",
        "OA DS typed chunk dataset, uncompressed": "#B23B3B",
        "converted NF1 OA NT profile-image table": "#6F4E37",
        "converted OME-TIFF images": "#C86A1B",
        "converted OME-Zarr images": "#6B7280",
    }
    colors = [color_map.get(mode, "#6B7280") for mode in summary_df["mode"]]
    labels = [operation_labels.get(name, name) for name in summary_df["operation"]]
    x = np.arange(len(summary_df))

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    fig.suptitle(
        "NF1 Cell Painting: profiling joins and image format read times",
        fontsize=14,
    )

    bars = axes[0, 0].bar(x, summary_df["avg_seconds"] * 1000, color=colors)
    axes[0, 0].set_title("Query time (ms)")
    axes[0, 0].set_ylabel("milliseconds")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.5)
    label_vertical_bars(axes[0, 0], bars, "%.1f")

    bars = axes[0, 1].bar(x, summary_df["result_size_mb"], color=colors)
    axes[0, 1].set_title("Result size (MB)")
    axes[0, 1].set_ylabel("MB")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)
    label_vertical_bars(axes[0, 1], bars, "%.2f")

    bars = axes[1, 0].bar(x, summary_df["rows_returned"], color=colors)
    # The row counts differ because joins at "profile rows" scale expand to ~15 k rows
    # (one per repeated observation), while "image rows" scale has one row per real image (15).
    axes[1, 0].set_title(
        "Rows returned per query (log scale)\n"
        "Profile-scale: ~15 k rows (repeated observations); image-scale: 15 rows (one per image)"
    )
    axes[1, 0].set_ylabel("rows")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1, max(summary_df["rows_returned"]) * 3)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.5)
    label_log_bars(axes[1, 0], bars, "%.0f")

    table_plot = table_summary[table_summary["table"] != "OME-IRIS source dataset"]
    table_x = np.arange(len(table_plot))
    bars = axes[1, 1].bar(
        table_x,
        table_plot["size_mb"],
        color=[table_colors.get(name, "#6B7280") for name in table_plot["table"]],
    )
    axes[1, 1].set_title("Benchmark artifact storage (MB)")
    axes[1, 1].set_ylabel("MB")
    axes[1, 1].set_xticks(table_x)
    axes[1, 1].set_xticklabels(
        [table_labels.get(name, name) for name in table_plot["table"]],
        rotation=25,
        ha="right",
        fontsize=9,
    )
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.5)
    label_vertical_bars(axes[1, 1], bars, "%.2f")

    legend_handles = [
        Patch(
            facecolor=color_map["without OME-Arrow"],
            label="Parquet path join (no OME-Arrow)",
        ),
        Patch(
            facecolor=color_map["with OA NT Parquet"],
            label="OA NT Parquet (OME-Arrow nested table)",
        ),
        Patch(facecolor=color_map["converted OME-TIFF"], label="OME-TIFF"),
        Patch(facecolor=color_map["converted OME-Zarr"], label="OME-Zarr"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
        fontsize=10,
        title=OA_KEY_TEXT,
        title_fontsize=8,
    )

    fig.text(0.5, 0.0, SOURCE_NOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(FIGURES_DIR / "compare_ome_iris_joins_detail.png", dpi=150)
    plt.close(fig)


def label_vertical_bars(ax, bars, fmt="%.3f") -> None:
    values = [bar.get_height() for bar in bars]
    if not values:
        return
    y_max = max(values)
    if y_max <= 0:
        return
    ax.set_ylim(0, y_max * 1.15)
    min_labeled_value = y_max * 0.04
    for bar, value in zip(bars, values):
        if value <= 0 or value < min_labeled_value:
            continue
        inset = max(value * 0.06, y_max * 0.01)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - inset,
            fmt % value,
            ha="center",
            va="top",
            color="white",
            fontsize=9,
        )


def label_log_bars(ax, bars, fmt="%.0f") -> None:
    for bar in bars:
        value = bar.get_height()
        if value <= 0:
            continue
        text_y = max(value / 1.6, 1.15)
        color = "white" if value >= 10 else "#333333"
        va = "center" if value >= 10 else "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            fmt % value,
            ha="center",
            va=va,
            color=color,
            fontsize=9,
        )


def plot_format_results(format_summary: pd.DataFrame) -> None:
    metrics = [
        ("write_avg_s", "Write all avg (s)", "seconds"),
        ("read_all_avg_s", "Read all avg (s)", "seconds"),
        ("read_random_avg_s", "Read random avg (s)", "seconds"),
        ("size_mb", "Size (MB)", "MB"),
    ]
    color_map = {
        "OME-Arrow nested bytes table, Parquet": "#6F4E37",
        "OME-Arrow nested bytes table, Vortex": "#2E7D4F",
        "OME-Arrow nested bytes table, Lance": "#8A5A9E",
        "OME-Arrow chunks zstd, NumPy, reopen": "#2E7D4F",
        "OME-Arrow chunks zstd, NumPy, open": "#3D7FBF",
        "OME-Arrow chunks zstd, rows, open": "#5A6B3A",
        "OME-Arrow chunks none, NumPy, open": "#B23B3B",
        "OME-TIFF (dir-per-image)": "#C86A1B",
        "OME-Zarr (dir-per-image)": "#6B7280",
    }
    label_map = {
        "OME-Arrow nested bytes table, Parquet": "OA NT\nParquet",
        "OME-Arrow nested bytes table, Vortex": "OA NT\nVortex",
        "OME-Arrow nested bytes table, Lance": "OA NT\nLance",
        "OME-Arrow chunks zstd, NumPy, reopen": "OA DS\nreopen",
        "OME-Arrow chunks zstd, NumPy, open": "OA DS\nParquet",
        "OME-Arrow chunks zstd, rows, open": "OA DS\nrows",
        "OME-Arrow chunks none, NumPy, open": "OA DS\nnone",
        "OME-TIFF (dir-per-image)": "TIFF",
        "OME-Zarr (dir-per-image)": "OME-Zarr",
    }
    user_facing_formats = [
        "OME-Arrow nested bytes table, Parquet",
        "OME-Arrow nested bytes table, Vortex",
        "OME-Arrow nested bytes table, Lance",
        "OME-Arrow chunks zstd, NumPy, open",
        "OME-TIFF (dir-per-image)",
        "OME-Zarr (dir-per-image)",
    ]
    user_facing_summary = (
        format_summary.set_index("format").loc[user_facing_formats].reset_index()
    )
    _plot_format_bars(
        user_facing_summary,
        metrics,
        color_map,
        label_map,
        "OME-IRIS NF1 converted image formats",
        FIGURES_DIR / "compare_ome_iris_joins_summary.png",
    )


def plot_format_diagnostics(format_summary: pd.DataFrame) -> None:
    metrics = [
        ("write_avg_s", "Write all avg (s)", "seconds"),
        ("read_all_avg_s", "Read all avg (s)", "seconds"),
        ("read_random_avg_s", "Read random avg (s)", "seconds"),
        ("size_mb", "Size (MB)", "MB"),
    ]
    diagnostic_formats = [
        "OME-Arrow chunks zstd, NumPy, reopen",
        "OME-Arrow chunks zstd, NumPy, open",
        "OME-Arrow chunks zstd, rows, open",
        "OME-Arrow chunks none, NumPy, open",
    ]
    color_map = {
        "OME-Arrow chunks zstd, NumPy, reopen": "#2E7D4F",
        "OME-Arrow chunks zstd, NumPy, open": "#3D7FBF",
        "OME-Arrow chunks zstd, rows, open": "#5A6B3A",
        "OME-Arrow chunks none, NumPy, open": "#B23B3B",
    }
    # Labels spell out "OA Dataset" so "DS" does not need decoding.
    label_map = {
        "OME-Arrow chunks zstd, NumPy, reopen": "OA Dataset\nzstd\nreopen each call",
        "OME-Arrow chunks zstd, NumPy, open": "OA Dataset\nzstd\nkeep handle open",
        "OME-Arrow chunks zstd, rows, open": "OA Dataset\nzstd\nchunk-row API",
        "OME-Arrow chunks none, NumPy, open": "OA Dataset\nuncompressed\nkeep handle open",
    }
    diagnostic_summary = (
        format_summary.set_index("format").loc[diagnostic_formats].reset_index()
    )
    _plot_format_bars(
        diagnostic_summary,
        metrics,
        color_map,
        label_map,
        # Title states the observation rather than describing the chart contents.
        "OME-Arrow Dataset: keeping the handle open reduces read latency;\n"
        "skipping compression cuts read time further at the cost of disk space",
        FIGURES_DIR / "compare_ome_iris_joins_diagnostics.png",
        legend_ncol=2,
    )


def _plot_format_bars(
    format_summary: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    color_map: dict[str, str],
    label_map: dict[str, str],
    title: str,
    output_path: Path,
    key_extra: str | None = None,
    legend_right: bool = False,
    legend_ncol: int | None = None,
) -> None:
    colors = [color_map.get(name, "#BAB0AC") for name in format_summary["format"]]
    labels = [label_map.get(name, name) for name in format_summary["format"]]
    x = np.arange(len(format_summary))

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    for ax, (col, ax_title, ylabel) in zip(axes.flat, metrics):
        bars = ax.bar(x, format_summary[col], color=colors)
        ax.set_title(ax_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        label_vertical_bars(ax, bars)

    legend_items = [
        (label_map.get(name, name).replace("\n", " "), color_map.get(name, "#BAB0AC"))
        for name in format_summary["format"]
    ]
    if legend_right:
        legend_title = OA_KEY_TEXT + (f"\n{key_extra}" if key_extra else "")
        handles = [Patch(facecolor=color, label=label) for label, color in legend_items]
        fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            frameon=True,
            fontsize=9,
            title=legend_title,
            title_fontsize=8,
        )
        fig.text(0.5, 0.01, SOURCE_NOTE, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 0.78, 0.92))
    else:
        add_colored_figure_key(fig, legend_items, extra=key_extra, ncol=legend_ncol)
        fig.text(0.5, 0.01, SOURCE_NOTE, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_colored_figure_key(
    fig,
    items: list[tuple[str, str]],
    extra: str | None = None,
    ncol: int | None = None,
) -> None:
    title = OA_KEY_TEXT
    if extra:
        title = f"{title}\n{extra}"
    handles = [Patch(facecolor=color, label=label) for label, color in items]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=ncol if ncol is not None else min(4, len(items)),
        title=title,
        frameon=True,
    )



def plot_3d_results(summary_3d: pd.DataFrame) -> None:
    metrics = [
        ("write_avg_s", "Write all avg (s)", "seconds"),
        ("read_all_avg_s", "Read all volumes avg (s)", "seconds"),
        ("read_plane_avg_s", "Read one z-plane avg (s)", "seconds"),
        ("read_subvolume_avg_s", "Read 16x128x128 region avg (s)", "seconds"),
        ("size_mb", "On-disk size (MB)", "MB"),
    ]
    color_map = {
        "OME-Arrow 3D z-plane": "#2E7D4F",
        "OME-Arrow 3D block": "#3D7FBF",
        "OME-Arrow 3D block none": "#B23B3B",
        "Zarr 3D whole chunks": "#8A5A9E",
        "Zarr 3D block chunks": "#6B7280",
        "TIFF 3D source": "#C86A1B",
    }
    label_map = {
        "OME-Arrow 3D z-plane": "OA\nplane",
        "OME-Arrow 3D block": "OA\nblock",
        "OME-Arrow 3D block none": "OA\nraw",
        "Zarr 3D whole chunks": "OME-\nZarr\nwhole",
        "Zarr 3D block chunks": "OME-\nZarr\nblock",
        "TIFF 3D source": "TIFF\nsource",
    }
    colors = [color_map.get(name, "#BAB0AC") for name in summary_3d["format"]]
    labels = [label_map.get(name, name) for name in summary_3d["format"]]
    x = np.arange(len(summary_3d))
    image_count = int(summary_3d["image_count"].iloc[0])
    shape = str(summary_3d["shape"].iloc[0]).replace("TCZYX=", "TCZYX ")

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"3D nuclei segmentation benchmark: {image_count} source TIFF volumes, {shape}"
    )
    for ax, (col, title, ylabel) in zip(axes.flat, metrics):
        bars = ax.bar(x, summary_3d[col], color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        label_vertical_bars(ax, bars)
        if col == "write_avg_s":
            tiff_rows = np.flatnonzero(
                summary_3d["format"].to_numpy() == "TIFF 3D source"
            )
            if len(tiff_rows):
                y_max = max(summary_3d[col].max(), 0.001)
                ax.text(
                    tiff_rows[0],
                    y_max * 0.05,
                    "source\nnot\nrewritten",
                    ha="center",
                    va="bottom",
                    color="#333333",
                    fontsize=8,
                )

    axes.flat[-1].axis("off")
    legend_items = [
        (
            "OA plane = OME-Arrow z-plane chunks, Zstd",
            color_map["OME-Arrow 3D z-plane"],
        ),
        (
            "OA block = OME-Arrow 16x128x128 chunks, Zstd",
            color_map["OME-Arrow 3D block"],
        ),
        (
            "OA raw = OME-Arrow 16x128x128 chunks, uncompressed",
            color_map["OME-Arrow 3D block none"],
        ),
        (
            "OME-Zarr whole = one chunk per volume",
            color_map["Zarr 3D whole chunks"],
        ),
        (
            "OME-Zarr block = 16x128x128 chunks",
            color_map["Zarr 3D block chunks"],
        ),
        (
            "TIFF source = original source files, write not measured",
            color_map["TIFF 3D source"],
        ),
    ]
    handles = [Patch(facecolor=color, label=label) for label, color in legend_items]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.02,
        "A z-plane is one Z slice. The region read is z=10:26, y=64:192, x=64:192 from one volume. "
        + SOURCE_NOTE,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.94))
    fig.savefig(FIGURES_DIR / "compare_ome_iris_3d_summary.png", dpi=150)
    plt.close(fig)


if RUN_BENCHMARKS:
    dataset_directory = fetch_dataset()
    dataset_3d_directory = fetch_dataset(DATASET_3D_ID, SOURCE_3D_IDENTIFIER)
    tables_df = write_benchmark_tables(dataset_directory)
    format_results_df, format_timings_df = run_format_benchmarks(dataset_directory)
    results_3d_df, timings_3d_df = run_3d_benchmarks(dataset_3d_directory)
    results_df, timings_df = run_join_benchmarks()
else:
    results_df = pd.read_parquet(SUMMARY_PARQUET)
    timings_df = pd.read_parquet(RUNS_PARQUET)
    tables_df = pd.read_parquet(TABLE_SUMMARY_PARQUET)
    format_results_df = pd.read_parquet(FORMAT_SUMMARY_PARQUET)
    format_timings_df = pd.read_parquet(FORMAT_RUNS_PARQUET)
    results_3d_df = pd.read_parquet(SUMMARY_3D_PARQUET)
    timings_3d_df = pd.read_parquet(RUNS_3D_PARQUET)
    print(
        "Loaded existing OME-IRIS join benchmark data from",
        SUMMARY_PARQUET,
        RUNS_PARQUET,
        "and",
        TABLE_SUMMARY_PARQUET,
    )

print(results_df)
print(timings_df)
print(tables_df)
print(format_results_df)
print(format_timings_df)
print(results_3d_df)
print(timings_3d_df)
plot_format_results(format_results_df)
plot_format_diagnostics(format_results_df)
plot_3d_results(results_3d_df)
plot_results(results_df, tables_df)
