"""
@file: repack_gap_hdf5.py
@description: Repack the original gzip-compressed, chunked GAP HDF5 files into a contiguous, uncompressed layout for faster reads.
              Preserve the original files and the GAP-3/GAP-5 and train/val/test directory structure,
              and record conversion manifests, sampled equality checks, and source/output read benchmarks.
@author: Changxin Ye
@created: 2026-07-31
@version: 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = ("GAP-3", "GAP-5")
SPLITS = ("train", "val", "test")

# =============================================================================
# Configuration for running this script directly
# When running this file directly in PyCharm/VSCode, adjust the DEFAULT_* settings below.
# Command-line arguments remain available but are optional.
# =============================================================================

# Root directory of the original GAP data, containing GAP-3 and GAP-5.
DEFAULT_SOURCE_ROOT = SCRIPT_DIR.parent / "GAP_Download" / "GAP"

# Output root. With SCRIPT_DIR, converted files are saved in this GAP_fast directory.
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR

# For an initial trial, select only GAP-3; later use ("GAP-5",) or ("GAP-3", "GAP-5").
# Keep the trailing comma when specifying a single-element tuple.
DEFAULT_DATASETS = ("GAP-3", "GAP-5")

# For an initial trial, select only train; later use ("val", "test") or all three splits.
DEFAULT_SPLITS = ("train", "val", "test")

# Number of puzzles per split for sampled pixel-by-pixel equality checks; 0 disables sampling.
DEFAULT_VERIFY_SAMPLES = 3

# Number of sampled puzzle reads per split for source/output speed comparisons; 0 disables benchmarking.
DEFAULT_BENCHMARK_SAMPLES = 3

# Fixed seed for verification and benchmark sampling.
DEFAULT_SEED = 42

# False: stop if output files exist to prevent accidental overwrites; set True to rerun intentionally.
DEFAULT_OVERWRITE = False

# False: perform conversion; True: check paths and disk space without writing data files.
DEFAULT_DRY_RUN = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Repack GAP HDF5 into a contiguous, uncompressed layout for faster random single-puzzle reads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        type=Path,
        help="Original GAP root directory containing GAP-3 and GAP-5",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help="Repacked GAP root directory; defaults to the current GAP_fast directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DATASETS,
        help="Datasets to repack; GAP-3 or GAP-5 can be selected individually",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=SPLITS,
        help="Dataset splits to repack",
    )
    parser.add_argument(
        "--verify-samples",
        default=DEFAULT_VERIFY_SAMPLES,
        type=int,
        help="Number of sampled puzzles per split for element-wise equality checks; 0 disables sampling",
    )
    parser.add_argument(
        "--benchmark-samples",
        default=DEFAULT_BENCHMARK_SAMPLES,
        type=int,
        help="Number of sampled puzzle reads per split for source/output speed comparisons; 0 disables benchmarking",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="Fixed seed for verification and benchmark sampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=DEFAULT_OVERWRITE,
        help="Allow overwriting existing output files or .tmp files left by an interrupted run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Check source data, output paths, and estimated disk usage without performing conversion",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def human_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def copy_attributes(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def atomic_json_dump(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path, Path]]:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("source-root and output-root must differ; overwriting the original GAP data is prohibited")

    jobs: list[tuple[str, str, Path, Path]] = []
    for dataset in dict.fromkeys(args.datasets):
        for split in dict.fromkeys(args.splits):
            source_dir = source_root / dataset / split
            output_dir = output_root / dataset / split
            puzzles_path = source_dir / "puzzles.h5"
            labels_path = source_dir / "labels_indices.h5"
            if not puzzles_path.is_file():
                raise FileNotFoundError(f"Original puzzle file not found: {puzzles_path}")
            if not labels_path.is_file():
                raise FileNotFoundError(f"Original label file not found: {labels_path}")
            jobs.append((dataset, split, source_dir, output_dir))
    return jobs


def inspect_source(source_path: Path) -> dict:
    with h5py.File(source_path, "r") as source_file:
        if "puzzles" not in source_file:
            raise KeyError(f"{source_path} does not contain a puzzles dataset")
        dataset = source_file["puzzles"]
        if dataset.ndim != 5:
            raise ValueError(f"puzzles must have five dimensions [N,P,H,W,C]; got {dataset.shape}")
        if dataset.dtype != np.uint8:
            raise ValueError(f"puzzles must have dtype uint8; got {dataset.dtype}")
        return {
            "shape": tuple(int(value) for value in dataset.shape),
            "dtype": str(dataset.dtype),
            "chunks": None if dataset.chunks is None else tuple(int(value) for value in dataset.chunks),
            "compression": dataset.compression,
            "compression_opts": dataset.compression_opts,
            "uncompressed_bytes": int(dataset.size * dataset.dtype.itemsize),
            "source_file_bytes": int(source_path.stat().st_size),
        }


def validate_args(args: argparse.Namespace, jobs: list[tuple[str, str, Path, Path]]) -> list[dict]:
    if args.verify_samples < 0 or args.benchmark_samples < 0:
        raise ValueError("verify-samples and benchmark-samples must be nonnegative")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")

    reports: list[dict] = []
    required_bytes = 0
    for dataset, split, source_dir, output_dir in jobs:
        source_info = inspect_source(source_dir / "puzzles.h5")
        required_bytes += int(source_info["uncompressed_bytes"])
        protected_paths = (
            output_dir / "puzzles.h5",
            output_dir / "puzzles.h5.tmp",
            output_dir / "labels_indices.h5",
            output_dir / "labels_indices.h5.tmp",
            output_dir / "metadata.json",
            output_dir / "metadata_original.json",
            output_dir / "repack_info.json",
        )
        existing = [path for path in protected_paths if path.exists()]
        if not args.overwrite and existing:
            raise FileExistsError(
                f"Output directory already contains conversion artifacts: {existing}. Add --overwrite only if you intend to rerun"
            )
        reports.append(
            {
                "dataset": dataset,
                "split": split,
                "source_dir": source_dir,
                "output_dir": output_dir,
                "source": source_info,
            }
        )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    safety_bytes = max(2 * 1024**3, int(required_bytes * 0.05))
    if free_bytes < required_bytes + safety_bytes:
        raise OSError(
            f"Insufficient disk space: estimated output data requires {human_bytes(required_bytes)}, "
            f"including the safety margin requires {human_bytes(required_bytes + safety_bytes)}, "
            f"currently available {human_bytes(free_bytes)}"
        )
    print(f"Original root directory: {args.source_root.expanduser().resolve()}")
    print(f"Output root directory: {output_root}")
    print(f"Estimated additional uncompressed data: {human_bytes(required_bytes)}")
    print(f"Available disk space: {human_bytes(free_bytes)}")
    for report in reports:
        source = report["source"]
        print(
            f"  {report['dataset']}/{report['split']}: shape={source['shape']}, "
            f"source chunks={source['chunks']}, source compression={source['compression']}, "
            f"output layout=contiguous/uncompressed, estimated {human_bytes(source['uncompressed_bytes'])}"
        )
    return reports


def progress_message(done: int, total: int, started: float) -> None:
    interval = max(1, total // 20)
    if done == 1 or done == total or done % interval == 0:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"    Blocks {done}/{total} ({done / total:.1%}), "
            f"elapsed {elapsed / 60:.1f} min, estimated remaining {remaining / 60:.1f} min",
            flush=True,
        )


def repack_puzzles(source_path: Path, temporary_path: Path) -> dict:
    started = time.perf_counter()
    with h5py.File(source_path, "r") as source_file:
        source = source_file["puzzles"]
        if source.chunks is None:
            sample_block = min(64, int(source.shape[0]))
            piece_block = 1
        else:
            # Align with source chunk boundaries in the first two dimensions to avoid repeated gzip decompression.
            sample_block = int(source.chunks[0])
            piece_block = int(source.chunks[1])

        sample_ranges = [
            (start, min(start + sample_block, int(source.shape[0])))
            for start in range(0, int(source.shape[0]), sample_block)
        ]
        piece_ranges = [
            (start, min(start + piece_block, int(source.shape[1])))
            for start in range(0, int(source.shape[1]), piece_block)
        ]
        total_blocks = len(sample_ranges) * len(piece_ranges)

        with h5py.File(temporary_path, "w", libver="latest") as destination_file:
            copy_attributes(source_file, destination_file)
            # Use chunks=None and no compression to create a contiguous, uncompressed HDF5 dataset.
            destination = destination_file.create_dataset(
                "puzzles",
                shape=source.shape,
                dtype=source.dtype,
                chunks=None,
            )
            copy_attributes(source, destination)

            done = 0
            for sample_start, sample_end in sample_ranges:
                for piece_start, piece_end in piece_ranges:
                    block = source[
                        sample_start:sample_end,
                        piece_start:piece_end,
                        :,
                        :,
                        :,
                    ]
                    destination[
                        sample_start:sample_end,
                        piece_start:piece_end,
                        :,
                        :,
                        :,
                    ] = block
                    del block
                    done += 1
                    progress_message(done, total_blocks, started)
            destination_file.flush()

    return {
        "copy_seconds": float(time.perf_counter() - started),
        "source_aligned_sample_block": int(sample_block),
        "source_aligned_piece_block": int(piece_block),
        "copied_blocks": int(total_blocks),
    }


def repack_labels(source_path: Path, temporary_path: Path) -> dict:
    with h5py.File(source_path, "r") as source_file:
        if "labels" not in source_file:
            raise KeyError(f"{source_path} does not contain a labels dataset")
        source = source_file["labels"]
        labels = np.asarray(source[:])
        with h5py.File(temporary_path, "w", libver="latest") as destination_file:
            copy_attributes(source_file, destination_file)
            destination = destination_file.create_dataset(
                "labels",
                data=labels,
                dtype=source.dtype,
                chunks=None,
            )
            copy_attributes(source, destination)
            destination_file.flush()
    return {
        "shape": [int(value) for value in labels.shape],
        "dtype": str(labels.dtype),
        "destination_chunks": None,
        "destination_compression": None,
    }


def verification_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0:
        return []
    desired = min(length, count)
    selected: set[int] = set()
    for index in (0, length // 2, length - 1):
        if len(selected) >= desired:
            break
        selected.add(index)
    rng = np.random.default_rng(seed)
    while len(selected) < desired:
        selected.add(int(rng.integers(0, length)))
    return sorted(selected)


def verify_puzzles(source_path: Path, destination_path: Path, indices: list[int]) -> None:
    with h5py.File(source_path, "r") as source_file, h5py.File(destination_path, "r") as destination_file:
        source = source_file["puzzles"]
        destination = destination_file["puzzles"]
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise AssertionError("Source and output puzzles have different shapes or dtypes")
        if destination.chunks is not None or destination.compression is not None:
            raise AssertionError(
                f"Output puzzles do not use a contiguous, uncompressed layout: chunks={destination.chunks}, "
                f"compression={destination.compression}"
            )
        for index in indices:
            if not np.array_equal(source[index], destination[index]):
                raise AssertionError(f"puzzles at sample {index} differ from the original data")


def verify_labels(source_path: Path, destination_path: Path) -> None:
    with h5py.File(source_path, "r") as source_file, h5py.File(destination_path, "r") as destination_file:
        source = source_file["labels"]
        destination = destination_file["labels"]
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise AssertionError("Source and output labels have different shapes or dtypes")
        if not np.array_equal(source[:], destination[:]):
            raise AssertionError("Source and output label contents differ")


def benchmark_random_reads(path: Path, indices: list[int]) -> dict:
    if not indices:
        return {"samples": 0, "seconds": 0.0, "seconds_per_sample": 0.0}
    started = time.perf_counter()
    returned_bytes = 0
    with h5py.File(path, "r") as file:
        dataset = file["puzzles"]
        for index in indices:
            value = np.asarray(dataset[index])
            returned_bytes += int(value.nbytes)
    seconds = time.perf_counter() - started
    return {
        "samples": int(len(indices)),
        "seconds": float(seconds),
        "seconds_per_sample": float(seconds / len(indices)),
        "returned_bytes": int(returned_bytes),
    }


def write_metadata(source_dir: Path, output_dir: Path, source_info: dict) -> None:
    source_path = source_dir / "metadata.json"
    if not source_path.is_file():
        return
    original_path = output_dir / "metadata_original.json"
    shutil.copy2(source_path, original_path)
    metadata = json.loads(source_path.read_text(encoding="utf-8"))
    metadata["source_storage"] = {
        "chunks": source_info["chunks"],
        "compression": source_info["compression"],
        "compression_opts": source_info["compression_opts"],
    }
    metadata["compression"] = None
    metadata["compression_level"] = None
    metadata["storage_layout"] = "contiguous"
    metadata["repacked_for_random_sample_training"] = True
    atomic_json_dump(metadata, output_dir / "metadata.json")


def convert_job(
    dataset_name: str,
    split: str,
    source_dir: Path,
    output_dir: Path,
    source_info: dict,
    args: argparse.Namespace,
    job_index: int,
) -> None:
    print(f"\n[{dataset_name}/{split}] Starting conversion")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_puzzles = source_dir / "puzzles.h5"
    source_labels = source_dir / "labels_indices.h5"
    destination_puzzles = output_dir / "puzzles.h5"
    destination_labels = output_dir / "labels_indices.h5"
    temporary_puzzles = output_dir / "puzzles.h5.tmp"
    temporary_labels = output_dir / "labels_indices.h5.tmp"

    if args.overwrite:
        for path in (
            destination_puzzles,
            destination_labels,
            temporary_puzzles,
            temporary_labels,
        ):
            if path.exists():
                path.unlink()

    copy_stats = repack_puzzles(source_puzzles, temporary_puzzles)
    label_stats = repack_labels(source_labels, temporary_labels)

    verify_indices = verification_indices(
        int(source_info["shape"][0]),
        args.verify_samples,
        args.seed + job_index,
    )
    verify_puzzles(source_puzzles, temporary_puzzles, verify_indices)
    verify_labels(source_labels, temporary_labels)

    os.replace(temporary_puzzles, destination_puzzles)
    os.replace(temporary_labels, destination_labels)
    write_metadata(source_dir, output_dir, source_info)

    benchmark_indices = verification_indices(
        int(source_info["shape"][0]),
        args.benchmark_samples,
        args.seed + 1000 + job_index,
    )
    old_benchmark = benchmark_random_reads(source_puzzles, benchmark_indices)
    new_benchmark = benchmark_random_reads(destination_puzzles, benchmark_indices)
    old_speed = float(old_benchmark["seconds_per_sample"])
    new_speed = float(new_benchmark["seconds_per_sample"])
    speedup = old_speed / new_speed if new_speed > 0 else float("inf")

    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "split": split,
        "source_directory": str(source_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "transformation": {
            "description": "Preserve array contents and labels while rewriting puzzles/labels into a contiguous, uncompressed HDF5 layout",
            "puzzles_shape": list(source_info["shape"]),
            "puzzles_dtype": source_info["dtype"],
            "source_chunks": source_info["chunks"],
            "source_compression": source_info["compression"],
            "source_compression_opts": source_info["compression_opts"],
            "destination_chunks": None,
            "destination_compression": None,
            "copy_strategy": "aligned_source_sample_and_piece_chunk_slabs",
        },
        "copy": copy_stats,
        "labels": label_stats,
        "verification": {
            "status": "passed",
            "sample_indices": verify_indices,
            "full_label_comparison": True,
        },
        "benchmark_indices": benchmark_indices,
        "source_random_read": old_benchmark,
        "destination_random_read": new_benchmark,
        "random_read_speedup": float(speedup),
        "source_file_bytes": int(source_info["source_file_bytes"]),
        "destination_file_bytes": int(destination_puzzles.stat().st_size),
    }
    atomic_json_dump(manifest, output_dir / "repack_info.json")
    print(
        f"[{dataset_name}/{split}] Completed: conversion {copy_stats['copy_seconds'] / 60:.1f} min, "
        f"output size {human_bytes(destination_puzzles.stat().st_size)}, "
        f"random reads {old_speed:.4f}s/puzzle -> {new_speed:.4f}s/puzzle, "
        f"speedup {speedup:.1f}x"
    )


def main() -> None:
    args = parse_args()
    jobs = selected_jobs(args)
    reports = validate_args(args, jobs)
    if args.dry_run:
        print("\ndry-run completed: no data was written.")
        return

    total_started = time.perf_counter()
    for job_index, (job, report) in enumerate(zip(jobs, reports)):
        dataset_name, split, source_dir, output_dir = job
        convert_job(
            dataset_name,
            split,
            source_dir,
            output_dir,
            report["source"],
            args,
            job_index,
        )
    elapsed = time.perf_counter() - total_started
    print(f"\nAll conversions completed in {elapsed / 60:.1f} min.")
    print(f"For training, use: --data-root {args.output_root.expanduser().resolve()}")


if __name__ == "__main__":
    main()
