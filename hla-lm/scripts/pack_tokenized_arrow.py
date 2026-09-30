#!/usr/bin/env python
import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKED_DATASET_PATH = REPO_ROOT / "lit_gpt" / "packed_dataset.py"
spec = importlib.util.spec_from_file_location("packed_dataset", PACKED_DATASET_PATH)
packed_dataset = importlib.util.module_from_spec(spec)
sys.modules["packed_dataset"] = packed_dataset
assert spec.loader is not None
spec.loader.exec_module(packed_dataset)
PackedDatasetBuilder = packed_dataset.PackedDatasetBuilder


def pack_split(files, outdir, prefix, chunk_size, max_chunks, eos_token_id, vocab_size, progress_chunks):
    builder = PackedDatasetBuilder(
        outdir=outdir,
        prefix=prefix,
        chunk_size=chunk_size,
        sep_token=eos_token_id,
        vocab_size=vocab_size,
    )
    n_docs = 0
    n_tokens = 0
    next_report = progress_chunks
    for path in files:
        dataset = Dataset.from_file(path)
        if "input_ids" not in dataset.column_names:
            print(f"Skipping {path}: no input_ids column")
            continue
        for row in dataset:
            ids = row["input_ids"]
            if not ids:
                continue
            arr = np.asarray(ids + [eos_token_id], dtype=builder.dtype)
            builder.add_array(arr)
            n_docs += 1
            n_tokens += len(ids) + 1
            if progress_chunks > 0 and len(builder.filenames) >= next_report:
                print(
                    f"{prefix}: wrote {len(builder.filenames)} chunks "
                    f"from {n_docs} docs / {n_tokens} tokens"
                )
                next_report += progress_chunks
            if max_chunks > 0 and len(builder.filenames) >= max_chunks:
                return len(builder.filenames), n_docs, n_tokens
    if builder._idx > 0:
        builder.write_reminder()
    return len(builder.filenames), n_docs, n_tokens


def arrow_files(root):
    return sorted(glob.glob(os.path.join(root, "*.arrow")))


def files_with_input_ids(files):
    valid = []
    for path in files:
        dataset = Dataset.from_file(path)
        if "input_ids" in dataset.column_names:
            valid.append(path)
        else:
            print(f"Skipping {path}: no input_ids column")
    return valid


def prepare_output_dir(outdir, prefixes, overwrite):
    outdir.mkdir(parents=True, exist_ok=True)
    existing = []
    for prefix in prefixes:
        existing.extend(outdir.glob(f"{prefix}_*.bin"))
    if not existing:
        return
    if not overwrite:
        existing_names = ", ".join(path.name for path in existing[:5])
        raise RuntimeError(
            f"{outdir} already contains packed chunks ({existing_names}). "
            "Pass --overwrite to replace chunks with the selected prefixes."
        )
    for path in existing:
        path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--validation_dir", default="")
    parser.add_argument("--train_prefix", default="train_slim")
    parser.add_argument("--validation_prefix", default="validation")
    parser.add_argument("--chunk_size", type=int, default=262144)
    parser.add_argument("--max_train_chunks", type=int, default=16)
    parser.add_argument("--max_validation_chunks", type=int, default=8)
    parser.add_argument("--holdout_validation_files", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--progress_chunks", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    prepare_output_dir(outdir, [args.train_prefix, args.validation_prefix], args.overwrite)

    train_files = arrow_files(args.train_dir)
    if not train_files:
        raise RuntimeError(f"No arrow files found in {args.train_dir}")

    if args.validation_dir:
        train_files = files_with_input_ids(train_files)
        val_files = files_with_input_ids(arrow_files(args.validation_dir))
        if not val_files:
            raise RuntimeError(f"No validation arrow files with input_ids found in {args.validation_dir}")
    else:
        all_files = files_with_input_ids(train_files)
        if args.holdout_validation_files <= 0:
            raise RuntimeError("--holdout_validation_files must be positive when --validation_dir is not provided")
        if len(all_files) <= args.holdout_validation_files:
            raise RuntimeError(
                f"Need more than {args.holdout_validation_files} tokenized arrow files to create train/validation split"
            )
        train_files = all_files[:-args.holdout_validation_files]
        val_files = all_files[-args.holdout_validation_files:]

    if not train_files:
        raise RuntimeError("No train arrow files with input_ids found")

    print(f"Packing {len(train_files)} train arrow files")
    print(f"Packing {len(val_files)} validation arrow files")

    n_train, train_docs, train_tokens = pack_split(
        train_files,
        outdir,
        args.train_prefix,
        args.chunk_size,
        args.max_train_chunks,
        args.eos_token_id,
        args.vocab_size,
        args.progress_chunks,
    )
    n_val, val_docs, val_tokens = pack_split(
        val_files,
        outdir,
        args.validation_prefix,
        args.chunk_size,
        args.max_validation_chunks,
        args.eos_token_id,
        args.vocab_size,
        args.progress_chunks,
    )
    print(
        f"Wrote {n_train} train chunks from {train_docs} docs / {train_tokens} tokens "
        f"and {n_val} validation chunks from {val_docs} docs / {val_tokens} tokens to {outdir}"
    )


if __name__ == "__main__":
    main()
