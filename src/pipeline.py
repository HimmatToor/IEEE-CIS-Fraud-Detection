import argparse
import os
from pathlib import Path

import boto3
import torch

from build_graph import build_hetero_graph, load_raw_tables

RAW_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def upload_file(s3, bucket: str, local_path: Path, key: str) -> None:
    print(f"uploading {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def upload_raw(s3, bucket: str, data_dir: Path) -> None:
    for filename in RAW_FILES:
        path = data_dir / filename
        if path.exists():
            upload_file(s3, bucket, path, f"raw/{filename}")


def build_and_upload_graph(s3, bucket: str, data_dir: Path, out_dir: Path, split: str) -> None:
    df = load_raw_tables(data_dir, split)
    graph = build_hetero_graph(df)
    graph.validate(raise_on_error=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_graph.pt"
    torch.save(graph, out_path)

    upload_file(s3, bucket, out_path, f"processed/{split}_graph.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection-data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--upload-raw", action="store_true", help="also upload the raw CSVs to s3://bucket/raw/")
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("pass --bucket or set the S3_BUCKET env var")

    s3 = boto3.client("s3")

    if args.upload_raw:
        upload_raw(s3, args.bucket, args.data_dir)

    for split in ("train", "test"):
        build_and_upload_graph(s3, args.bucket, args.data_dir, args.out_dir, split)


if __name__ == "__main__":
    main()
