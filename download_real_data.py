"""Download ALL real KASCADE data from S3 and save locally.
Saves matrices and features separately per run as individual .npz files
in data/real_kascade/ directory. No quality cuts applied.
"""
import numpy as np
import boto3, io, os, time
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = 'kascade-with-arr-times-npz'
OUT_DIR = 'data/real_kascade'

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # List all runs
    all_files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET):
        all_files.extend([obj['Key'] for obj in page.get('Contents', [])])
    runs = sorted(set(f.split('_')[0] for f in all_files))
    runs = [r for r in runs if r[-1] in {'0', '1'}]
    print(f"{len(runs)} runs to download", flush=True)

    # Check which runs are already downloaded
    existing = set()
    for f in os.listdir(OUT_DIR):
        if f.endswith('_matrices.npz'):
            existing.add(f.replace('_matrices.npz', ''))
    print(f"{len(existing)} already downloaded, {len(runs) - len(existing)} remaining", flush=True)

    total_events = 0
    for run in tqdm(runs, desc="Downloading"):
        if run in existing:
            # Count events from existing file
            try:
                m = np.load(f"{OUT_DIR}/{run}_matrices.npz", allow_pickle=True)
                total_events += len(m['matrices'])
            except:
                pass
            continue

        try:
            for suffix in ['matrices', 'features', 'extra_features']:
                buf = io.BytesIO()
                s3.download_fileobj(BUCKET, f'{run}_{suffix}.npz', buf)
                buf.seek(0)
                with open(f"{OUT_DIR}/{run}_{suffix}.npz", 'wb') as f:
                    f.write(buf.getvalue())

            # Count events
            m = np.load(f"{OUT_DIR}/{run}_matrices.npz", allow_pickle=True)
            total_events += len(m['matrices'])

        except Exception as e:
            print(f"  Failed {run}: {e}", flush=True)
            continue

    print(f"\nTotal: {total_events:,} events from {len(runs)} runs", flush=True)
    print(f"Saved to {OUT_DIR}/", flush=True)
    print(f"Time: {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == '__main__':
    main()
