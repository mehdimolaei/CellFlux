#!/usr/bin/env python3
"""
Download the CellFlux assets needed to generate images, without doing it by hand.

What it fetches (verified against the live repos on 2026-06):

  HuggingFace  suyc21/CellFlux  (https://huggingface.co/suyc21/CellFlux)
    checkpoints/cellflux/<dataset>/checkpoint.pth   <- the GENERATOR weights
    data/rxrx1/rxrx1_df_subset.csv                  (rxrx1 eval index)
    data/cpg0000/metadata_large_gene2vec_subset.csv (cpg0000 eval index)
    data/cpg0000/cpg0000_combined_embeddings.csv    (cpg0000 embeddings)

  Zenodo  record 8307629  (the IMPA pre-processed data, ONE 12.5 GB zip)
    IMPA_reproducibility.zip  <- raw .npy images + metadata + embeddings
                                 (this is the ONLY source of BBBC021 images,
                                  its bbbc021_df_all.csv index, and emb_fp.csv)

Notes / honesty:
  * The HuggingFace paths above are verified to exist.
  * The INTERNAL layout of the Zenodo zip is NOT hard-coded here because it was
    not independently verified end-to-end. Instead the script downloads the zip,
    lists its contents, and selectively extracts only entries matching your
    dataset (default: 'bbbc021'), then reports where the index/embedding CSVs
    landed so you can paste them into configs/<dataset>.yaml.

Usage:
  python download_assets.py --dataset bbbc021 --dest ./assets
  python download_assets.py --dataset bbbc021 --dest ./assets --skip-zenodo   # checkpoint only
  python download_assets.py --dataset bbbc021 --dest ./assets --keep-zip      # don't delete the 12.5 GB zip

Re-running is safe: completed downloads are skipped, partial downloads resume.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

HF_BASE = "https://huggingface.co/suyc21/CellFlux/resolve/main"
ZENODO_ZIP = "https://zenodo.org/records/8307629/files/IMPA_reproducibility.zip?download=1"

# HuggingFace files to grab per dataset. BBBC021's index/embedding CSVs are NOT
# on HuggingFace (they come from the Zenodo zip), so only the checkpoint is listed.
HF_FILES = {
    "bbbc021": [
        "checkpoints/cellflux/bbbc021/checkpoint.pth",
    ],
    "rxrx1": [
        "checkpoints/cellflux/rxrx1/checkpoint.pth",
        "data/rxrx1/rxrx1_df_subset.csv",
    ],
    "cpg0000": [
        "checkpoints/cellflux/cpg0000/checkpoint.pth",
        "data/cpg0000/metadata_large_gene2vec_subset.csv",
        "data/cpg0000/cpg0000_combined_embeddings.csv",
    ],
}


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def download(url, dest: Path, resume=True):
    """Stream a URL to dest with a progress bar and HTTP-range resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Discover total size.
    head = Request(url, method="GET", headers={"Range": "bytes=0-0"})
    try:
        with urlopen(head) as r:
            cr = r.headers.get("Content-Range")
            total = int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length", 0))
    except HTTPError:
        total = 0

    if dest.exists() and total and dest.stat().st_size == total:
        print(f"  [skip] {dest}  ({human(total)} already present)")
        return dest

    start = tmp.stat().st_size if (resume and tmp.exists()) else 0
    if start and total and start >= total:
        start = 0  # corrupt/over-long partial -> restart
    headers = {"Range": f"bytes={start}-"} if start else {}
    mode = "ab" if start else "wb"

    req = Request(url, headers=headers)
    with urlopen(req) as r, open(tmp, mode) as f:
        done = start
        block = 1024 * 256
        while True:
            chunk = r.read(block)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                sys.stdout.write(f"\r  {dest.name}: {human(done)}/{human(total)} ({pct:4.1f}%)")
            else:
                sys.stdout.write(f"\r  {dest.name}: {human(done)}")
            sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.rename(dest)
    return dest


def extract_zip_subset(zip_path: Path, out_dir: Path, pattern: str):
    """List the zip and extract only members whose path contains `pattern`."""
    pattern = pattern.lower()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        matched = [n for n in names if pattern in n.lower()]
        print(f"\nZip has {len(names)} entries; {len(matched)} match '{pattern}'.")
        if not matched:
            print("No matches. Full top-level listing so you can pick a pattern:")
            tops = sorted({n.split('/')[0] + ('/' + n.split('/')[1] if '/' in n.strip('/') else '')
                           for n in names})
            for t in tops[:60]:
                print("   ", t)
            return []
        for i, n in enumerate(matched, 1):
            zf.extract(n, out_dir)
            if i % 200 == 0 or i == len(matched):
                sys.stdout.write(f"\r  extracted {i}/{len(matched)}")
                sys.stdout.flush()
        sys.stdout.write("\n")
    return [out_dir / n for n in matched]


# ---------------------------------------------------------------------------
# Azure Blob staging (optional). Uses `azcopy` + a container SAS URL, e.g.
#   https://<account>.blob.core.windows.net/<container>?<sas-token>
# so no `az login` is needed.
# ---------------------------------------------------------------------------
def blob_url(sas_url: str, subpath: str = "") -> str:
    """Insert `subpath` into a container SAS URL, keeping the ?<sas> at the end."""
    if "?" in sas_url:
        base, query = sas_url.split("?", 1)
        base = base.rstrip("/")
        return f"{base}/{subpath}?{query}" if subpath else f"{base}?{query}"
    base = sas_url.rstrip("/")
    return f"{base}/{subpath}" if subpath else base


def run_azcopy(azcopy: str, src: str, dst: str, recursive: bool = True):
    if not shutil.which(azcopy):
        raise SystemExit(
            f"'{azcopy}' not found. Install azcopy: https://aka.ms/downloadazcopy "
            "(or pass --azcopy /path/to/azcopy)."
        )
    cmd = [azcopy, "copy", src, dst] + (["--recursive"] if recursive else [])
    # Don't print the SAS token.
    safe = "<src>" if "?" in src else src
    safe_dst = "<dst-with-sas>" if "?" in dst else dst
    print(f"  $ {azcopy} copy {safe} {safe_dst}{' --recursive' if recursive else ''}")
    subprocess.run(cmd, check=True)


def pull_from_blob(azcopy: str, sas_url: str, dest: Path):
    """Download previously-staged assets from blob into `dest` (skips web)."""
    dest.mkdir(parents=True, exist_ok=True)
    run_azcopy(azcopy, blob_url(sas_url, "*"), str(dest), recursive=True)
    print(f"Pulled staged assets from blob into {dest}")


def push_to_blob(azcopy: str, sas_url: str, dest: Path):
    """Upload the local checkpoints/ and datasets/ trees to blob for reuse."""
    for sub in ["checkpoints", "datasets"]:
        local = dest / sub
        if local.exists():
            run_azcopy(azcopy, str(local), blob_url(sas_url, ""), recursive=True)
    print("Uploaded assets to blob (checkpoints/ and datasets/).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bbbc021", "rxrx1", "cpg0000"], default="bbbc021")
    ap.add_argument("--dest", default="./assets", help="output root directory")
    ap.add_argument("--skip-zenodo", action="store_true", help="only download HF checkpoint/CSVs")
    ap.add_argument("--skip-hf", action="store_true", help="only download the Zenodo data zip")
    ap.add_argument("--extract-pattern", default=None, help="zip members to extract (default: the dataset name)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the 12.5 GB zip after extraction")
    ap.add_argument("--from-blob", metavar="SAS_URL", default=None,
                    help="Pull already-staged assets from this Azure container SAS URL "
                         "instead of downloading from the web.")
    ap.add_argument("--upload-blob", metavar="SAS_URL", default=None,
                    help="After downloading, upload checkpoints/ and datasets/ to this "
                         "Azure container SAS URL for reuse by future VMs/jobs.")
    ap.add_argument("--azcopy", default="azcopy", help="path to the azcopy binary")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    pattern = args.extract_pattern or args.dataset
    print(f"Destination: {dest}\nDataset: {args.dataset}\n")

    # Fast path: data already staged in blob -> just pull it down, done.
    if args.from_blob:
        print("== Azure Blob: pulling staged assets ==")
        pull_from_blob(args.azcopy, args.from_blob, dest)
        print("\nDone. Edit configs/%s*.yaml to point at the pulled files, then run "
              "generate.py." % args.dataset)
        return

    # 1) HuggingFace: checkpoint (+ CSVs for rxrx1/cpg0000)
    if not args.skip_hf:
        print("== HuggingFace: model checkpoint + CSVs ==")
        for rel in HF_FILES[args.dataset]:
            url = f"{HF_BASE}/{rel}"
            # flatten checkpoints/cellflux/<ds>/checkpoint.pth -> checkpoints/<ds>/checkpoint.pth
            out = dest / rel.replace("checkpoints/cellflux/", "checkpoints/")
            download(url, out)

    # 2) Zenodo: the 12.5 GB IMPA data zip (raw .npy images, index, embeddings)
    if not args.skip_zenodo:
        print("\n== Zenodo: IMPA pre-processed data (12.5 GB zip) ==")
        zip_path = dest / "IMPA_reproducibility.zip"
        download(ZENODO_ZIP, zip_path)
        data_dir = dest / "datasets"
        extracted = extract_zip_subset(zip_path, data_dir, pattern)
        if extracted and not args.keep_zip:
            print("Removing the zip to reclaim space (use --keep-zip to keep it).")
            os.remove(zip_path)

        # Help the user fill in configs/<dataset>.yaml
        print("\n== Locating CSVs for your config ==")
        csvs = [p for p in (data_dir).rglob("*.csv")] if data_dir.exists() else []
        for needle in ["df_all", "df_subset", "metadata", "emb", "embedding"]:
            for c in csvs:
                if needle in c.name.lower():
                    print(f"  {needle:>9} -> {c}")

    # Optionally stage everything to blob so future runs skip the 12.5 GB download.
    if args.upload_blob:
        print("\n== Azure Blob: uploading staged assets ==")
        push_to_blob(args.azcopy, args.upload_blob, dest)

    print("\nDone. Next: edit configs/%s*.yaml so image_path / data_index_path / "
          "embedding_path point at the files above, then run generate.py." % args.dataset)


if __name__ == "__main__":
    main()
