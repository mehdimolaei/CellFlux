#!/usr/bin/env python3
"""
Rewrite the three data paths in a CellFlux dataset config, preserving comments.

Line-based replacement (not a full YAML reparse) so the comments in
configs/*.yaml survive. Only the keys you pass are touched.

Example:
  python scripts/patch_config.py --config configs/bbbc021_all.yaml \
      --image_path /mnt/blobdata/bbbc021_all \
      --data_index_path /mnt/blobdata/bbbc021_all/metadata/bbbc021_df_all.csv \
      --embedding_path /mnt/blobdata/embeddings/emb_fp.csv
"""
import argparse
import re
import sys
from pathlib import Path

KEYS = ["image_path", "data_index_path", "embedding_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    for k in KEYS:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args()

    path = Path(args.config)
    lines = path.read_text().splitlines(keepends=True)
    updates = {k: getattr(args, k) for k in KEYS if getattr(args, k)}
    if not updates:
        sys.exit("Nothing to do: pass at least one of --image_path/--data_index_path/--embedding_path")

    seen = set()
    for i, line in enumerate(lines):
        for k, v in updates.items():
            # match 'key:' at start of line, keep any trailing '# comment'
            m = re.match(rf"^(\s*{re.escape(k)}\s*:)(\s*[^#\n]*)(#.*)?$", line)
            if m:
                comment = m.group(3) or ""
                newline = f"{m.group(1)} {v} {comment}".rstrip() + "\n"
                lines[i] = newline
                seen.add(k)
                print(f"  {k}: -> {v}")

    missing = set(updates) - seen
    if missing:
        print(f"WARNING: keys not found in {path}: {sorted(missing)}")

    if args.dry_run:
        print("--- dry-run, not written ---")
        return
    path.write_text("".join(lines))
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
