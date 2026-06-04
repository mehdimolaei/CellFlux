# Generating your first CellFlux image (single GPU)

This is a practical quickstart that does **not** require Slurm or training.
It uses the helper scripts added to this repo: `download_assets.py`,
`generate.py`, and `scripts/generate_bbbc021.sh`.

## 0. Requirements
- A **Linux machine with an NVIDIA CUDA GPU** (the env pins `torch==2.5.1+cu124`).
  macOS / Apple Silicon cannot run the pinned stack as-is.
- The conda env: `conda env create -f environment.yml && conda activate cellflux`
- Disk: ~13 GB for the download, ~1–2 GB after extracting only BBBC021.

## 0.5. Smoke-test with NO download first (recommended)
Validate that the whole generate->save pipeline runs on your machine before
downloading anything:
```bash
python generate.py --dataset bbbc021 --config bbbc021_all \
    --output_dir outputs/dryrun --dry_run
```
This fabricates a tiny synthetic dataset (2 control + 2 treated 64x64 images +
a 6-row CSV + a random embedding), runs on **CPU**, no checkpoint required, and
writes PNGs to `outputs/dryrun/fid_samples/epoch-0/`. The images are meaningless
(random-init model) — the point is to confirm the plumbing works and to show you
the exact on-disk layout real samples must follow:
```
outputs/dryrun/_dryrun/images/plate1/w1/ctrl_1.npy   # a control image
outputs/dryrun/_dryrun/index.csv                     # the index CSV
outputs/dryrun/_dryrun/emb.csv                        # the embedding CSV
```
> First run downloads the InceptionV3 weights (~100 MB) that the FID metric
> constructs, even though `--dry_run` skips the FID computation. Needs internet once.

### "Do I really need all 12.5 GB of samples?"
**No.** The model only needs a *few* control + treated images per compound — the
dry-run proves that with just 2+2. The 12.5 GB is purely how the public data is
**packaged** (all three datasets, full size, in one zip). Plain HTTP can't pull a
sub-folder out of a zip, so getting even a few *real* BBBC021 images means either
downloading the whole zip once (step 1) or asking someone to share a handful of
`.npy` files that you drop into the layout shown above.

## 1. Download everything automatically
```bash
python download_assets.py --dataset bbbc021 --dest ./assets
```
This pulls the **BBBC021 generator checkpoint** from HuggingFace and the
**12.5 GB IMPA data zip** from Zenodo, then extracts only the BBBC021 files.
(Use `--skip-zenodo` to grab just the checkpoint, or `--keep-zip` to keep the zip.)

## 2. Point the config at the downloaded files
Edit `configs/bbbc021_all.yaml` (the script prints the exact CSV locations):
```yaml
image_path:       assets/datasets/.../bbbc021_all/
data_index_path:  assets/datasets/.../bbbc021_df_all.csv
embedding_path:   assets/datasets/.../emb_fp.csv
```

## 3. Generate
```bash
bash scripts/generate_bbbc021.sh assets/checkpoints/bbbc021/checkpoint.pth outputs/my_first_run
```
Outputs:
- `outputs/my_first_run/fid_samples/epoch-0/<compound>/<id>.png` — synthetic treated cells
- `outputs/my_first_run/snapshots/0_0.png` — a quick preview grid
- the overall **FID** score is printed to the log

---

## Minimal file set for BBBC021 (if you want to download as little as possible)

Generation **seeds from a real control image** and **conditions on a compound
embedding**, so three things are mandatory:

| File | Role | Source |
|------|------|--------|
| `checkpoint.pth` | generator weights | HuggingFace `checkpoints/cellflux/bbbc021/` |
| `bbbc021_df_all.csv` | index: which images, splits, controls vs treated | Zenodo zip |
| `emb_fp.csv` | per-compound conditioning embedding | Zenodo zip |
| `*.npy` cell images | the actual control (and, for FID, treated) images | Zenodo zip |

You do **not** need every `.npy`. Only images for rows the **test** loader
actually reads are loaded from disk:
- every **treated** row in `SPLIT == test`, and
- at least one **control** row sharing that row's `BATCH` value
  (control is picked at random from the same batch — see
  `training/data_utils.py:read_files_pert`).

Train-split rows must still be **listed** in the CSV (they define the compound
vocabulary and the embedding lookup at `training/dataloader.py:162`), but their
`.npy` files are never opened during eval, so they don't have to exist on disk.

### CSV columns the loader expects (BBBC021)
The index CSV is read with `index_col=0`. Required columns:

| Column | Meaning | Used by |
|--------|---------|---------|
| `SAMPLE_KEY` | encodes the on-disk `.npy` path (parsed by splitting on `_`) | `read_files_pert` |
| `CPD_NAME` | compound name; must match a row in the embedding CSV | embedding lookup |
| `ANNOT` | annotation / MoA label | `y2id` |
| `STATE` | `0` = control, `1` = treated | control/treated split |
| `BATCH` | batch id; control is matched to treated within a batch | batch matching |
| `DOSE` | dose (BBBC021 only) | returned in batch dict |
| `SPLIT` | `train` or `test` | fold split |

### Image format
- `.npy`, shape `H × W × C` (channel-last), pixel values in `0–255`
  (the transform divides by 255 and normalizes to `[-1, 1]`).
- BBBC021 has **3 channels**; the UNet `condition_dim` is **1024** (so `emb_fp.csv`
  must have 1024-dim rows). See `models/model_configs.py`.

### Embedding CSV
- Read with `index_col=0`; the index is the **compound name** (`CPD_NAME`).
- Must contain a row for **every compound in the train-split treated set**.
- Row width must equal the model `condition_dim` (1024 for BBBC021).

> Hand-trimming the CSV to a few compounds works but is fiddly (you must keep
> control rows in the same batch, and keep every test compound present in the
> train set + embedding CSV). The path of least resistance is to let
> `download_assets.py` extract the whole BBBC021 subset and subset at runtime
> via the `mol_list:` field in the YAML config.
