# CellFlux — Options for Controlling Synthetic Image Generation

A practical reference for every knob that changes *what* you generate and *how*.
Written for the BBBC021 setup you have running locally (MPS), but the same flags
apply on GPU.

---

## 1. The mental model (read this first)

CellFlux does **not** invent a cell from scratch. For each output it:

1. takes a **real control (DMSO) cell image** as the starting point,
2. **conditions on a compound's chemical fingerprint** (`emb_fp.csv`), and
3. runs an **ODE** that morphs the control image into what that cell would look
   like *under that perturbation*.

So three independent things shape the output:

| Lever | Question it answers | Where you set it |
|---|---|---|
| **Which perturbation** | *what drug effect to apply* | config `mol_list` / `ood_set`, the CSV |
| **Starting point** | *which control + how much randomness* | `--use_initial`, `--noise_level`, `--seed` |
| **Quality / strength** | *how good & how strong the effect* | `--ode_options nfe`, `--ode_method`, `--cfg_scale` |

---

## 2. Full CLI flag reference (`generate.py`)

| Flag | Default | Effect |
|---|---|---|
| `--device` | `cuda`→cpu | `cpu`, `mps` (your Mac), or `cuda` (GPU). |
| `--batch_size` | 32 | Images per batch. Lower = less memory, slower throughput. |
| `--fid_samples` | 50000 | **Total** images to generate (stops once reached). Use small numbers (8–256) for quick runs. |
| `--use_initial` | 0 | **Starting point:** `0`=pure noise, `1`=real control image, `2`=control **+ noise**. Use `1` or `2` for "perturb a real cell". |
| `--noise_level` | 0.2 | Only for `--use_initial 2`. Higher = more variation/diversity from the same control. |
| `--cfg_scale` | 0.2 | **Perturbation strength** (classifier-free guidance). `0`=plain conditional; `>0` amplifies the drug effect; too high → artifacts. Try `0.0`–`0.5`. |
| `--ode_method` | midpoint | ODE solver: `heun2` (good default), `euler` (fast/rough), `midpoint`, `dopri5` (adaptive, high quality, slow), `rk4`. |
| `--ode_options` | `{"step_size":0.01}` | Solver settings. With `--edm_schedule`: `'{"nfe": 50}'` = number of steps (**main quality knob**). For `dopri5`: `'{"atol":1e-5,"rtol":1e-5}'`. |
| `--edm_schedule` | off | Use EDM step placement (recommended on). Enables the `nfe` option. |
| `--seed` | 0 | Changes the random noise → **different variations** of the same compound. |
| `--use_ema` | off | Use EMA weights — **required** for the pretrained checkpoint. |
| `--interpolate` | off | Save the **control→treated morph** (intermediate ODE steps) as grids instead of final images. |
| `--save_fid_samples` | off | Write individual PNGs (per compound). Turn on to keep outputs. |
| `--compute_fid` | off | Also score FID (processes *all* `fid_samples`, runs InceptionV3). Off = stop after first batch. |
| `--skip_fid` | off | Skip InceptionV3 entirely (faster; needed on MPS). Images still saved. |
| `--output_dir` | — | Where results go (relative to repo root). Each run → its own folder. |
| `--image_path` / `--data_index_path` / `--embedding_path` | from config | Override the data paths from the YAML (handy on the Mac / Azure). |
| `--config` | `bbbc021_all` | Which dataset config in `configs/`. |
| `--iter_ctrl` | off | Iterate over control images (each paired with a random treated) instead of over treated rows. Rarely needed. |

---

## 3. Controlling WHICH perturbation you generate

This is what you asked about most. There are three places, from easiest to most surgical:

### (a) `mol_list` in the config — restrict to specific compounds  *(recommended)*
In `configs/bbbc021_all.yaml`:
```yaml
mol_list: [taxol, DMSO]      # only generate 'taxol', keep DMSO controls
```
- `null` (default) = **all** compounds.
- **IMPORTANT:** always include **`DMSO`** in the list. Controls are named `DMSO`,
  and generation needs a same-batch control to start from. `mol_list: [taxol]`
  alone removes all controls and the run will error ("No control samples found").
- Multiple: `mol_list: [taxol, nocodazole, cisplatin, DMSO]`.

### (b) `ood_set` in the config — held-out compounds
```yaml
ood_set: [docetaxel, AZ841, cytochalasin D, simvastatin,
          cyclohexamide, latrunculin B, epothilone B, lactacystin]
```
These 8 are **excluded by default** (the model never trained on them — they test
generalization). To generate one anyway, **remove it from `ood_set`**. It already
has an embedding, so it will work — but quality reflects unseen-compound generalization.

### (c) The CSV (`bbbc021_df_all.csv`) — the source of truth
Columns that matter:

| Column | Meaning | How it affects generation |
|---|---|---|
| `CPD_NAME` | compound name | matched to `emb_fp.csv` for conditioning; filtered by `mol_list` |
| `ANNOT` | mode-of-action | grouping label |
| `STATE` | `0`=control(DMSO), `1`=treated | defines source vs target |
| `BATCH` | plate/batch id | control is matched to treated **within the same batch** |
| `SPLIT` | `train` / `test` | **eval generates from `test` treated rows** |
| `DOSE` | concentration | present in data, but the conditioning is per-compound (not dose-specific) |

Editing the CSV (e.g. flipping some rows to `SPLIT=test`, or trimming rows) changes
exactly which images get generated — but `mol_list` is the safer lever for most cases.

---

## 4. Compound catalog (what's available in your data)

`*` = held-out (in `ood_set`, excluded unless you remove it). 12 modes-of-action:

| Mode of action | Compounds |
|---|---|
| **Actin disruptors** | cytochalasin B, cytochalasin D*, latrunculin B* |
| **Aurora kinase inhibitors** | AZ258, AZ841* |
| **Cholesterol-lowering** | mevinolin/lovastatin, simvastatin* |
| **DNA damage** | chlorambucil, cisplatin, etoposide, mitomycin C |
| **DNA replication** | camptothecin, floxuridine, methotrexate, mitoxantrone |
| **Eg5 inhibitors** | AZ138 |
| **Epithelial** | PP-2 |
| **Kinase inhibitors** | PD-169316, alsterpaullone, bryostatin |
| **Microtubule destabilizers** | colchicine, demecolcine, nocodazole, vincristine |
| **Microtubule stabilizers** | docetaxel*, epothilone B*, taxol |
| **Protein degradation** | ALLN, MG-132, lactacystin*, proteasome inhibitor I |
| **Protein synthesis** | anisomycin, cyclohexamide*, emetine |
| **(control)** | DMSO — the untreated baseline you start from |

---

## 5. Controlling the starting point & variation

- **`--use_initial 1`** → start from the real control, minimal randomness → the
  output is "this exact control cell, perturbed." Most faithful.
- **`--use_initial 2 --noise_level 0.5`** → adds noise to the control → more
  diverse outputs (good for generating *many* varied samples of one compound).
- **`--use_initial 0`** → ignore controls, start from pure noise → least grounded.
- **`--seed 1`, `--seed 2`, …** → same settings, different random draws → different
  sample variations. Change the seed to get a fresh batch.

---

## 6. Controlling quality vs. speed

- **`--ode_options '{"nfe": 50}'`** — the big one. More steps = higher fidelity,
  linearly slower. `20` (quick) → `50` (default quality) → `100` (diminishing returns).
- **`--ode_method heun2`** — good quality/speed. `dopri5` (with `atol/rtol`) is
  highest quality but slow; `euler` is fastest/roughest.
- **`--cfg_scale`** — `0.0` faithful; `0.2`–`0.5` pushes the perturbation effect
  harder (more obvious morphological change, risk of artifacts).

---

## 7. Output & amount

- **`--fid_samples`** — how many total images (e.g. `16` for a peek, `256`+ for a set).
- **`--batch_size`** — per-batch; on MPS keep small (4–8).
- **`--save_fid_samples`** — actually write the PNGs (per-compound subfolders).
- **`--output_dir outputs/<name>`** — each run gets its own folder; results land in
  `outputs/<name>/fid_samples/epoch-0/<compound>/*.png` and a grid in `snapshots/`.
- **`--interpolate`** — instead of finals, save the morphing sequence
  (control → … → treated) to `outputs/<name>/interpolation/`.

---

## 8. Where each control lives

| To change… | Edit… |
|---|---|
| which compounds / hold-outs | `configs/bbbc021_all.yaml` → `mol_list`, `ood_set` |
| source data / splits / doses | `bbbc021_df_all.csv` |
| starting point, quality, count, device | **CLI flags** on `generate.py` |
| the conditioning vectors themselves | `emb_fp.csv` (regenerate fingerprints) |

---

## 9. Ready-to-run recipes (MPS)

**Higher-quality batch of one compound (nocodazole):**
First set `mol_list: [nocodazole, DMSO]` in the config, then:
```bash
/tmp/cfenv/bin/python generate.py \
  --dataset bbbc021 --config bbbc021_all \
  --resume assets/checkpoints/bbbc021/checkpoint.pth \
  --image_path assets/zenodo/bbbc021_all \
  --data_index_path assets/zenodo/bbbc021_all/metadata/bbbc021_df_all.csv \
  --embedding_path assets/zenodo/embeddings/emb_fp.csv \
  --output_dir outputs/nocodazole_hq \
  --device mps --batch_size 4 --fid_samples 16 \
  --use_ema --edm_schedule --skip_fid --save_fid_samples \
  --ode_method heun2 --ode_options '{"nfe": 50}' \
  --use_initial 1 --cfg_scale 0.3
```

**See the control→treated morph (interpolation):**
```bash
  ... same as above ... \
  --output_dir outputs/morph --interpolate --batch_size 2 --fid_samples 2
```

**More diverse samples of one compound (vary seed + noise):**
```bash
  ... --use_initial 2 --noise_level 0.6 --seed 7 ...
```

**All compounds, quick peek:**
Leave `mol_list: null`, use `--fid_samples 48 --ode_options '{"nfe": 20}'`.
Outputs sort into one subfolder per compound automatically.
