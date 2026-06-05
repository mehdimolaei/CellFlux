# Running CellFlux generation as an Azure ML job

This runs `generate.py` on a managed GPU compute cluster, with the data and
checkpoint **mounted from Blob Storage** (no manual copy onto a VM).

## Prerequisites
- An Azure ML workspace + the `az ml` CLI v2 (`az extension add -n ml`).
- Data already in blob. Stage it once with:
  ```bash
  python download_assets.py --dataset bbbc021 --dest ./assets \
      --upload-blob "https://<acct>.blob.core.windows.net/<container>?<SAS>"
  ```
  That puts `checkpoints/bbbc021/checkpoint.pth` and `datasets/...` in your container.

## 1. Set defaults (so you can omit -g/-w on every call)
```bash
az configure --defaults group=<resource-group> workspace=<aml-workspace>
```

## 2. Create a GPU compute cluster (1× T4, scales to zero when idle)
```bash
az ml compute create --name gpu-t4 \
  --type AmlCompute --size Standard_NC4as_T4_v3 \
  --min-instances 0 --max-instances 1
```

## 3. Register the environment, data, and checkpoint assets
Edit the `path:` in `data_asset.yml` / `checkpoint_asset.yml` to match where you
staged the data (datastore path or blob URL), then:
```bash
az ml environment create -f azureml/environment.yml
az ml data create        -f azureml/data_asset.yml
az ml data create        -f azureml/checkpoint_asset.yml
```

## 4. Submit the job
Check the sub-paths in `generate_job.yml` (the `metadata/...` and `embeddings/...`
parts) match your blob layout, then:
```bash
az ml job create -f azureml/generate_job.yml --web
```
The generated PNGs + FID land in the job's `results` output, browsable/downloadable
from the Studio UI or:
```bash
az ml job download --name <job-name> --output-name results
```

## Notes
- **Driver/CUDA:** the AML GPU host provides the NVIDIA driver; the CUDA 12.4
  userspace comes from the pip wheels in the repo's `environment.yml`, so the base
  image just needs to be a standard AML Ubuntu image.
- **Eval vs training:** this spec evaluates/generates. To train, swap `generate.py`
  for `train.py` (drop `--eval_only`-style flags), raise `--epochs`, and use a
  larger/managed multi-GPU cluster; AML handles the distributed launch via the
  `distribution:` block (PyTorch).
- **Mount vs download:** `ro_mount` streams files on demand (good for a one-pass
  generation). For heavy training, prefer `mode: download` or copy to local SSD so
  random small-file reads aren't bottlenecked by the mount.
