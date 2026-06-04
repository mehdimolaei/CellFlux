# Minimal single-GPU generation / evaluation entrypoint for CellFlux.
#
# This is a thin wrapper around the existing training code that strips out the
# Slurm / DistributedDataParallel machinery so you can generate synthetic
# perturbed cell images from a single GPU (or CPU for a smoke test).
#
# It mirrors what `train.py --eval_only` does, but:
#   * never initializes torch.distributed,
#   * uses the non-distributed CellDataLoader_Eval data loader,
#   * loads ONLY the model weights from the checkpoint (no optimizer state).
#
# Example (after running download_assets.py and editing configs/bbbc021_all.yaml):
#
#   python generate.py \
#       --dataset bbbc021 --config bbbc021_all \
#       --resume assets/checkpoints/bbbc021/checkpoint.pth \
#       --output_dir outputs/my_first_run \
#       --batch_size 16 --fid_samples 256 \
#       --use_ema --edm_schedule --skewed_timesteps \
#       --ode_method heun2 --ode_options '{"nfe": 50}' \
#       --use_initial 2 --noise_level 1.0 \
#       --compute_fid --save_fid_samples
#
# Generated PNGs land in:  <output_dir>/fid_samples/epoch-0/<compound>/<id>.png
# A preview grid lands in:  <output_dir>/snapshots/0_0.png

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
from training.dataloader import CellDataLoader_Eval
from training.eval_loop import eval_model

logger = logging.getLogger(__name__)


def load_yaml_config(yaml_path):
    with open("configs/" + yaml_path + ".yaml", "r") as file:
        return yaml.safe_load(file)


def setup_dry_run(args):
    """Fabricate a tiny synthetic dataset (2 control + 2 treated images + CSVs)
    and point `args` at it, so the full generate->save pipeline can be exercised
    on CPU with NO download and NO checkpoint. Output images are meaningless
    (random-init model); this only validates the plumbing and shows the exact
    on-disk layout that real samples must follow.
    """
    import pandas as pd
    from models.model_configs import MODEL_CONFIGS

    root = Path(args.output_dir) / "_dryrun"
    img_root = root / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    H = W = 64
    C = int(getattr(args, "n_channels", 3))

    def make_img(sample_key):
        # bbbc021-style parse: "A_B_C..." -> <img_root>/A/B/<C...>.npy
        a, b, *rest = sample_key.split("_")
        d = img_root / a / b
        d.mkdir(parents=True, exist_ok=True)
        arr = (np.random.rand(H, W, C) * 255).astype(np.uint8)
        np.save(d / ("_".join(rest) + ".npy"), arr)

    cpd, annot, batch = "taxol", "Microtubule", "b1"
    # (SAMPLE_KEY, CPD_NAME, ANNOT, STATE[0=ctrl,1=trt], SPLIT)
    spec = [
        ("plate1_w1_ctrl_0", "control", "DMSO", 0, "train"),  # train defines the
        ("plate1_w2_trt_0",  cpd,       annot,  1, "train"),  # compound vocabulary
        ("plate1_w1_ctrl_1", "control", "DMSO", 0, "test"),
        ("plate1_w1_ctrl_2", "control", "DMSO", 0, "test"),
        ("plate1_w2_trt_1",  cpd,       annot,  1, "test"),
        ("plate1_w2_trt_2",  cpd,       annot,  1, "test"),
    ]
    rows = []
    for key, cp, an, state, split in spec:
        make_img(key)
        rows.append(dict(SAMPLE_KEY=key, CPD_NAME=cp, ANNOT=an,
                         STATE=state, BATCH=batch, DOSE=1.0, SPLIT=split))
    index_csv = root / "index.csv"
    pd.DataFrame(rows).to_csv(index_csv)  # default RangeIndex -> read with index_col=0

    dim = MODEL_CONFIGS[args.dataset].get("condition_dim", 1024)
    emb_csv = root / "emb.csv"
    pd.DataFrame(np.random.randn(1, dim), index=[cpd]).to_csv(emb_csv)

    # Repoint args at the fake data + cheap, safe generation settings.
    args.image_path = str(img_root)
    args.data_index_path = str(index_csv)
    args.embedding_path = str(emb_csv)
    args.device = "cpu"
    args.use_ema = False
    args.resume = ""               # random-init model, no checkpoint needed
    args.batch_size = 2
    args.fid_samples = 4
    args.num_workers = 0
    args.compute_fid = False       # skip FID math (metric is still built once)
    args.save_fid_samples = True
    args.use_initial = 2
    args.noise_level = 1.0
    args.cfg_scale = 0.0
    args.edm_schedule = True
    args.ode_method = "heun2"
    args.ode_options = {"nfe": 5}
    args.interpolate = False
    args.discrete_flow_matching = False
    print(f"[dry-run] fabricated tiny dataset under {root}")
    print(f"[dry-run] image layout example: {img_root}/plate1/w1/ctrl_1.npy")


def pick_device(requested):
    """Honor an explicit --device (cuda/mps/cpu), falling back to CPU if the
    requested accelerator isn't available. Default 'cuda' -> CPU off-GPU."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if requested == "mps" and mps_ok:
        return torch.device("mps")
    if requested in ("cuda", "mps"):
        logger.warning(f"Requested device '{requested}' unavailable -> CPU.")
    return torch.device("cpu")


def load_weights_only(model, resume_path):
    """Load just the model state_dict from a checkpoint (ignore optimizer/epoch)."""
    # weights_only=False: the checkpoint stores its args as a SimpleNamespace, which
    # PyTorch 2.6+ blocks under the default weights_only=True. Safe for trusted files.
    if resume_path.startswith("https"):
        ckpt = torch.hub.load_state_dict_from_url(
            resume_path, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing[:8]}...")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected[:8]}...")
    logger.info(f"Loaded model weights from {resume_path}")


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- single-process setup (no torch.distributed) ---
    args.distributed = False
    args.num_tasks = 1
    args.global_rank = 0
    args.gpu = 0

    if getattr(args, "dry_run", False):
        setup_dry_run(args)
    elif not args.resume:
        raise SystemExit("--resume <checkpoint.pth> is required for generation.")

    device = pick_device(args.device)
    if device.type == "cpu":
        logger.warning("Running on CPU. Generation works but is slow; keep "
                       "--batch_size and --fid_samples small.")
    elif device.type == "mps":
        logger.warning("Running on Apple MPS. Faster than CPU, but if an op is "
                       "unsupported, fall back with --device cpu.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=4)

    logger.info(f"Initializing dataset: {args.dataset}")
    if args.dataset not in ["bbbc021", "rxrx1", "cpg0000"]:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")
    datamodule = CellDataLoader_Eval(args)
    data_loader_test = datamodule.test_dataloader()

    logger.info("Initializing model")
    model = instantiate_model(
        architechture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )
    if args.resume:
        load_weights_only(model, args.resume)
    else:
        logger.warning("No checkpoint (--resume) given: using RANDOM weights. "
                       "Outputs are meaningless; this only validates the pipeline.")
    model.to(device)
    model.eval()

    # The dataloader builds the conditioning embedding table on CPU (it only checks
    # for CUDA). Move it to the run device so it matches the (MPS/CUDA) inputs.
    datamodule.embedding_matrix = datamodule.embedding_matrix.to(device)

    if args.use_initial in [1, 2]:
        logger.info("Generating FROM CONTROL IMAGE (control -> treated).")
    else:
        logger.info("Generating FROM RANDOM NOISE.")

    eval_stats = eval_model(
        model,
        data_loader_test,
        device,
        epoch=0,
        fid_samples=args.fid_samples,
        args=args,
        datamodule=datamodule,
        use_initial=args.use_initial,
        interpolate=args.interpolate,
    )
    logger.info(f"Done. Stats: {eval_stats}")
    logger.info(f"Images saved under: {Path(args.output_dir) / 'fid_samples'}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Fabricate a tiny synthetic dataset and run end-to-end on CPU with "
             "no download and no checkpoint (validates the pipeline only).",
    )
    # Path overrides: take precedence over the YAML config. Essential on Azure ML,
    # where the data mounts at a runtime path not known when the config was written.
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip the InceptionV3 FID metric entirely (faster, and "
                             "avoids unsupported-op errors on CPU/MPS). Images are still saved.")
    parser.add_argument("--image_path", dest="cli_image_path", default=None,
                        help="Override image_path from the YAML config.")
    parser.add_argument("--data_index_path", dest="cli_data_index_path", default=None,
                        help="Override data_index_path from the YAML config.")
    parser.add_argument("--embedding_path", dest="cli_embedding_path", default=None,
                        help="Override embedding_path from the YAML config.")
    args = parser.parse_args()
    yaml_config = load_yaml_config(args.config)
    args_dict = vars(args)
    args_dict.update(yaml_config)  # yaml (dataset_name, paths, n_channels...) wins
    args = SimpleNamespace(**args_dict)
    # Re-apply CLI path overrides AFTER the YAML merge so they actually win.
    for src, dst in [("cli_image_path", "image_path"),
                     ("cli_data_index_path", "data_index_path"),
                     ("cli_embedding_path", "embedding_path")]:
        val = getattr(args, src, None)
        if val:
            setattr(args, dst, val)
    main(args)
