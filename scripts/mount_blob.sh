#!/bin/bash
# Mount an Azure Blob container as a local filesystem with BlobFuse2 and point
# the CellFlux config at it. Best for EVAL or light use; for heavy multi-epoch
# TRAINING prefer copying data to local SSD (see download_assets.py --from-blob),
# since random small-file reads over a mount are slower.
#
# Prerequisites: Ubuntu, and a container SAS token (read/list).
#
# Configure via environment variables, then run:
#   export AZ_ACCOUNT=myaccount
#   export AZ_CONTAINER=mycontainer
#   export AZ_SAS='sv=2023-...&sig=...'      # SAS token WITHOUT a leading '?'
#   export MOUNT_DIR=/mnt/blobdata           # optional (default shown)
#   export CACHE_DIR=/mnt/blobfusecache      # optional; put on fast local SSD
#   bash scripts/mount_blob.sh
#
# After mounting, edit configs/<dataset>.yaml (this script patches bbbc021 if the
# expected files are found; otherwise it prints what it saw so you can adjust).
set -euo pipefail

: "${AZ_ACCOUNT:?set AZ_ACCOUNT}"
: "${AZ_CONTAINER:?set AZ_CONTAINER}"
: "${AZ_SAS:?set AZ_SAS (SAS token without leading '?')}"
MOUNT_DIR="${MOUNT_DIR:-/mnt/blobdata}"
CACHE_DIR="${CACHE_DIR:-/mnt/blobfusecache}"
CONFIG="${CONFIG:-configs/bbbc021_all.yaml}"

# 1) Install blobfuse2 if missing.
if ! command -v blobfuse2 >/dev/null 2>&1; then
  echo "Installing blobfuse2..."
  source /etc/os-release
  wget -q "https://packages.microsoft.com/config/ubuntu/${VERSION_ID}/packages-microsoft-prod.deb" -O /tmp/pmp.deb
  sudo dpkg -i /tmp/pmp.deb
  sudo apt-get update -y
  sudo apt-get install -y blobfuse2
fi

# 2) Write a blobfuse2 config with a local file cache (helps random reads).
CFG=/tmp/blobfuse2.yaml
cat > "$CFG" <<EOF
allow-other: true
logging:
  level: log_warning
components:
  - libfuse
  - file_cache
  - attr_cache
  - azstorage
libfuse:
  attribute-expiration-sec: 120
  entry-expiration-sec: 120
file_cache:
  path: ${CACHE_DIR}
  timeout-sec: 600
attr_cache:
  timeout-sec: 120
azstorage:
  type: block
  account-name: ${AZ_ACCOUNT}
  endpoint: https://${AZ_ACCOUNT}.blob.core.windows.net
  container: ${AZ_CONTAINER}
  mode: sas
  sas: ${AZ_SAS}
EOF

sudo mkdir -p "$MOUNT_DIR" "$CACHE_DIR"
sudo chown "$(id -u):$(id -g)" "$MOUNT_DIR" "$CACHE_DIR" || true

# 3) Mount (idempotent: unmount first if already mounted).
if mountpoint -q "$MOUNT_DIR"; then
  echo "$MOUNT_DIR already mounted; remounting."
  blobfuse2 unmount "$MOUNT_DIR" || sudo umount "$MOUNT_DIR" || true
fi
blobfuse2 mount "$MOUNT_DIR" --config-file="$CFG"
echo "Mounted $AZ_CONTAINER at $MOUNT_DIR"
echo "Top-level contents:"
ls -la "$MOUNT_DIR" || true

# 4) Best-effort: patch the bbbc021 config if the expected files are present.
IMG=$(find "$MOUNT_DIR" -type d -iname "*bbbc021*" | head -1 || true)
IDX=$(find "$MOUNT_DIR" -iname "bbbc021_df_all.csv" | head -1 || true)
EMB=$(find "$MOUNT_DIR" -iname "emb_fp.csv" | head -1 || true)
if [[ -n "$IMG" && -n "$IDX" && -n "$EMB" ]]; then
  python scripts/patch_config.py --config "$CONFIG" \
    --image_path "$IMG" --data_index_path "$IDX" --embedding_path "$EMB"
else
  echo "Could not auto-locate all three files under $MOUNT_DIR."
  echo "  image dir : ${IMG:-NOT FOUND}"
  echo "  index csv : ${IDX:-NOT FOUND}"
  echo "  embed csv : ${EMB:-NOT FOUND}"
  echo "Edit $CONFIG manually, or re-run scripts/patch_config.py with the right paths."
fi

echo "To unmount later:  blobfuse2 unmount $MOUNT_DIR"
