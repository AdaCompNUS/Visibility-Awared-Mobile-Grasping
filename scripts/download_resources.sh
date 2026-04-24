#!/usr/bin/env bash
# Download precomputed resource files for Mobile Grasping in Dynamic Environments.
# These files are too large to store in the repository.
#
# Usage:
#   bash scripts/download_resources.sh
#
# The script downloads into the resources/ directory relative to the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESOURCE_DIR="$REPO_ROOT/resources"

mkdir -p "$RESOURCE_DIR"

DOWNLOAD_URL="${MOBILE_GRASPING_RESOURCE_URL:-https://www.dropbox.com/scl/fo/9gxri23a1fn4lmudmhat0/AJ3HfBHsj3XLkFANqXZsbb8?rlkey=5co0nluo78otf3103w5g9vwx9&st=yflfjy6u&dl=1}"

# ── Files to download ───────────────────────────────────────────
FILES=(
    "capability_map.pkl"
    "inverse_reachability_map.pkl"
    "torso_map.pkl"
    "reachability_map.pkl"
    "costmap.npz"
)

missing=()
for fname in "${FILES[@]}"; do
    [ -f "$RESOURCE_DIR/$fname" ] || missing+=("$fname")
done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "All precomputed resources already exist in $RESOURCE_DIR"
    exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

archive="$tmp_dir/resources.zip"
echo "Downloading precomputed resources archive..."
curl -fL -o "$archive" "$DOWNLOAD_URL"
unzip -q "$archive" -d "$tmp_dir/extracted"

echo "Installing resources to $RESOURCE_DIR"
for fname in "${missing[@]}"; do
    src="$(find "$tmp_dir/extracted" -type f -name "$fname" -print -quit)"
    if [ -z "$src" ]; then
        echo "ERROR: $fname was not found in downloaded archive." >&2
        exit 1
    fi
    cp "$src" "$RESOURCE_DIR/$fname"
    echo "  [installed] $fname"
done

echo "Done."
