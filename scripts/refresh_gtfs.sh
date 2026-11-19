#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/data/gtfs"
mkdir -p "$DIR"
cd "$DIR"

BASE="https://raw.githubusercontent.com/reclaimchennai/India_GTFS/main/tamil-nadu/chennai"
for name in mtc-bus metro-cmrl suburban-rail-mrts; do
  echo "Downloading $name..."
  curl -fsSL -o "${name}.zip" "$BASE/${name}.zip"
  rm -rf "$name"
  mkdir -p "$name"
  unzip -qo "${name}.zip" -d "$name"
done

rm -rf "$DIR/_cache"
echo "GTFS refreshed. Cache will rebuild on next API start."
