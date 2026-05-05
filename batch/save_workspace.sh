#!/usr/bin/env bash
# Mirror the run-relevant artifacts out of the cloned repo into
# /workspace/watermark_encoding/ and /workspace/decoding/ so they can be
# uploaded to S3 with whatever tool you prefer (Runpod UI, aws-cli, etc.).
#
# Layout produced (clean, no code, no junk):
#   /workspace/watermark_encoding/
#     ├── data/
#     │   ├── images/         2560 PNGs
#     │   ├── baseline/       10 PNGs
#     │   └── metadata.json
#     └── models/             8 LoRA dirs (only *_last.safetensors)
#   /workspace/decoding/
#     ├── checkpoints/        *.pth + summary JSONs
#     └── results/            *.md, *.json, figures/
#
# Excluded by construction: src/, scripts/, configs/, data/splits.json (all
# tracked in git), __pycache__/, *.log, .smoke/, .DS_Store, *.gitkeep.
#
# Default mode is "hardlink" (cp -al): zero extra disk, behaves like real
# files for upload tools. Use --copy for a real copy if you need to upload
# from a different filesystem than the source repo.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODE="hardlink"          # hardlink | copy
KEEP_500STEPS=0          # 1 to keep both LoRA snapshots (_last + _500steps)
INCLUDE_OPTIMIZER=1      # 0 to strip optimizer_state_dict from .pth (~halves size)

usage() {
  cat <<EOF
Usage: bash save_workspace.sh [options]

Options:
  --workspace PATH      Workspace root (default: /workspace)
  --copy                Copy files instead of hardlinking (uses real disk)
  --keep-500steps       Include LoRA *_500steps.safetensors (default: drop)
  --strip-optimizer     Strip optimizer_state_dict from .pth files (~halves size)
  -h | --help           This help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)        WORKSPACE="$2"; shift 2 ;;
    --copy)             MODE="copy"; shift ;;
    --keep-500steps)    KEEP_500STEPS=1; shift ;;
    --strip-optimizer)  INCLUDE_OPTIMIZER=0; shift ;;
    -h|--help)          usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

REPO="$(find "${WORKSPACE}" -maxdepth 2 -type d -name ".git" -printf "%h\n" 2>/dev/null | head -n1 || true)"
if [[ -z "${REPO}" ]]; then
  REPO="$(find "${WORKSPACE}" -maxdepth 2 -type d -name "decoding" -printf "%h\n" 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${REPO}" || ! -d "${REPO}" ]]; then
  echo "[save] could not find a cloned repo under ${WORKSPACE}" >&2
  exit 2
fi

WE_SRC="${REPO}/watermark_encoding"
DEC_SRC="${REPO}/decoding"
WE_DST="${WORKSPACE}/watermark_encoding"
DEC_DST="${WORKSPACE}/decoding"

echo "[save] workspace: ${WORKSPACE}"
echo "[save] repo:      ${REPO}"
echo "[save] mode:      ${MODE}"

rm -rf "${WE_DST}" "${DEC_DST}"
mkdir -p "${WE_DST}/data/images" \
         "${WE_DST}/data/baseline" \
         "${WE_DST}/models" \
         "${DEC_DST}/checkpoints" \
         "${DEC_DST}/results/figures"

# transfer helper: hardlink in place when possible, else fall back to copy.
xfer() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  if [[ "${MODE}" == "copy" ]]; then
    cp -r "${src}" "${dst}"
  else
    cp -al "${src}" "${dst}" 2>/dev/null || cp -r "${src}" "${dst}"
  fi
}

# optional: strip optimizer state from .pth files (reduces ViT 983MB -> ~340MB)
if [[ ${INCLUDE_OPTIMIZER} -eq 0 ]]; then
  echo "[save] stripping optimizer_state_dict from checkpoints in repo first"
  python3 - "${REPO}" <<'PY'
import os, sys, glob, torch
repo = sys.argv[1]
ckpts = sorted(
    glob.glob(os.path.join(repo, "decoding/checkpoints/*.pth"))
    + glob.glob(os.path.join(repo, "decoding/checkpoints/separate/bit_*_best.pth"))
)
for path in ckpts:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        before = os.path.getsize(path)
        ckpt.pop("optimizer_state_dict", None)
        torch.save(ckpt, path)
        after = os.path.getsize(path)
        print(f"  {os.path.relpath(path, repo)}: {before/1e6:.0f}MB -> {after/1e6:.0f}MB")
    else:
        print(f"  {os.path.relpath(path, repo)}: no optimizer state, kept as-is")
PY
fi

# ---------- watermark_encoding ----------
if [[ -d "${WE_SRC}/data/images" ]]; then
  echo "[save] watermarked images ..."
  shopt -s nullglob
  pngs=( "${WE_SRC}/data/images/"*.png )
  if [[ ${#pngs[@]} -gt 0 ]]; then
    if [[ "${MODE}" == "copy" ]]; then
      cp -r "${WE_SRC}/data/images/." "${WE_DST}/data/images/"
    else
      cp -al "${WE_SRC}/data/images/." "${WE_DST}/data/images/" 2>/dev/null \
        || cp -r "${WE_SRC}/data/images/." "${WE_DST}/data/images/"
    fi
  fi
fi
if [[ -d "${WE_SRC}/data/baseline" ]]; then
  cp -al "${WE_SRC}/data/baseline/." "${WE_DST}/data/baseline/" 2>/dev/null \
    || cp -r "${WE_SRC}/data/baseline/." "${WE_DST}/data/baseline/"
fi
if [[ -f "${WE_SRC}/data/metadata.json" ]]; then
  cp -f "${WE_SRC}/data/metadata.json" "${WE_DST}/data/metadata.json"
fi
if [[ -d "${WE_SRC}/models" ]]; then
  for d in "${WE_SRC}/models/"watermark_s*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    mkdir -p "${WE_DST}/models/${name}"
    for f in "$d"/*_last.safetensors; do
      [[ -f "$f" ]] && xfer "$f" "${WE_DST}/models/${name}/$(basename "$f")"
    done
    if [[ ${KEEP_500STEPS} -eq 1 ]]; then
      for f in "$d"/*_500steps.safetensors; do
        [[ -f "$f" ]] && xfer "$f" "${WE_DST}/models/${name}/$(basename "$f")"
      done
    fi
  done
fi

# ---------- decoding (artifacts only) ----------
shopt -s nullglob
# checkpoints (top-level + separate/)
for f in "${DEC_SRC}/checkpoints/"*.pth "${DEC_SRC}/checkpoints/"*.json; do
  [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/checkpoints/$(basename "$f")"
done
if [[ -d "${DEC_SRC}/checkpoints/separate" ]]; then
  mkdir -p "${DEC_DST}/checkpoints/separate"
  for f in "${DEC_SRC}/checkpoints/separate/"*.pth "${DEC_SRC}/checkpoints/separate/"*.json; do
    [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/checkpoints/separate/$(basename "$f")"
  done
fi

# results: markdowns, JSONs, figures (skip the empty top-level figures/ if any)
for f in "${DEC_SRC}/results/"*.md "${DEC_SRC}/results/"*.json; do
  [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/results/$(basename "$f")"
done
if [[ -d "${DEC_SRC}/results/test_results" ]]; then
  mkdir -p "${DEC_DST}/results/test_results"
  for f in "${DEC_SRC}/results/test_results/"*.json; do
    [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/results/test_results/$(basename "$f")"
  done
fi
if [[ -d "${DEC_SRC}/results/figures" ]]; then
  for f in "${DEC_SRC}/results/figures/"*.png; do
    [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/results/figures/$(basename "$f")"
  done
  # nested per-model robustness subdirs (e.g. robustness_separate/)
  for d in "${DEC_SRC}/results/figures/"*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    mkdir -p "${DEC_DST}/results/figures/${name}"
    for f in "$d"/*.png; do
      [[ -f "$f" ]] && xfer "$f" "${DEC_DST}/results/figures/${name}/$(basename "$f")"
    done
  done
fi

# ---------- final summary ----------
echo
echo "[save] === produced ==="
du -sh "${WE_DST}/data/images" 2>/dev/null     | sed 's/^/  /' || true
du -sh "${WE_DST}/data/baseline" 2>/dev/null   | sed 's/^/  /' || true
du -sh "${WE_DST}/models" 2>/dev/null          | sed 's/^/  /' || true
du -sh "${DEC_DST}/checkpoints" 2>/dev/null    | sed 's/^/  /' || true
du -sh "${DEC_DST}/results" 2>/dev/null        | sed 's/^/  /' || true

echo
echo "[save] === counts ==="
echo "  watermarked images:    $(find "${WE_DST}/data/images" -name '*.png' 2>/dev/null | wc -l)  (expect 2560)"
echo "  baseline images:       $(find "${WE_DST}/data/baseline" -name '*.png' 2>/dev/null | wc -l)  (expect 10)"
echo "  LoRA safetensors:      $(find "${WE_DST}/models" -name '*.safetensors' 2>/dev/null | wc -l)"
echo "  checkpoint .pth files: $(find "${DEC_DST}/checkpoints" -name '*.pth' 2>/dev/null | wc -l)"
echo "  result markdowns:      $(find "${DEC_DST}/results" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)"
echo "  result figures:        $(find "${DEC_DST}/results/figures" -name '*.png' 2>/dev/null | wc -l)"

# defensive: ensure we never staged code or junk
forbidden="$(find "${DEC_DST}" "${WE_DST}" \
  \( -name "__pycache__" -o -name "*.pyc" -o -name ".smoke" \
     -o -name "*.log" -o -name ".DS_Store" -o -name ".gitkeep" \) 2>/dev/null || true)"
if [[ -n "${forbidden}" ]]; then
  echo "[save] cleaning forbidden entries:"
  echo "${forbidden}" | sed 's/^/    /'
  rm -rf -- ${forbidden}
fi

echo
echo "[save] ready to upload:"
echo "    ${WE_DST}/"
echo "    ${DEC_DST}/"
echo "[save] (drag both folders into your S3 bucket via the Runpod UI)"
