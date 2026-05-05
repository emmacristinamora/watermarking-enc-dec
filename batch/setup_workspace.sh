#!/usr/bin/env bash
# Run on a fresh Runpod (or any) VM after:
#   1. cloning the repo into /workspace/<repo>/
#   2. downloading the S3 folders watermark_encoding/ and decoding/ into
#      /workspace/watermark_encoding/ and /workspace/decoding/
#
# This script merges the S3 contents into the cloned repo so the training
# scripts find data and checkpoints at their expected paths. Source folders
# under /workspace/ are emptied as files move; idempotent — you can re-run
# after partial failures.
#
# Layout before:
#   /workspace/<repo>/                    git checkout
#   /workspace/watermark_encoding/        from S3 (data/ + models/)
#   /workspace/decoding/                  from S3 (checkpoints/ + results/)
#
# Layout after:
#   /workspace/<repo>/watermark_encoding/data/{images,baseline}/
#                                        /metadata.json
#                    /watermark_encoding/models/watermark_s*/
#   /workspace/<repo>/decoding/checkpoints/...
#                    /decoding/results/...
#
# Use --copy to copy instead of move (keeps S3 staging on disk). Default is
# move (faster on same filesystem, no duplicate disk usage).

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODE="move"          # move | copy
INSTALL_DEPS=0       # 1 to also pip-install missing Python deps
SKIP_VERIFY=0

usage() {
  cat <<EOF
Usage: bash setup_workspace.sh [options]

Options:
  --workspace PATH   Workspace root containing repo + S3 folders (default: /workspace)
  --copy             Hardlink instead of moving (preserves source dirs)
  --install-deps     pip install torch / torchvision / torchcam if missing
  --skip-verify      Skip sanity counts (faster on huge image dirs)
  -h | --help        This help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)    WORKSPACE="$2"; shift 2 ;;
    --copy)         MODE="copy"; shift ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --skip-verify)  SKIP_VERIFY=1; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

# locate the cloned git repo under the workspace
REPO="$(find "${WORKSPACE}" -maxdepth 2 -type d -name ".git" -printf "%h\n" 2>/dev/null | head -n1 || true)"
if [[ -z "${REPO}" ]]; then
  # fallback: any directory in workspace that has a decoding/ subdir
  REPO="$(find "${WORKSPACE}" -maxdepth 2 -type d -name "decoding" -printf "%h\n" 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${REPO}" || ! -d "${REPO}" ]]; then
  echo "[setup] could not find a cloned repo under ${WORKSPACE}" >&2
  exit 2
fi
echo "[setup] workspace: ${WORKSPACE}"
echo "[setup] repo:      ${REPO}"

WE_SRC="${WORKSPACE}/watermark_encoding"
DEC_SRC="${WORKSPACE}/decoding"
WE_DST="${REPO}/watermark_encoding"
DEC_DST="${REPO}/decoding"

if [[ ! -d "${WE_SRC}" && ! -d "${DEC_SRC}" ]]; then
  echo "[setup] neither ${WE_SRC} nor ${DEC_SRC} exists — nothing to import" >&2
  exit 3
fi

# rsync flags: -a preserves timestamps/perms, --remove-source-files turns
# rsync into a "move that merges" instead of a destructive overwrite, and
# --ignore-existing means we never clobber a file already in the repo
# (e.g. you re-ran setup and a checkpoint is already there).
RSYNC_MOVE=(rsync -a --remove-source-files --ignore-existing)
RSYNC_COPY=(rsync -a --link-dest)   # hardlink-copy for --copy mode

import_dir() {
  local src="$1" dst="$2"
  if [[ ! -d "${src}" ]]; then
    echo "[setup] skip ${src} (does not exist)"
    return 0
  fi
  mkdir -p "${dst}"
  if [[ "${MODE}" == "copy" ]]; then
    echo "[setup] hardlink-copy ${src}/ -> ${dst}/"
    cp -al "${src}/." "${dst}/" 2>/dev/null || rsync -a "${src}/" "${dst}/"
  else
    echo "[setup] move ${src}/ -> ${dst}/"
    "${RSYNC_MOVE[@]}" "${src}/" "${dst}/"
    # clean up empty source directories left behind by --remove-source-files
    find "${src}" -type d -empty -delete 2>/dev/null || true
  fi
}

# ---------- watermark_encoding (data + models) ----------
if [[ -d "${WE_SRC}" ]]; then
  echo "[setup] === watermark_encoding ==="
  for sub in data models; do
    if [[ -d "${WE_SRC}/${sub}" ]]; then
      import_dir "${WE_SRC}/${sub}" "${WE_DST}/${sub}"
    fi
  done
  # the version-controlled metadata.json fallback under encoding/data/ — copy
  # the canonical one over it if present, so both locations agree.
  if [[ -f "${WE_DST}/data/metadata.json" && -d "${REPO}/encoding/data" ]]; then
    cp -f "${WE_DST}/data/metadata.json" "${REPO}/encoding/data/metadata.json"
    echo "[setup] synced metadata.json into encoding/data/ for git fallback"
  fi
fi

# ---------- decoding (checkpoints + results only — never overwrite code) ----------
if [[ -d "${DEC_SRC}" ]]; then
  echo "[setup] === decoding ==="
  # whitelist: only these subdirs are permitted to come from S3.
  # everything else (src/, scripts/, configs/, data/splits.json) lives in git.
  for sub in checkpoints results figures; do
    if [[ -d "${DEC_SRC}/${sub}" ]]; then
      import_dir "${DEC_SRC}/${sub}" "${DEC_DST}/${sub}"
    fi
  done
  # warn loudly if S3 had stale code copies
  for stale in src scripts configs; do
    if [[ -d "${DEC_SRC}/${stale}" ]]; then
      echo "[setup] WARNING: ${DEC_SRC}/${stale}/ exists in S3 staging — IGNORED."
      echo "[setup]          Code lives in git, not S3. Delete from S3 next sync."
    fi
  done
fi

# ---------- optional dep install ----------
if [[ ${INSTALL_DEPS} -eq 1 ]]; then
  echo "[setup] checking Python deps ..."
  python3 - <<'PY' || pip install --quiet torch torchvision torchcam pyyaml tqdm matplotlib scikit-learn pillow numpy
import importlib, sys
need = ["torch", "torchvision", "torchcam", "yaml", "tqdm", "matplotlib", "sklearn", "PIL", "numpy"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", missing)
    sys.exit(1)
PY
fi

# ---------- sanity counts ----------
if [[ ${SKIP_VERIFY} -eq 0 ]]; then
  echo
  echo "[setup] === sanity counts ==="
  if [[ -d "${WE_DST}/data/images" ]]; then
    n_img="$(find "${WE_DST}/data/images" -maxdepth 1 -name "*.png" | wc -l)"
    echo "  watermarked images:   ${n_img}  (expect 2560)"
  fi
  if [[ -d "${WE_DST}/data/baseline" ]]; then
    n_base="$(find "${WE_DST}/data/baseline" -maxdepth 1 -name "*.png" | wc -l)"
    echo "  baseline images:      ${n_base}  (expect 10)"
  fi
  if [[ -d "${WE_DST}/models" ]]; then
    n_lora="$(find "${WE_DST}/models" -maxdepth 1 -type d -name "watermark_s*" | wc -l)"
    echo "  LoRA dirs:            ${n_lora}  (expect 8)"
  fi
  if [[ -d "${WE_DST}/data/baseline" ]]; then
    has_meta="$(test -f "${WE_DST}/data/metadata.json" && echo yes || echo MISSING)"
    echo "  metadata.json:        ${has_meta}"
  fi
  if [[ -d "${DEC_DST}/checkpoints" ]]; then
    n_pth="$(find "${DEC_DST}/checkpoints" -name "*.pth" | wc -l)"
    echo "  checkpoint .pth files: ${n_pth}"
  fi
fi

# ---------- final cleanup of empty staging dirs ----------
for d in "${WE_SRC}" "${DEC_SRC}"; do
  if [[ -d "${d}" ]]; then
    if [[ -z "$(ls -A "${d}" 2>/dev/null)" ]]; then
      rmdir "${d}" 2>/dev/null && echo "[setup] removed empty ${d}"
    else
      echo "[setup] note: ${d} still has files left over:"
      find "${d}" -maxdepth 2 -type f | head -5 | sed 's/^/    /'
    fi
  fi
done

echo
echo "[setup] done. you can now run, e.g.:"
echo "    cd ${REPO}/decoding"
echo "    python scripts/train_separate.py --epochs 25 --batch-size 4 --image-size 1024"
