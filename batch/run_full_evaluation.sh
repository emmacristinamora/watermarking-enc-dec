#!/usr/bin/env bash
# Run every evaluation script in sequence at native 1024x1024 and save the
# outputs into deterministic locations under decoding/results/. After it
# finishes, every number/figure referenced by the markdown reports exists at
# a path the report can link to.
#
# Outputs (relative to decoding/):
#   results/architecture_comparison.md            (3-way table)
#   results/figures/architecture_comparison.png   (grouped bar chart)
#   results/robustness_resnet.json                (shared ResNet, 6 attacks)
#   results/robustness_separate.json              (8x separate, 6 attacks)
#   results/robustness_vit.json                   (ViT-B/16, 6 attacks)
#   results/figures/robustness_resnet/{per_bit,jpeg_curve}.png
#   results/figures/robustness_separate/{per_bit,jpeg_curve}.png
#   results/figures/robustness_vit/{per_bit,jpeg_curve}.png
#   results/figures/comparison_prompt_{00..09}.png
#   results/figures/diff_slider_{0..7}.png
#   results/figures/fft_prompt_{00..N}.png
#   results/figures/gradcam_per_bit.png
#   .full_eval/logs/<step>.log                    (one log per step)
#   .full_eval/manifest.json                      (start/end/duration per step)
#
# Usage:
#   bash decoding/scripts/run_full_evaluation.sh                # full run
#   bash decoding/scripts/run_full_evaluation.sh --skip-signal  # skip signal analysis
#   bash decoding/scripts/run_full_evaluation.sh --skip-vit     # skip ViT
#   bash decoding/scripts/run_full_evaluation.sh --batch-size 8 # bump batch (faster on big GPU)
#
# Designed for the GPU VM (CUDA), but also runs on Apple MPS / CPU. Runs are
# resumable: each step is skipped if its output already exists; pass --force
# to redo everything.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${DECODING}/.." && pwd)"

BATCH_SIZE=4
NUM_WORKERS=2
SKIP_SIGNAL=0
SKIP_COMPARE=0
SKIP_ROB_RESNET=0
SKIP_ROB_SEPARATE=0
SKIP_ROB_VIT=0
FORCE=0

# Optional explicit data overrides (default: rely on default_data_paths()
# pointing at watermark_encoding/ in the repo root).
META=""
IMAGES=""
SPLITS="${DECODING}/data/splits.json"

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [options]

Options:
  --batch-size N           Pass to all eval scripts (default: ${BATCH_SIZE})
  --num-workers N          DataLoader workers (default: ${NUM_WORKERS})
  --metadata PATH          Override metadata.json
  --images   PATH          Override watermarked images dir
  --splits   PATH          Override splits.json (default: decoding/data/splits.json)
  --force                  Re-run every step even if output exists
  --skip-signal            Skip signal_analysis.py
  --skip-compare           Skip compare_architectures.py
  --skip-rob-resnet        Skip robustness_eval --model resnet
  --skip-rob-separate      Skip robustness_eval --model separate
  --skip-rob-vit           Skip robustness_eval --model vit
  -h | --help              This help
EOF
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)        NUM_WORKERS="$2"; shift 2 ;;
    --metadata)           META="$2"; shift 2 ;;
    --images)             IMAGES="$2"; shift 2 ;;
    --splits)             SPLITS="$2"; shift 2 ;;
    --force)              FORCE=1; shift ;;
    --skip-signal)        SKIP_SIGNAL=1; shift ;;
    --skip-compare)       SKIP_COMPARE=1; shift ;;
    --skip-rob-resnet)    SKIP_ROB_RESNET=1; shift ;;
    --skip-rob-separate)  SKIP_ROB_SEPARATE=1; shift ;;
    --skip-rob-vit)       SKIP_ROB_VIT=1; shift ;;
    -h|--help)            usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

cd "${DECODING}"
LOG_DIR="${DECODING}/.full_eval/logs"
MANIFEST="${DECODING}/.full_eval/manifest.json"
mkdir -p "${LOG_DIR}" "${DECODING}/results/figures"

PYTHON="${PYTHON:-python3}"

# ---------- preflight ----------
echo "[run] preflight checks"
DEFAULT_META="${REPO}/watermark_encoding/data/metadata.json"
[[ -z "${META}" && -f "${DEFAULT_META}" ]] && META="${DEFAULT_META}"
[[ -z "${META}" && -f "${REPO}/encoding/data/metadata.json" ]] && META="${REPO}/encoding/data/metadata.json"
DEFAULT_IMAGES="${REPO}/watermark_encoding/data/images"
[[ -z "${IMAGES}" && -d "${DEFAULT_IMAGES}" ]] && IMAGES="${DEFAULT_IMAGES}"

if [[ ! -f "${META}" ]]; then
  echo "[run] ERROR: cannot find metadata.json (looked at ${DEFAULT_META} and the encoding/data fallback)" >&2
  exit 2
fi
if [[ ! -d "${IMAGES}" ]]; then
  echo "[run] ERROR: cannot find images dir at ${DEFAULT_IMAGES}" >&2
  exit 2
fi
echo "[run] metadata: ${META}"
echo "[run] images:   ${IMAGES}  ($(find "${IMAGES}" -maxdepth 1 -name '*.png' | wc -l) PNGs)"
echo "[run] splits:   ${SPLITS}"

CKPT_RESNET="${DECODING}/checkpoints/baseline_resnet50.pth"
CKPT_SEPARATE="${DECODING}/checkpoints/separate"
CKPT_VIT="${DECODING}/checkpoints/vit_best.pth"
[[ -f "${CKPT_RESNET}" ]]   && echo "[run] resnet ckpt:   ${CKPT_RESNET}"   || echo "[run] WARN: missing ${CKPT_RESNET}"
[[ -d "${CKPT_SEPARATE}" ]] && echo "[run] separate dir:  ${CKPT_SEPARATE}" || echo "[run] WARN: missing ${CKPT_SEPARATE}"
[[ -f "${CKPT_VIT}" ]]      && echo "[run] vit ckpt:      ${CKPT_VIT}"      || echo "[run] WARN: missing ${CKPT_VIT}"

# ---------- helpers ----------
declare -a STEPS_RUN=()
declare -a STEPS_SKIPPED=()
declare -a STEPS_TIMING=()
overall_status=0

start_manifest() {
  printf '{\n  "started_at": "%s",\n  "steps": [\n' "$(date -u +%FT%TZ)" > "${MANIFEST}"
}

append_manifest() {
  local label="$1" status="$2" duration="$3" output="$4" sep="$5"
  printf '    %s{ "step": "%s", "status": "%s", "duration_seconds": %s, "output": "%s" }\n' \
    "${sep}" "${label}" "${status}" "${duration}" "${output}" >> "${MANIFEST}"
}

close_manifest() {
  printf '  ],\n  "ended_at": "%s",\n  "overall_status": %d\n}\n' "$(date -u +%FT%TZ)" "${overall_status}" >> "${MANIFEST}"
}

run_step() {
  # run_step <label> <output_to_check> <skip_flag> <command...>
  local label="$1" output="$2" skip_flag="$3"; shift 3
  local logfile="${LOG_DIR}/${label}.log"
  if [[ "${skip_flag}" == "1" ]]; then
    echo "[run] SKIP ${label} (--skip-${label#*_})"
    STEPS_SKIPPED+=("${label}")
    append_manifest "${label}" "skipped" 0 "${output}" "${MANIFEST_SEP}"
    MANIFEST_SEP=","
    return 0
  fi
  if [[ "${FORCE}" -eq 0 && -e "${output}" ]]; then
    echo "[run] SKIP ${label} (output exists: ${output} — pass --force to redo)"
    STEPS_SKIPPED+=("${label}")
    append_manifest "${label}" "cached" 0 "${output}" "${MANIFEST_SEP}"
    MANIFEST_SEP=","
    return 0
  fi
  echo
  echo "==================================================="
  echo "[run] STEP ${label}"
  echo "      cmd: $*"
  echo "      log: ${logfile}"
  echo "==================================================="
  local t0 t1 dt
  t0="$(date +%s)"
  if "$@" >"${logfile}" 2>&1; then
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[run] ${label}: PASSED (${dt}s)"
    STEPS_RUN+=("${label}")
    STEPS_TIMING+=("${dt}")
    append_manifest "${label}" "passed" "${dt}" "${output}" "${MANIFEST_SEP}"
  else
    t1="$(date +%s)"; dt=$(( t1 - t0 ))
    echo "[run] ${label}: FAILED (${dt}s) — see ${logfile}"
    tail -n 25 "${logfile}" || true
    overall_status=1
    append_manifest "${label}" "failed" "${dt}" "${output}" "${MANIFEST_SEP}"
  fi
  MANIFEST_SEP=","
}

# ---------- pipeline ----------
start_manifest
MANIFEST_SEP=""

# 1. signal analysis
SIGNAL_OUTPUT="${DECODING}/results/figures/gradcam_per_bit.png"
run_step "01_signal_analysis" "${SIGNAL_OUTPUT}" "${SKIP_SIGNAL}" \
  "${PYTHON}" scripts/signal_analysis.py \
    --metadata "${META}" --images "${IMAGES}" \
    --output-dir "${DECODING}/results/figures" \
    --checkpoint "${CKPT_RESNET}"

# 2. compare_architectures
COMPARE_MD="${DECODING}/results/architecture_comparison.md"
run_step "02_compare_architectures" "${COMPARE_MD}" "${SKIP_COMPARE}" \
  "${PYTHON}" scripts/compare_architectures.py \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --resnet-checkpoint "${CKPT_RESNET}" \
    --separate-checkpoint-dir "${CKPT_SEPARATE}" \
    --vit-checkpoint "${CKPT_VIT}" \
    --report-md "${COMPARE_MD}" \
    --chart-png "${DECODING}/results/figures/architecture_comparison.png"

# 3-5. robustness per architecture (separate output dirs so figures don't collide)
ROB_RESNET_JSON="${DECODING}/results/robustness_resnet.json"
ROB_RESNET_DIR="${DECODING}/results/figures/robustness_resnet"
mkdir -p "${ROB_RESNET_DIR}"
run_step "03_robustness_resnet" "${ROB_RESNET_JSON}" "${SKIP_ROB_RESNET}" \
  "${PYTHON}" scripts/robustness_eval.py \
    --model resnet --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --resnet-checkpoint "${CKPT_RESNET}" \
    --output-dir "${ROB_RESNET_DIR}" \
    --results-json "${ROB_RESNET_JSON}"

ROB_SEP_JSON="${DECODING}/results/robustness_separate.json"
ROB_SEP_DIR="${DECODING}/results/figures/robustness_separate"
mkdir -p "${ROB_SEP_DIR}"
run_step "04_robustness_separate" "${ROB_SEP_JSON}" "${SKIP_ROB_SEPARATE}" \
  "${PYTHON}" scripts/robustness_eval.py \
    --model separate --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --separate-checkpoint-dir "${CKPT_SEPARATE}" \
    --output-dir "${ROB_SEP_DIR}" \
    --results-json "${ROB_SEP_JSON}"

ROB_VIT_JSON="${DECODING}/results/robustness_vit.json"
ROB_VIT_DIR="${DECODING}/results/figures/robustness_vit"
mkdir -p "${ROB_VIT_DIR}"
run_step "05_robustness_vit" "${ROB_VIT_JSON}" "${SKIP_ROB_VIT}" \
  "${PYTHON}" scripts/robustness_eval.py \
    --model vit --batch-size 16 --num-workers "${NUM_WORKERS}" \
    --metadata "${META}" --images "${IMAGES}" --splits "${SPLITS}" \
    --vit-checkpoint "${CKPT_VIT}" \
    --output-dir "${ROB_VIT_DIR}" \
    --results-json "${ROB_VIT_JSON}"

close_manifest

# ---------- summary ----------
echo
echo "==================================================="
echo "[run] SUMMARY"
echo "==================================================="
echo "ran:     ${STEPS_RUN[*]:-(none)}"
echo "skipped: ${STEPS_SKIPPED[*]:-(none)}"
echo "manifest: ${MANIFEST}"
echo
echo "outputs to feed back into the report writer:"
for f in \
    "${COMPARE_MD}" \
    "${ROB_RESNET_JSON}" "${ROB_SEP_JSON}" "${ROB_VIT_JSON}" \
    "${DECODING}/results/figures/architecture_comparison.png" \
    "${DECODING}/results/figures/gradcam_per_bit.png"; do
  if [[ -e "${f}" ]]; then
    printf "  ✓ %-65s (%s)\n" "${f#${REPO}/}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
  else
    printf "  ✗ %-65s (missing)\n" "${f#${REPO}/}"
  fi
done
echo
n_fig=$(find "${DECODING}/results/figures" -maxdepth 1 -name '*.png' | wc -l)
echo "  ${n_fig} top-level signal/compare figures in results/figures/"
n_rob=$(find "${DECODING}/results/figures" -maxdepth 2 -path '*/robustness_*' -name '*.png' | wc -l)
echo "  ${n_rob} robustness figures in results/figures/robustness_*/"

if [[ "${overall_status}" -eq 0 ]]; then
  echo "[run] ALL STEPS PASSED OR CACHED"
else
  echo "[run] ONE OR MORE STEPS FAILED — inspect ${LOG_DIR}/"
fi
exit "${overall_status}"
