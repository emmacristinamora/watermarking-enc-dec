#!/usr/bin/env bash
# End-to-end smoke test for the decoding pipeline.
# Runs every script in --smoke mode against the synthetic 64-image fixture in
# decoding/.smoke/. Designed to finish in under 5 minutes on a MacBook with no
# GPU. Prints PASSED or FAILED per step.

set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECODING_ROOT="$(cd "${HERE}/.." && pwd)"
LOG_DIR="${DECODING_ROOT}/.smoke/logs"
mkdir -p "${LOG_DIR}"

PYTHON="${PYTHON:-python3}"
overall_status=0

run_step() {
  local label="$1"
  shift
  local logfile="${LOG_DIR}/${label}.log"
  echo
  echo "==================================================="
  echo "[smoke] running: ${label}"
  echo "        cmd: $*"
  echo "        log: ${logfile}"
  echo "==================================================="
  if "$@" >"${logfile}" 2>&1; then
    echo "[smoke] ${label}: PASSED"
  else
    echo "[smoke] ${label}: FAILED (see ${logfile})"
    tail -n 20 "${logfile}" || true
    overall_status=1
  fi
}

cd "${DECODING_ROOT}"

# 1. ensure the synthetic fixture exists (idempotent)
run_step "00_fixture" "${PYTHON}" scripts/_smoke_utils.py

# 2. train_separate
run_step "01_train_separate" "${PYTHON}" scripts/train_separate.py --smoke

# 3. train_vit
run_step "02_train_vit" "${PYTHON}" scripts/train_vit.py --smoke

# 4. signal_analysis (skip Grad-CAM by default to keep it under budget; the
#    flag still exercises the analysis dispatch)
run_step "03_signal_analysis" "${PYTHON}" scripts/signal_analysis.py --smoke --skip-gradcam

# 5. robustness_eval
run_step "04_robustness_eval" "${PYTHON}" scripts/robustness_eval.py --smoke --model resnet

# 6. compare_architectures
run_step "05_compare_architectures" "${PYTHON}" scripts/compare_architectures.py --smoke

echo
echo "==================================================="
if [[ "${overall_status}" -eq 0 ]]; then
  echo "[smoke] ALL STEPS PASSED"
else
  echo "[smoke] ONE OR MORE STEPS FAILED"
fi
echo "==================================================="
exit "${overall_status}"
