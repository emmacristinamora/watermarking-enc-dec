#!/usr/bin/env bash
# Stage and commit decoding-pipeline results back to the current branch
# from a training VM. Models (*.pth), smoke fixtures, logs, and bytecode
# are deliberately excluded; large artifacts belong in S3, not git.
#
# Examples:
#   bash decoding/scripts/commit_results.sh -m "ViT-B/16 final results"
#   bash decoding/scripts/commit_results.sh --dry-run
#   bash decoding/scripts/commit_results.sh --no-figures -m "tables only"
#   bash decoding/scripts/commit_results.sh --no-push    -m "stage locally"

set -euo pipefail

usage() {
  cat <<EOF
Usage: bash decoding/scripts/commit_results.sh [options]

Required:
  -m MSG          Commit message (omit only with --dry-run)

Optional:
  --no-figures    Skip results/figures/*.png (force-added by default)
  --no-pull       Skip git pull --rebase before staging
  --no-push       Commit locally but do not push
  --dry-run       Show what would be staged, then unstage
  -h | --help     This help
EOF
  exit "${1:-1}"
}

MSG=""
INCLUDE_FIGURES=1
DO_PULL=1
DO_PUSH=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)           MSG="${2:-}"; shift 2 ;;
    --no-figures) INCLUDE_FIGURES=0; shift ;;
    --no-pull)    DO_PULL=0; shift ;;
    --no-push)    DO_PUSH=0; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[commit_results] repo:   ${REPO_ROOT}"
echo "[commit_results] branch: ${BRANCH}"

if [[ "${BRANCH}" == "main" || "${BRANCH}" == "master" ]]; then
  echo "[commit_results] refusing to commit directly to ${BRANCH}." >&2
  exit 2
fi

if [[ ${DO_PULL} -eq 1 ]]; then
  echo "[commit_results] git pull --rebase origin ${BRANCH}"
  git pull --rebase origin "${BRANCH}"
fi

# Inform (don't fail) if any .pth files exist that we will *not* be staging.
existing_pth="$(find decoding -name "*.pth" -not -path "decoding/.smoke/*" 2>/dev/null || true)"
if [[ -n "${existing_pth}" ]]; then
  echo "[commit_results] note: the following .pth files exist locally but will be excluded:"
  echo "${existing_pth}" | sed 's/^/    /'
  echo "[commit_results] (sync them to S3 if you want them preserved)"
fi

# Stage tracked categories. `|| true` keeps the script alive when a glob
# matches nothing (e.g. no JSON results yet on a fresh VM).
echo "[commit_results] staging markdown + JSON results, configs, source ..."
shopt -s nullglob

stage_if_any() {
  local files=("$@")
  if [[ ${#files[@]} -gt 0 ]]; then
    git add -- "${files[@]}"
  fi
}

stage_if_any decoding/results/*.md
stage_if_any decoding/results/*.json
stage_if_any decoding/results/test_results/*.json
stage_if_any decoding/configs/*.yaml

# Source/script changes are tracked already; safe to add the directories.
git add decoding/src decoding/scripts 2>/dev/null || true

if [[ ${INCLUDE_FIGURES} -eq 1 ]]; then
  figs=( decoding/results/figures/*.png decoding/figures/*.png )
  if [[ ${#figs[@]} -gt 0 ]]; then
    echo "[commit_results] force-staging ${#figs[@]} figure(s) (gitignored by default)"
    git add -f -- "${figs[@]}"
  fi
fi

# Final safety net: never let a .pth slip through.
if git diff --cached --name-only | grep -E '\.pth$' >/dev/null; then
  echo "[commit_results] ERROR: a .pth file ended up staged. unstaging and aborting." >&2
  git diff --cached --name-only | grep -E '\.pth$' >&2
  git reset >/dev/null
  exit 3
fi

echo
echo "[commit_results] staged changes:"
git diff --cached --stat || true
echo

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[commit_results] dry-run: unstaging."
  git reset >/dev/null
  exit 0
fi

if [[ -z "${MSG}" ]]; then
  echo "[commit_results] ERROR: -m \"<message>\" is required to commit." >&2
  git reset >/dev/null
  exit 1
fi

if git diff --cached --quiet; then
  echo "[commit_results] nothing to commit. exiting cleanly."
  exit 0
fi

git commit -m "${MSG}"

if [[ ${DO_PUSH} -eq 1 ]]; then
  echo "[commit_results] git push origin ${BRANCH}"
  git push origin "${BRANCH}"
fi

echo "[commit_results] done."
