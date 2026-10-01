#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PID=""
LAUNCHER_PID=""
BOOT_PID=""

shutdown() {
  trap - SIGTERM SIGINT EXIT
  if [[ -n "$BOOT_PID" ]]; then
    kill -TERM "$BOOT_PID" 2>/dev/null || true
  fi
  if [[ -n "$LAUNCHER_PID" ]]; then
    kill -TERM "$LAUNCHER_PID" 2>/dev/null || true
  fi
  if [[ -n "$BASE_PID" ]]; then
    kill -TERM "$BASE_PID" 2>/dev/null || true
  fi
  wait "$BOOT_PID" "$LAUNCHER_PID" "$BASE_PID" 2>/dev/null || true
}

# On a signal, clean up and leave; otherwise a SIGTERM during the boot update would fall
# through and start ComfyUI.
trap 'shutdown; exit 143' SIGTERM SIGINT
trap shutdown EXIT

export PYTHONPATH="/opt/10sorlabs${PYTHONPATH:+:$PYTHONPATH}"
cd /workspace/runpod-slim

COMFYUI_VENV="/workspace/runpod-slim/ComfyUI/.venv-cu128"
if [[ -d "$COMFYUI_VENV" && ! -f "$COMFYUI_VENV/bin/activate" ]]; then
  echo "Removing an incomplete ComfyUI environment from an interrupted first start..."
  rm -rf -- "$COMFYUI_VENV"
fi

# The dashboard comes up first so the pod is reachable while ComfyUI is being updated.
echo "Starting 10sorLabs Model Grabber on port ${LAUNCHER_PORT:-3000}..."
python3.12 -m launcher.bootstrap &
LAUNCHER_PID=$!

# Boot-time ComfyUI update. The base image starts ComfyUI from its own script, so the
# update has to finish BEFORE that script runs: ComfyUI then starts once, already on the
# pinned release. Idempotent (a quick no-op when up to date), time-limited, logs to
# /workspace/selfism-boot.log, and can never fail the boot. Disable with
# SELFISM_AUTO_COMFY_UPDATE=0.
# Run in the background + wait so a pod stop (SIGTERM) is handled immediately.
python3.12 -m launcher.selfism_boot &
BOOT_PID=$!
wait "$BOOT_PID" || echo "ComfyUI boot update hook exited abnormally; continuing."
BOOT_PID=""

echo "Starting stock RunPod ComfyUI services..."
/usr/local/bin/runpod-base-start.sh &
BASE_PID=$!

wait -n "$BASE_PID" "$LAUNCHER_PID"
