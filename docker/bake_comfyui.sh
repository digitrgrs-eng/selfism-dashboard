#!/usr/bin/env bash
# Build-time step: bake ComfyUI at a pinned release into the 10sorLabs/RunPod base image.
#
# Runs inside the base image (root, python3.12 + system site-packages already hold a
# CUDA PyTorch stack — typically cu128).  We OVERRIDE torch/torchvision/torchaudio to
# the cu130 wheels so ComfyUI v0.38.1 can enable comfy_kitchen's CUDA backend
# (disabled when torch.version.cuda < 13.0).  Result:
#   * /opt/comfyui-baked      = git checkout of $COMFYUI_TAG (commit $COMFYUI_SHA),
#                               origin -> upstream, existing custom_nodes/ and user/ kept
#   * system site-packages    = torch 2.10.0+cu130 (+ matching vision/audio), then
#                               ComfyUI's requirements with those packages constrained
#   * /opt/comfyui-baked/.runpod-bundle-version updated (RunPod bundle marker)
# Nothing is written to /workspace here: the base start.sh copies the baked tree on first boot.
# Note: the runtime venv path may still be named .venv-cu128 (historical); torch itself is cu130.
set -Eeuo pipefail

: "${COMFYUI_TAG:=v0.38.1}"
: "${COMFYUI_SHA:=20ca544ee0436721d8eb5f544665e490609f72c8}"
: "${COMFYUI_UPSTREAM:=https://github.com/Comfy-Org/ComfyUI.git}"
: "${BAKED:=/opt/comfyui-baked}"
: "${PYTHON:=python3.12}"
: "${SKIP_SMOKE:=0}"

log() { echo "[bake-comfyui] $*"; }

PIP=("$PYTHON" -m pip)
site="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
log "python: $("$PYTHON" --version); site-packages: $site"

# 1. Override the base CUDA PyTorch stack (usually cu128) with cu130, then pin it.
# ComfyUI v0.38.1 disables comfy_kitchen CUDA when torch.version.cuda < 13.0.
torch_base="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
[ -n "$torch_base" ] || { echo "no torch found"; exit 1; }
log "base torch: $torch_base — upgrading to 2.10.0+cu130"
"${PIP[@]}" install --no-cache-dir --disable-pip-version-check --break-system-packages \
  --upgrade \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

# Constraints: hold every package that ties ComfyUI to the (now cu130) CUDA build.
constraints=/tmp/bake-constraints.txt
"${PIP[@]}" freeze --disable-pip-version-check \
  | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|pillow|opencv-[a-z-]+)==' > "$constraints"
if [ -f /opt/comfyui-runtime-constraints.txt ]; then
  grep -E '^[A-Za-z]' /opt/comfyui-runtime-constraints.txt >> "$constraints" || true
fi
sort -u "$constraints" -o "$constraints"
log "holding:"; sed 's/^/  /' "$constraints"
torch_before="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
case "$torch_before" in
  *cu130*) ;;
  *) echo "expected cu130 torch after upgrade, got: $torch_before" >&2; exit 1 ;;
esac
cuda_meta="$("$PYTHON" -c 'import torch; print(torch.version.cuda or "")')"
case "$cuda_meta" in
  13*) ;;
  *) echo "expected torch.version.cuda starting with 13 after upgrade, got: $cuda_meta" >&2; exit 1 ;;
esac
log "cu130 torch ready: $torch_before (cuda $cuda_meta)"
"${PIP[@]}" freeze --disable-pip-version-check > /tmp/bake-freeze-before.txt

# 2. ComfyUI source: a real git checkout of the tag, remote = upstream.
rm -rf /tmp/comfyui-src
git init -q /tmp/comfyui-src
git -C /tmp/comfyui-src remote add origin "$COMFYUI_UPSTREAM"
git -C /tmp/comfyui-src fetch -q --depth 1 origin "refs/tags/$COMFYUI_TAG:refs/tags/$COMFYUI_TAG"
resolved="$(git -C /tmp/comfyui-src rev-parse "refs/tags/$COMFYUI_TAG^{commit}")"
if [ "$resolved" != "$COMFYUI_SHA" ]; then
  echo "Tag $COMFYUI_TAG resolves to $resolved, expected $COMFYUI_SHA; refusing." >&2
  exit 1
fi
git -C /tmp/comfyui-src checkout -q --detach "refs/tags/$COMFYUI_TAG"

# Replace the old baked ComfyUI, keeping what is not part of ComfyUI itself
# (image-managed custom nodes and the pre-populated Manager cache).
mkdir -p "$BAKED"
find "$BAKED" -mindepth 1 -maxdepth 1 ! -name custom_nodes ! -name user -exec rm -rf -- {} +
mkdir -p "$BAKED/custom_nodes" "$BAKED/user"
cp -a /tmp/comfyui-src/. "$BAKED/"
rm -rf /tmp/comfyui-src
git config --global --add safe.directory "$BAKED"
# The bundle marker is image metadata, not part of the checkout: keep `git status` clean.
echo '/.runpod-bundle-version' >> "$BAKED/.git/info/exclude"
git -C "$BAKED" remote get-url origin | grep -qx "$COMFYUI_UPSTREAM"
[ "$(git -C "$BAKED" rev-parse HEAD)" = "$COMFYUI_SHA" ]
[ "$(git -C "$BAKED" describe --tags --exact-match)" = "$COMFYUI_TAG" ]
log "checked out $COMFYUI_TAG ($COMFYUI_SHA)"

# 3. Requirements into system site-packages (the venv created at boot sees them).
PIP_CONSTRAINT="$constraints" "${PIP[@]}" install --no-cache-dir --disable-pip-version-check \
  --break-system-packages -c "$constraints" -r "$BAKED/requirements.txt"

# 4. Guard rails: the cu130 CUDA stack must be untouched and the pins must hold.
torch_after="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
if [ "$torch_after" != "$torch_before" ]; then
  echo "torch changed during the ComfyUI requirements install: $torch_before -> $torch_after" >&2
  exit 1
fi
case "$torch_after" in
  *cu130*) ;;
  *) echo "expected cu130 torch after requirements, got: $torch_after" >&2; exit 1 ;;
esac
cuda_after="$("$PYTHON" -c 'import torch; print(torch.version.cuda or "")')"
case "$cuda_after" in
  13*) ;;
  *) echo "expected torch.version.cuda starting with 13, got: $cuda_after" >&2; exit 1 ;;
esac
"${PIP[@]}" freeze --disable-pip-version-check > /tmp/bake-freeze-after.txt
changed_protected="$(diff <(grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|pillow|opencv-[a-z-]+)==' /tmp/bake-freeze-before.txt) \
                          <(grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|pillow|opencv-[a-z-]+)==' /tmp/bake-freeze-after.txt) || true)"
if [ -n "$changed_protected" ]; then
  echo "protected packages changed:" >&2; echo "$changed_protected" >&2; exit 1
fi
log "torch intact: $torch_after (cuda $cuda_after)"
log "package changes (freeze diff):"
diff /tmp/bake-freeze-before.txt /tmp/bake-freeze-after.txt | grep '^[<>]' | sed 's/^/  /' || true

# The same check selfism_int8.py makes at boot: source supports int8_tensorwise and
# comfy-kitchen is new enough -> the boot hook becomes a no-op.
"$PYTHON" - "$BAKED" <<'PY'
import importlib.metadata as m, sys
from pathlib import Path
root = Path(sys.argv[1])
kitchen = m.version('comfy-kitchen')
parts = tuple(int(p) for p in kitchen.split('+')[0].split('.')[:3])
assert parts >= (0, 2, 16), 'comfy-kitchen too old: ' + kitchen
assert any('int8_tensorwise' in (root / f).read_text(errors='replace') for f in ('comfy/quant_ops.py', 'comfy/ops.py')), 'int8_tensorwise missing'
print('comfy-kitchen', kitchen, '- int8_tensorwise present')
PY

# 5. Smoke test on a throw-away copy (keeps the baked tree free of runtime files):
# imports torch + ComfyUI core on CPU, loads all built-in nodes, then exits.
if [ "$SKIP_SMOKE" != "1" ]; then
  rm -rf /tmp/comfy-smoke /tmp/comfy-smoke-user
  mkdir -p /tmp/comfy-smoke-user
  cp -a "$BAKED" /tmp/comfy-smoke
  (cd /tmp/comfy-smoke && timeout 600 "$PYTHON" main.py --cpu --quick-test-for-ci --disable-all-custom-nodes \
      --disable-api-nodes --user-directory /tmp/comfy-smoke-user --database-url sqlite:///:memory:) \
    || { echo "ComfyUI smoke test FAILED" >&2; exit 1; }
  rm -rf /tmp/comfy-smoke /tmp/comfy-smoke-user
  log "ComfyUI smoke test passed"
fi

# 5b. Seed user defaults so a first-boot `cp -r /opt/comfyui-baked` already has preview=none.
# entrypoint also re-applies these on every boot for existing volumes (see apply_preview_defaults.py).
mkdir -p "$BAKED/user/default" "$BAKED/user/__manager"
cat > "$BAKED/user/default/comfy.settings.json" <<'JSON'
{
  "Comfy.Execution.PreviewMethod": "none"
}
JSON
cat > "$BAKED/user/__manager/config.ini" <<'INI'
[default]
preview_method = none
INI

# 6. RunPod bundle marker: a start.sh that knows about it re-syncs ComfyUI core files from
# the baked tree whenever the marker differs from the one on the volume (it excludes
# models, input, output, user, custom_nodes and .venv*).  The base start.sh shipped in
# this image predates that logic and ignores the file.
cat > "$BAKED/.runpod-bundle-version" <<EOF
COMFYUI_VERSION=$COMFYUI_TAG
COMFYUI_COMMIT=$COMFYUI_SHA
SELFISM_BAKED=1
EOF

# 7. Clean up build leftovers so the layer only holds the real changes.
rm -rf /tmp/bake-* /root/.cache /root/.gitconfig
find "$BAKED" -name __pycache__ -prune -exec rm -rf -- {} + 2>/dev/null || true
log "done: $(du -sh "$BAKED" | cut -f1) baked at $BAKED"
