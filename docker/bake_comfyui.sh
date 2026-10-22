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
# The base image may still ship /opt/comfyui-runtime-constraints.txt with cu128 torch
# pins; those must not be merged for torch/torchvision/torchaudio or pip will see
# conflicting ==cu128,==cu130 constraints.
constraints=/tmp/bake-constraints.txt
"${PIP[@]}" freeze --disable-pip-version-check \
  | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|pillow|opencv-[a-z-]+)==' > "$constraints"
if [ -f /opt/comfyui-runtime-constraints.txt ]; then
  grep -E '^[A-Za-z]' /opt/comfyui-runtime-constraints.txt \
    | grep -viE '^(torch|torchvision|torchaudio)==' >> "$constraints" || true
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
# onnxruntime's CUDA provider dlopens libcudart/libcudnn by soname. Torch's cu130
# wheels ship those under site-packages/nvidia/*/lib, which the dynamic linker does
# not search until they are registered. Do it once in the image so cards that never
# run the pod repair (Wan Animate) still load CUDAExecutionProvider.
"$PYTHON" - << 'PY'
from pathlib import Path
import sys
folders = sorted({str(p.resolve()) for s in sys.path if s for p in Path(s).glob("nvidia/*/lib") if p.is_dir()})
if not folders:
    raise SystemExit("torch cu130 did not ship nvidia/*/lib; CUDAExecutionProvider cannot load")
Path("/etc/ld.so.conf.d/comfyui-nvidia.conf").write_text("\n".join(folders) + "\n")
print("ldconfig nvidia libs: " + ", ".join(folders))
PY
ldconfig
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

# 6. RunPod bundle marker: a start.sh that knows about it re-syncs ComfyUI core files from
# the baked tree whenever the marker differs from the one on the volume (it excludes
# models, input, output, user, custom_nodes and .venv*).  The base start.sh shipped in
# this image predates that logic and ignores the file.
cat > "$BAKED/.runpod-bundle-version" <<EOF
COMFYUI_VERSION=$COMFYUI_TAG
COMFYUI_COMMIT=$COMFYUI_SHA
SELFISM_BAKED=1
EOF

# 7. Bake the custom-node packs Krast ORG + Depth and the Hearmeman video cards
# install, plus the small pip deps those packs still resolve at pod boot.
# One codeload tarball per pinned sha: no .git history. example_workflows/ and a
# workflows/ tree that is not Python are removed (RES4LYF is mostly those).
# The launcher copies /opt/10sorlabs/node_cache/<name> when the marker matches and
# only then falls back to a shallow GitHub fetch. requirements_baked means the
# deps below were installed into system site-packages under the cu130 constraints,
# so a copied pack does no pip work.
: "${SELFISM_CATALOG_JSON:=/tmp/selfism-catalog.json}"
NODE_CACHE=/opt/10sorlabs/node_cache
# Keep this list equal to krast_org_depth_nodes plus the two Hearmeman packs.
# tests/test_bake.py checks that.
BAKED_NODE_PACKS=(
  rgthree-comfy
  RES4LYF
  ComfyUI-KJNodes
  ComfyUI-Impact-Pack
  ComfyUI-Impact-Subpack
  ComfyUI-Pixaroma
  ComfyUI-Adaptive-Guidance
  ComfyUI-Crystools
  ComfyUI-1hewNodes
  CRT-Nodes
  ComfyUI_Eclipse
  sd-perturbed-attention
  comfyui-krea2-controlnet
  comfyui_controlnet_aux
  ComfyUI-AutoGrid
  hearmemanai-upscale
  openrouter_node
)
if [ ! -f "$SELFISM_CATALOG_JSON" ]; then
  echo "catalog JSON missing at $SELFISM_CATALOG_JSON; refusing to bake nodes" >&2
  exit 1
fi
mkdir -p "$NODE_CACHE"
BAKED_PACKS_CSV=$(IFS=,; echo "${BAKED_NODE_PACKS[*]}")
PIP_CONSTRAINT="$constraints" SELFISM_CATALOG_JSON="$SELFISM_CATALOG_JSON" \
  NODE_CACHE="$NODE_CACHE" BAKED_PACKS="$BAKED_PACKS_CSV" "$PYTHON" - <<'PY'
import json, os, shutil, tarfile, urllib.request
from pathlib import Path

catalog = json.loads(Path(os.environ["SELFISM_CATALOG_JSON"]).read_text())
names = [part for part in os.environ["BAKED_PACKS"].split(",") if part]
nodes = {node["name"]: node for node in catalog["nodes"]}
missing = [name for name in names if name not in nodes]
if missing:
    raise SystemExit("baked pack list is not in the catalog: " + ", ".join(missing))
cache = Path(os.environ["NODE_CACHE"])
# sam2 is installed later from a pinned commit. opencv-contrib would be a third cv2.
# The CPU distribution `onnxruntime` shares the import name with onnxruntime-gpu.
skip_names = {"torch", "torchvision", "torchaudio", "sam2", "opencv-contrib-python", "opencv-python", "onnxruntime"}
wanted = []

def requirement_names(node, extracted: Path):
    explicit = node.get("requirements_packages")
    if explicit is not None:
        return list(explicit)
    path = extracted / "requirements.txt"
    if not path.is_file():
        return []
    names = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http://", "https://")):
            continue
        base = line.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].split("!", 1)[0].split("~", 1)[0].strip()
        if base.lower().replace("_", "-") in skip_names:
            continue
        names.append(line)
    return names

def strip_examples(root: Path):
    for directory in list(root.rglob("*")):
        if directory.is_dir() and directory.name in {"example_workflows", "example_workflow", ".github"}:
            shutil.rmtree(directory, ignore_errors=True)
    workflows = root / "workflows"
    if workflows.is_dir() and not any(workflows.rglob("*.py")):
        shutil.rmtree(workflows)
    for cache_dir in list(root.rglob("__pycache__")):
        shutil.rmtree(cache_dir, ignore_errors=True)

for name in names:
    node = nodes[name]
    repo = node["repo"]
    ref = node["ref"]
    url = repo.removesuffix(".git").removeprefix("https://github.com/")
    archive_url = f"https://codeload.github.com/{url}/tar.gz/{ref}"
    print(f"[bake-nodes] {name} {ref[:12]}", flush=True)
    work = Path("/tmp/bake-node") / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    archive = work / "source.tar.gz"
    urllib.request.urlretrieve(archive_url, archive)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(work / "extracted", filter="data")
    roots = list((work / "extracted").iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise SystemExit(f"unexpected archive layout for {name}")
    strip_examples(roots[0])
    for package in requirement_names(node, roots[0]):
        if package not in wanted:
            wanted.append(package)
    marker = {
        "repo": repo if repo.endswith(".git") else repo + ".git",
        "ref": ref,
        "source": "image-cache",
        "requirements_baked": True,
    }
    # Repo URLs in the catalog already end in .git. Keep the catalog string.
    marker["repo"] = node["repo"]
    (roots[0] / ".selfism-source.json").write_text(json.dumps(marker), encoding="utf-8")
    destination = cache / name
    shutil.rmtree(destination, ignore_errors=True)
    shutil.move(str(roots[0]), destination)
    if (destination / ".git").exists():
        raise SystemExit(f"{name} bake still has a .git directory")
    shutil.rmtree(work, ignore_errors=True)

aux = cache / "comfyui_controlnet_aux"
if aux.is_dir():
    # mediapipe 1.x tasks API (what the pinned aux face node selects) needs this file.
    # The node does not download it; a missing file raises and the preprocessor returns nothing.
    face = aux / "ckpts" / "mediapipe" / "face_landmarker.task"
    face.parent.mkdir(parents=True, exist_ok=True)
    face_url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    print("[bake-nodes] face_landmarker.task", flush=True)
    urllib.request.urlretrieve(face_url, face)
    if face.stat().st_size < 1_000_000:
        raise SystemExit("face landmarker model is truncated")

# Extra small deps the pinned requirements do not always name, but the packs import.
for package in ("deepdiff", "pynvml", "py-cpuinfo", "pilgram", "soundfile", "pyyaml", "tiktoken", "mpmath"):
    if package not in wanted:
        wanted.append(package)
blocked = [package for package in wanted if package.split("[", 1)[0].split("=", 1)[0].strip().lower().replace("_", "-") in skip_names]
if blocked:
    raise SystemExit("refusing to bake blocked packages: " + ", ".join(blocked))
print("[bake-nodes] pip " + " ".join(wanted), flush=True)
Path("/tmp/bake-node-requirements.txt").write_text("\n".join(wanted) + "\n", encoding="utf-8")
PY
# The CPU distribution and onnxruntime-gpu both install the `onnxruntime` import.
# Remove the CPU one before the requirements file installs onnxruntime-gpu==1.30.0
# (CUDA 13.0 / cuDNN 9, the build that matches torch 2.10.0+cu130).
"$PYTHON" - << 'PY'
import importlib.metadata as m
import subprocess
import sys
names = {dist.metadata["Name"].lower() for dist in m.distributions()}
if "onnxruntime" in names:
    subprocess.check_call([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "--break-system-packages", "--disable-pip-version-check", "onnxruntime",
    ])
PY
if [ -s /tmp/bake-node-requirements.txt ]; then
  PIP_CONSTRAINT="$constraints" "${PIP[@]}" install --no-cache-dir --disable-pip-version-check \
    --break-system-packages -c "$constraints" -r /tmp/bake-node-requirements.txt
fi
# mediapipe>=0.8.0 resolves to 1.1.1 (py3 wheel, cp312). Its metadata requires
# opencv-contrib-python, which would be a third cv2 beside the image's opencv-python
# and opencv-python-headless. controlnet_aux does not call cv2 contrib modules, so
# mediapipe is installed with --no-deps and only its non-OpenCV dependencies.
# albumentationsx is pinned to 2.3.0 in the requirements list: 2.4+ asks for
# numpy>=2.2.6 and torch>=2.13, which would fight the cu130 pin. 2.3.0 still
# imports as albumentations and does not require an OpenCV package.
PIP_CONSTRAINT="$constraints" "${PIP[@]}" install --no-cache-dir --disable-pip-version-check \
  --break-system-packages -c "$constraints" --no-deps 'mediapipe==1.1.1'
PIP_CONSTRAINT="$constraints" "${PIP[@]}" install --no-cache-dir --disable-pip-version-check \
  --break-system-packages -c "$constraints" 'absl-py~=2.3' 'certifi' 'sounddevice~=0.5' 'flatbuffers~=25.9'
# Impact-Pack's requirements.txt is an unpinned git+ HEAD. Bake the commit that
# `git+https://github.com/facebookresearch/sam2` resolves to (repo default branch,
# 2024-12-16) so the SAM2 nodes import. SAM2_BUILD_CUDA=0 skips the optional nvcc
# extension; upstream allows that and inference still runs on torch CUDA. Pod boot
# does not pip this: requirements_baked skips the pack's requirements.txt.
SAM2_BUILD_CUDA=0 PIP_CONSTRAINT="$constraints" "${PIP[@]}" install --no-cache-dir \
  --disable-pip-version-check --break-system-packages --no-build-isolation -c "$constraints" \
  'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
# get_available_providers lists compiled-in providers and does not need a GPU.
"$PYTHON" - << 'PY'
import importlib.metadata as m
import onnxruntime
dists = {dist.metadata["Name"].lower() for dist in m.distributions()}
if "onnxruntime" in dists:
    raise SystemExit("CPU onnxruntime is installed and would shadow onnxruntime-gpu")
if "onnxruntime-gpu" not in dists:
    raise SystemExit("onnxruntime-gpu is not installed")
if m.version("onnxruntime-gpu") != "1.30.0":
    raise SystemExit("onnxruntime-gpu version is " + m.version("onnxruntime-gpu"))
providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider missing from " + repr(providers))
for banned in ("opencv-contrib-python", "opencv-contrib-python-headless"):
    if banned in dists:
        raise SystemExit(banned + " was installed; it would add another cv2")
import mediapipe
import sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
print("onnxruntime", m.version("onnxruntime-gpu"), providers)
print("mediapipe", mediapipe.__version__)
print("sam2", sam2.__file__)
PY
# The cu130 stack must still be the one the ComfyUI bake just verified.
torch_nodes="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
if [ "$torch_nodes" != "$torch_after" ]; then
  echo "torch changed during the node-pack bake: $torch_after -> $torch_nodes" >&2
  exit 1
fi
log "baked node packs: $(du -sh "$NODE_CACHE" | cut -f1) at $NODE_CACHE"

# 8. Clean up build leftovers so the layer only holds the real changes.
rm -rf /tmp/bake-* /root/.cache /root/.gitconfig
find "$BAKED" -name __pycache__ -prune -exec rm -rf -- {} + 2>/dev/null || true
log "done: $(du -sh "$BAKED" | cut -f1) baked at $BAKED"
