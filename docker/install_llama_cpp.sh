#!/usr/bin/env bash
# Build-time step (docker/base.Dockerfile): bake the llama-cpp-python CUDA wheel that the
# Qwen / CUDA environment repair (launcher/selfism_runtime.py) otherwise downloads on
# every new volume. Same wheel (same URL, pinned by sha256), same --no-deps install, same
# runtime deps the repair adds. The repair is unchanged: on the pod it sees
# llama-cpp-python 0.4.1 (the +cu128 wheel) through the venv's --system-site-packages, runs its GPU
# probe, and only downloads the wheel again if that probe fails.
set -Eeuo pipefail

: "${PYTHON:=python3.12}"
: "${LLAMA_CPP_URL:=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.4.1-cu128-linux-20260926/llama_cpp_python-0.4.1%2Bcu128-cp312-cp312-linux_x86_64.whl}"
: "${LLAMA_CPP_SHA256:=cae7e9082e12ddf377ccbd2d19a9d3c14b86fafb9b9d702d0bd846563c0d1ca9}"
: "${LLAMA_CPP_VERSION:=0.4.1}"  # the wheel metadata (its filename says +cu128)
# What selfism_runtime.RUNTIME_DEPS resolves to; numpy, jinja2 and typing-extensions
# are already in the image and are not touched.
: "${DISKCACHE_VERSION:=5.6.3}"

log() { echo "[bake-llama] $*"; }
PIP=("$PYTHON" -m pip)
torch_before="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
"${PIP[@]}" freeze --disable-pip-version-check | sort > /tmp/llama-freeze-before.txt

wheel_dir="$(mktemp -d)"
wheel="$wheel_dir/llama_cpp_python-0.4.1+cu128-cp312-cp312-linux_x86_64.whl"
LLAMA_CPP_URL="$LLAMA_CPP_URL" LLAMA_CPP_SHA256="$LLAMA_CPP_SHA256" LLAMA_WHEEL="$wheel" "$PYTHON" - << 'PY'
import hashlib, os, urllib.request
from pathlib import Path
dest = Path(os.environ["LLAMA_WHEEL"])
urllib.request.urlretrieve(os.environ["LLAMA_CPP_URL"], dest)
digest = hashlib.sha256()
with dest.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
got, expect = digest.hexdigest(), os.environ["LLAMA_CPP_SHA256"]
if got != expect:
    raise SystemExit("llama-cpp-python wheel sha256 " + got + " != " + expect)
print("llama-cpp-python wheel", dest.stat().st_size, got)
PY
"${PIP[@]}" install --no-cache-dir --disable-pip-version-check --break-system-packages \
  --no-deps "$wheel"
"${PIP[@]}" install --no-cache-dir --disable-pip-version-check --break-system-packages \
  --no-deps "diskcache==$DISKCACHE_VERSION"
rm -rf "$wheel_dir"

LLAMA_CPP_VERSION="$LLAMA_CPP_VERSION" "$PYTHON" - << 'PY'
import importlib.metadata as m, os
from pathlib import Path
version = m.version("llama-cpp-python")
if version != os.environ["LLAMA_CPP_VERSION"]:
    raise SystemExit("llama-cpp-python is " + version)
import diskcache, llama_cpp
import llama_cpp.llama_chat_format as chat
assert hasattr(chat, "Qwen35ChatHandler"), "Qwen35ChatHandler missing"
lib = Path(llama_cpp.__file__).resolve().parent / "lib"
cuda = sorted(p.name for p in lib.glob("libggml-cuda*.so*"))
if not cuda:
    raise SystemExit("llama.cpp CUDA backend missing from " + str(lib))
# The CUDA backend links the CUDA 12 runtime from the nvidia *-cu12 wheels.
need = {"libcudart.so.12", "libcublas.so.12", "libnccl.so.2"}
import sys
found = {p.name for s in sys.path if s for p in Path(s).glob("nvidia/*/lib/*") if p.name in need}
if found != need:
    raise SystemExit("CUDA 12 libraries for llama.cpp missing: " + ", ".join(sorted(need - found)))
print("llama-cpp-python", version, "backend", ", ".join(cuda), "diskcache", diskcache.__version__)
PY

torch_after="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
[ "$torch_after" = "$torch_before" ] || { echo "torch changed during the llama-cpp install: $torch_before -> $torch_after" >&2; exit 1; }
"${PIP[@]}" freeze --disable-pip-version-check | sort > /tmp/llama-freeze-after.txt
changed="$(comm -3 /tmp/llama-freeze-before.txt /tmp/llama-freeze-after.txt | tr -d '\t' | grep -viE '^(llama[-_]cpp[-_]python|diskcache)( @ |==)' || true)"
if [ -n "$changed" ]; then
  echo "the llama-cpp install changed other packages:" >&2; echo "$changed" >&2; exit 1
fi
rm -rf /tmp/llama-freeze-* /root/.cache
log "baked llama-cpp-python $LLAMA_CPP_VERSION (torch still $torch_after)"
