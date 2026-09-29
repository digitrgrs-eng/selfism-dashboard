"""Fixed, auditable repair operation; never accepts arbitrary shell commands."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WHEEL = ('https://github.com/JamePeng/llama-cpp-python/releases/download/'
         'v0.4.1-cu128-linux-20260926/'
         'llama_cpp_python-0.4.1%2Bcu128-cp312-cp312-linux_x86_64.whl')

VERIFY = '''
import ctypes
from pathlib import Path
import torch
import llama_cpp
import llama_cpp.llama_chat_format as chat
from llama_cpp import _ggml
assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"
assert hasattr(chat, "Qwen35ChatHandler"), "Qwen35ChatHandler missing"
folder = Path(llama_cpp.__file__).resolve().parent / "lib"
llama_cpp.llama_backend_init()
_ggml.ggml_backend_load_all_from_path(ctypes.c_char_p(str(folder).encode()))
assert llama_cpp.llama_supports_gpu_offload(), "llama.cpp GPU offload unavailable"
print("Qwen35ChatHandler: True", flush=True)
print("GPU offload: True", flush=True)
print("GPU:", torch.cuda.get_device_name(), flush=True)
'''

def cuda_folders():
    return sorted({str(p.resolve()) for s in sys.path if s
                   for p in Path(s).glob('nvidia/*/lib') if p.is_dir()})

def run(command, **kwargs):
    print('Running: ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3,12):
        raise SystemExit('This repair targets Linux x86_64 / Python 3.12 / CUDA 12.8 only.')
    folders = cuda_folders()
    if not folders:
        raise SystemExit('No existing NVIDIA libraries found. Repair the base PyTorch environment first.')
    for filename in ('libcublas.so.12', 'libnccl.so.2'):
        if not any((Path(p)/filename).exists() for p in folders):
            raise SystemExit('Missing '+filename+'; no blind CUDA package upgrade will be attempted.')
    config = Path('/etc/ld.so.conf.d/comfyui-nvidia.conf')
    text = '\n'.join(folders)+'\n'
    if config.exists() and config.read_text() != text:
        backup = config.with_suffix('.conf.pre-selfism')
        if not backup.exists(): backup.write_text(config.read_text())
    config.write_text(text)
    run(['ldconfig'], timeout=60)
    print('CUDA library paths configured.', flush=True)
    if args.paths_only: return
    before = importlib.metadata.version('torch')
    try: version = importlib.metadata.version('llama-cpp-python')
    except importlib.metadata.PackageNotFoundError: version = ''
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    probe = subprocess.run([sys.executable, '-c', VERIFY], env=env, timeout=120)
    if version != '0.4.1+cu128' or probe.returncode:
        run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
             '--force-reinstall', '--no-deps', '--timeout', '30', '--retries', '3', WHEEL], timeout=1800)
    if importlib.metadata.version('torch') != before:
        raise SystemExit('PyTorch version changed unexpectedly.')
    run([sys.executable, '-c', VERIFY], env=env, timeout=120)
    marker=Path(sys.prefix)/'.selfism-runtime.json'
    marker.write_text(json.dumps({'llama_cpp':importlib.metadata.version('llama-cpp-python'),
                                  'torch':before,'gpu_verified':True}))
    print('Environment verified. Restart ComfyUI to use the updated library.', flush=True)

if __name__ == '__main__': main()
