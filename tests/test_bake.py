"""Static checks for the ComfyUI bake (Dockerfile RUN + docker/bake_comfyui.sh).

The real bake needs the ~11 GiB CUDA base rootfs, so it is verified by the image build;
these tests keep the pieces consistent and catch the mistakes that would only show up
after a 10-minute build or on a pod.
"""
import importlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / 'Dockerfile').read_text()
SCRIPT = ROOT / 'docker/bake_comfyui.sh'
WORKFLOW = (ROOT / '.github/workflows/build-image.yml').read_text()
int8 = importlib.import_module('launcher.selfism_int8')


def dockerfile_arg(name):
    return re.search(r'^ARG %s=(\S+)$' % name, DOCKERFILE, re.M).group(1)


def test_baked_tag_and_sha_match_the_boot_hook_target():
    assert dockerfile_arg('COMFYUI_TAG') == int8.TARGET_TAG
    assert dockerfile_arg('COMFYUI_SHA') == int8.TARGET_SHA
    text = SCRIPT.read_text()
    assert ': "${COMFYUI_TAG:=%s}"' % int8.TARGET_TAG in text
    assert ': "${COMFYUI_SHA:=%s}"' % int8.TARGET_SHA in text
    assert int8.UPSTREAM in text


def test_script_is_valid_bash_and_strict():
    assert SCRIPT.read_text().startswith('#!/usr/bin/env bash')
    assert 'set -Eeuo pipefail' in SCRIPT.read_text()
    bash = shutil.which('bash')
    if bash:
        subprocess.run([bash, '-n', str(SCRIPT)], check=True)


def test_script_has_no_crlf():
    assert b'\r' not in SCRIPT.read_bytes()


def test_script_protects_the_cuda_stack_and_verifies_it():
    text = SCRIPT.read_text()
    for name in int8.PROTECTED:
        if name != 'transformers':
            assert name in text
    assert 'torch|torchvision|torchaudio|numpy|transformers' in text
    assert '-c "$constraints"' in text            # pip constraint file is applied
    assert 'torch changed' in text                # and the result is verified
    assert 'int8_tensorwise' in text              # boot hook's no-op condition is asserted
    assert '0, 2, 16' in text                     # same comfy-kitchen floor as selfism_int8


def test_script_upgrades_torch_to_cu130_and_asserts_metadata():
    text = SCRIPT.read_text()
    assert 'https://download.pytorch.org/whl/cu130' in text
    assert 'torch==2.10.0+cu130' in text
    assert 'torchvision==0.25.0+cu130' in text
    assert 'torchaudio==2.10.0+cu130' in text
    assert '--break-system-packages' in text
    assert '*cu130*' in text                      # version string check after upgrade + after reqs
    assert 'torch.version.cuda' in text
    assert 'expected torch.version.cuda starting with 13' in text
    # constraints are rebuilt from the freeze AFTER the cu130 upgrade
    upgrade_at = text.index('whl/cu130')
    constraints_at = text.index('hold every package that ties ComfyUI to the (now cu130)')
    assert upgrade_at < constraints_at


def test_script_keeps_git_checkout_with_upstream_remote():
    text = SCRIPT.read_text()
    assert 'git init' in text and 'remote add origin' in text
    assert 'describe --tags --exact-match' in text
    # custom_nodes/ and user/ of the old baked tree survive the replacement
    assert '! -name custom_nodes ! -name user' in text
    assert '.runpod-bundle-version' in text


def test_dockerfile_bakes_and_keeps_the_template_contract():
    assert 'bake_comfyui.sh' in DOCKERFILE
    assert len(re.findall(r'^RUN ', DOCKERFILE, re.M)) == 2     # r2deps stage + the bake
    for line in ('COPY --link launcher/ /opt/10sorlabs/launcher/',
                 'COPY --link --chmod=755 docker/entrypoint.sh /start.sh'):
        assert line in DOCKERFILE
    assert 'ENTRYPOINT' not in DOCKERFILE and 'EXPOSE' not in DOCKERFILE and 'USER ' not in DOCKERFILE
    assert 'SELFISM_AUTO_COMFY_UPDATE=1' in DOCKERFILE
    # the bake must come before the COPYs of our own files so those stay cheap layers
    assert DOCKERFILE.index('bake_comfyui.sh') < DOCKERFILE.index('COPY --link launcher/')


def test_workflow_frees_enough_disk_for_unpacking_the_base():
    # The hosted runner has ~86 GiB free before and ~114 GiB after this cleanup; unpacking
    # the base rootfs (~11 GiB) plus the bake needs well under that.
    assert 'docker system prune --all --force' in WORKFLOW
    assert 'rm -rf /usr/share/dotnet' in WORKFLOW
    assert re.search(r'available < (\d+) \* 1024 \* 1024 \* 1024', WORKFLOW)
