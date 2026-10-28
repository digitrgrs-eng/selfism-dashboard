"""Static checks for the base image (docker/base.Dockerfile + build-base.yml)."""
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'docker/base.Dockerfile').read_text()
LLAMA = ROOT / 'docker/install_llama_cpp.sh'
BASE_WF = (ROOT / '.github/workflows/build-base.yml').read_text()
sys.path.insert(0, str(ROOT / 'scripts'))
base_image = importlib.import_module('base_image')
runtime = importlib.import_module('launcher.selfism_runtime')
int8 = importlib.import_module('launcher.selfism_int8')


def test_base_nodes_file_matches_the_catalog():
    assert (ROOT / 'docker/base-nodes.json').read_text() == base_image.nodes_document()
    names = [n['name'] for n in json.loads((ROOT / 'docker/base-nodes.json').read_text())['nodes']]
    catalog = json.loads((ROOT / 'catalog/selfism.json').read_text())
    assert names == catalog['krast_org_depth_nodes'] + ['hearmemanai-upscale', 'openrouter_node', 'ComfyUI-AusBoss']


def test_base_tag_is_a_hash_of_every_input():
    tag = base_image.base_tag()
    assert re.fullmatch(r'base-[0-9a-f]{16}', tag)
    for name in base_image.INPUTS:
        assert (ROOT / name).is_file(), name
    for path in ('docker/bake_comfyui.sh', 'docker/install_llama_cpp.sh', 'docker/base-constraints.txt', 'docker/base-nodes.json'):
        assert path in base_image.INPUTS
        assert 'source=' + path in BASE
        assert '- ' + path in BASE_WF          # push trigger paths
    assert '- docker/base.Dockerfile' in BASE_WF


def test_base_bakes_from_the_same_upstream_digest_and_comfyui_pin():
    assert 'FROM 10sorllabs/comfyui-workflow-launcher@sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846 AS fat' in BASE
    assert 'ARG COMFYUI_TAG=%s' % int8.TARGET_TAG in BASE
    assert 'ARG COMFYUI_SHA=%s' % int8.TARGET_SHA in BASE
    assert 'SELFISM_BASE_CONSTRAINTS=/tmp/base-constraints.txt bash /tmp/bake_comfyui.sh' in BASE
    assert 'libportaudio2' in BASE and 'libgl1-mesa-dri' in BASE


def test_base_is_flattened_into_one_copy_of_the_filesystem():
    final = BASE.split('FROM scratch', 1)[1]
    assert 'COPY --from=fat --exclude=usr/local/lib/python3.12 / /' in final
    for package in ('torch', 'triton', 'nvidia'):
        assert '--exclude=%s' % package in final
        assert 'COPY --from=fat /usr/local/lib/python3.12/dist-packages/%s/ /usr/local/lib/python3.12/dist-packages/%s/' % (package, package) in final
    assert 'COPY --from=r2deps /r2deps/ /opt/r2deps/' in final
    assert BASE.startswith('# syntax=docker/dockerfile:1.7-labs')    # COPY --exclude


def test_base_restores_the_upstream_image_config():
    final = BASE.split('FROM scratch', 1)[1]
    assert 'ENTRYPOINT ["/start.sh"]' in final
    assert 'WORKDIR /workspace/runpod-slim' in final
    assert 'USER root' in final
    assert 'EXPOSE 22/tcp 3000/tcp 8080/tcp 8188/tcp 8888/tcp' in final
    assert 'curl --fail --silent http://127.0.0.1:3000/api/health || exit 1' in final
    for env in ('PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                'LD_LIBRARY_PATH=/usr/local/cuda/lib64', 'NVIDIA_VISIBLE_DEVICES=all',
                'NVIDIA_DRIVER_CAPABILITIES=all', 'COMFYUI_DIR=/workspace/runpod-slim/ComfyUI',
                'LCT_API_BASE=https://rapidcache.10sorlabs.com', 'IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg'):
        assert env in final


def test_constraints_pin_the_old_image_versions():
    text = (ROOT / 'docker/base-constraints.txt').read_text()
    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
    assert len(lines) > 300
    assert all(re.fullmatch(r'[A-Za-z0-9._-]+==\S+', l) for l in lines)
    for pin in ('torch==2.10.0+cu130', 'torchvision==0.25.0+cu130', 'torchaudio==2.10.0+cu130',
                'onnxruntime_gpu==1.30.0', 'mediapipe==1.1.1', 'triton==3.6.0', 'comfy_kitchen=='):
        assert any(l.startswith(pin) for l in lines), pin
    assert not any(l.lower().startswith(('sam_2==', 'sageattention==', 'llama')) for l in lines)
    bake = (ROOT / 'docker/bake_comfyui.sh').read_text()
    assert 'SELFISM_BASE_CONSTRAINTS' in bake and '"${base_pins[@]}"' in bake


def test_llama_wheel_is_the_one_the_repair_downloads():
    text = LLAMA.read_text()
    assert runtime.WHEEL in text
    assert 'cae7e9082e12ddf377ccbd2d19a9d3c14b86fafb9b9d702d0bd846563c0d1ca9' in text
    assert 'LLAMA_CPP_VERSION:=0.4.1+cu128' in text
    assert "version != '0.4.1+cu128'" in (ROOT / 'launcher/selfism_runtime.py').read_text()
    assert '--no-deps "$wheel"' in text
    assert 'diskcache' in runtime.RUNTIME_DEPS and 'diskcache==$DISKCACHE_VERSION' in text
    assert 'Qwen35ChatHandler' in text and 'libcublas.so.12' in text
    assert 'torch changed during the llama-cpp install' in text
    assert 'the llama-cpp install changed other packages' in text
    subprocess.run(['bash', '-n', str(LLAMA)], check=True)


def test_base_workflow_skips_existing_tags_and_reports_the_digest():
    assert 'docker buildx imagetools inspect' in BASE_WF
    assert 'file: docker/base.Dockerfile' in BASE_WF
    assert 'scripts/base_image.py tag' in BASE_WF
    assert 'steps.build.outputs.digest' in BASE_WF
    assert 'compression' not in BASE_WF          # gzip, as before


def test_compare_script_flags_lost_or_changed_packages(tmp_path):
    def inv(pkgs, files, freeze=None):
        return {'pip_packages': pkgs, 'pip_freeze': freeze or ['%s==%s' % (n, v) for n, v, _ in pkgs],
                'dpkg': ['a 1 amd64'], 'python': 'Python 3.12.3', 'which_python': ['/usr/bin/python3.12'],
                'comfyui_head': 'x', 'comfyui_tag': 'v0.38.1', 'comfyui_custom_nodes': [], 'node_cache': {},
                'ld_so_conf': '', 'files': files}
    site = '/usr/local/lib/python3.12/dist-packages'
    f = {'/a': {'type': 'file', 'mode': '0o644', 'uid': 0, 'gid': 0, 'size': 1, 'sha256': 'x'}}
    old = inv([['torch', '2.10.0+cu130', site]], f)
    good = inv([['torch', '2.10.0+cu130', site], ['llama_cpp_python', '0.4.1+cu128', site]], dict(f))
    bad = inv([['torch', '2.9.0', site]], {})
    paths = {}
    for name, data in (('old', old), ('good', good), ('bad', bad)):
        paths[name] = tmp_path / (name + '.json')
        paths[name].write_text(json.dumps(data))
    script = ROOT / 'scripts/compare_images.py'
    ok = subprocess.run([sys.executable, script, paths['old'], paths['good']], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    assert 'RESULT: OK' in ok.stdout
    ko = subprocess.run([sys.executable, script, paths['old'], paths['bad']], capture_output=True, text=True)
    assert ko.returncode == 1 and 'pip changed torch' in ko.stdout and 'file missing /a' in ko.stdout
