"""Extra Krea 2 LoRAs + Realism by Stable Yogi checkpoint: single-model Download rows backed by private R2."""
from __future__ import annotations
import json
import re
from pathlib import Path

from launcher.private_r2 import select_private
from tests.test_minimax import start_install

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'catalog/selfism.json').read_text(encoding='utf-8'))
R2_KEYS = {
    'lora-krea2-realism-gokay': 'loras/krea2_realism_lora.safetensors',
    'lora-krea2-realism-v2': 'loras/Krea2-realism-V2.safetensors',
    'lora-candid-snap': 'loras/candid_snap_krea2_v2.safetensors',
    'lora-summervibes-skin': 'loras/SummerVibesHM_krea2_epoch8.safetensors',
    'lora-mystic-xxx': 'loras/MysticXXX_KREA2_v3.safetensors',
    'lora-better-pussy': 'loras/krea2_better_pussy_poses_v4.2.1.safetensors',
    'krea2-stable-yogi-realism': 'diffusion_models/realismByStableYogi_v30INT8FP8Extended_3218460.safetensors',
}


def test_entries_have_pinned_metadata_and_r2_keys():
    files = CATALOG['files']
    for key, r2 in R2_KEYS.items():
        f = files[key]
        assert f['id'] == key and f['destination'] == 'models/' + r2
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0
        assert f['description'].startswith('Krea 2 ') and 'Trigger:' in f['description']
        # Fixed SHA256 lets private R2 serve the file without a Civitai metadata lookup.
        assert 'civitai_version' not in f
        assert f['auth'] == ('civitai' if f['url'].startswith('https://civitai.com/') else 'none')
    assert len({files[k]['destination'] for k in files}) == len(files)


def test_single_model_download_uses_catalog_entry(monkeypatch):
    for key in R2_KEYS:
        r, captured, _ = start_install(monkeypatch, {'profile': 'model', 'item': key})
        assert r.status_code == 200
        assert [f['id'] for f in captured['files']] == [key]


def test_private_r2_selected_by_sha_and_size():
    class Client:
        def head_object(self, Bucket, Key):
            spec = next(CATALOG['files'][k] for k, v in R2_KEYS.items() if v == Key)
            return {'ContentLength': spec['size_bytes'], 'Metadata': {'sha256': spec['sha256']}}
        def generate_presigned_url(self, *args, **kwargs):
            return 'https://example.r2.cloudflarestorage.com/signed'
    files = [dict(CATALOG['files'][k]) for k in R2_KEYS]
    selected, _ = select_private(files, Client(), 'selfism-models')
    assert sorted(selected) == list(range(len(files)))
