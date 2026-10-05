#!/usr/bin/env python3
"""Pin ComfyUI latent preview to none before ComfyUI starts.

Called from docker/entrypoint.sh. Idempotent. Never raises out of main().

Why three places:
  - comfyui_args.txt -> `--preview-method none` is what the base start script passes to main.py.
    On ComfyUI 0.38.x Manager's own preview toggle is disabled (per-queue preview), so the CLI is
    what "Comfy.Execution.PreviewMethod = default" falls back to.
  - user/default/comfy.settings.json -> frontend Live preview method. If a pod previously saved
    "auto"/"taesd", that overrides the CLI per prompt and can still load a broken taeh3/taehv.
  - user/__manager/config.ini -> Manager's preview_method. On older ComfyUI (or when CLI is the
    default NoPreviews), Manager would otherwise rewrite the preview from this file; keep it none
    so it cannot undo the CLI pin.
"""
from __future__ import annotations

import configparser
import json
import os
import sys
from pathlib import Path

PREVIEW_KEY = 'Comfy.Execution.PreviewMethod'


def paths():
    workspace = Path(os.environ.get('SELFISM_WORKSPACE', '/workspace/runpod-slim'))
    comfy = Path(os.environ.get('COMFYUI_DIR', str(workspace / 'ComfyUI')))
    return {
        'workspace': workspace,
        'comfy': comfy,
        'args': workspace / 'comfyui_args.txt',
        'settings': comfy / 'user' / 'default' / 'comfy.settings.json',
        'manager': comfy / 'user' / '__manager' / 'config.ini',
        'legacy_manager': comfy / 'user' / 'default' / 'ComfyUI-Manager' / 'config.ini',
    }


def pin_args(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding='utf-8').splitlines() if path.is_file() else []
    out, seen = [], False
    for line in lines:
        if line.lstrip().startswith('--preview-method'):
            if not seen:
                out.append('--preview-method none')
                seen = True
            continue
        out.append(line)
    if not seen:
        if not out or out[0].startswith('#'):
            # keep any header comment the base start may have written
            if not any(l.startswith('#') for l in out[:1]):
                out.insert(0, '# Selfism: latent preview pinned to none (see docker/comfyui_defaults).')
            out.append('--preview-method none')
        else:
            out.append('--preview-method none')
        seen = True
    text = '\n'.join(out).rstrip() + '\n'
    if not path.is_file() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')
        return 'wrote ' + str(path)
    return 'ok ' + str(path)


def pin_settings(path: Path, comfy: Path) -> str:
    if not (comfy / 'user').is_dir():
        # ComfyUI not copied yet (first boot); bake seeds user/, base start will copy it.
        return 'skip (no ComfyUI user dir yet)'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
    if data.get(PREVIEW_KEY) == 'none':
        return 'ok ' + str(path)
    data[PREVIEW_KEY] = 'none'
    path.write_text(json.dumps(data, indent=4) + '\n', encoding='utf-8')
    return 'wrote ' + str(path)


def pin_manager(path: Path, comfy: Path) -> str:
    if not (comfy / 'user').is_dir():
        return 'skip (no ComfyUI user dir yet)'
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser(strict=False)
    if path.is_file():
        try:
            cfg.read(path)
        except configparser.Error:
            pass
    if not cfg.has_section('default'):
        cfg.add_section('default')
    if cfg.get('default', 'preview_method', fallback='') == 'none':
        return 'ok ' + str(path)
    cfg.set('default', 'preview_method', 'none')
    with path.open('w', encoding='utf-8') as fh:
        cfg.write(fh)
    return 'wrote ' + str(path)


def main() -> int:
    p = paths()
    msgs = [pin_args(p['args']), pin_settings(p['settings'], p['comfy']), pin_manager(p['manager'], p['comfy'])]
    if p['legacy_manager'].parent.is_dir() or p['legacy_manager'].is_file():
        msgs.append(pin_manager(p['legacy_manager'], p['comfy']))
    print('[selfism-preview] ' + '; '.join(msgs), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:  # never block boot
        print('[selfism-preview] ignored error: ' + str(exc), flush=True)
        raise SystemExit(0)
