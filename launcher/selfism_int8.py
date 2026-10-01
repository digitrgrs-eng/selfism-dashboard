"""Fixed, auditable ComfyUI INT8 repair; never accepts arbitrary shell commands.

Selfora INT8 checkpoints need the ``int8_tensorwise`` quantisation format (ComfyUI
v0.27.0, commit 1a510f04, PR #14636) and ``comfy-kitchen`` >= 0.2.16.  Older ComfyUI
fails to load them with ``KeyError: 'int8_tensorwise'``.  This script moves a ComfyUI
checkout to one pinned upstream release tag and installs that release's requirements
under the protected-package constraints.  It is idempotent: when the checkout already
supports INT8 and comfy-kitchen is new enough it changes nothing.

Run it with the ComfyUI virtual environment's Python:
    <comfyui>/.venv-cu128/bin/python launcher/selfism_int8.py --comfy-dir <comfyui>
"""
import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

UPSTREAM = 'https://github.com/Comfy-Org/ComfyUI.git'
TARGET_TAG = 'v0.38.1'
TARGET_SHA = '20ca544ee0436721d8eb5f544665e490609f72c8'
MARKER = 'int8_tensorwise'
SOURCE_FILES = ('comfy/quant_ops.py', 'comfy/ops.py')
MIN_KITCHEN = (0, 2, 16)
PROTECTED = ('torch', 'torchvision', 'torchaudio', 'numpy', 'transformers')
BACKUP_NAME = '.selfism-int8-backup.json'
CONSTRAINTS_NAME = '.selfism-constraints.txt'


def log(message):
    print('[selfism-int8] ' + message, flush=True)


def run(command, **kwargs):
    log('Running: ' + ' '.join(map(str, command)))
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def git(comfy, *args, check=True, timeout=600):
    result = subprocess.run(['git', '-C', str(comfy), *args], capture_output=True,
                            text=True, timeout=timeout)
    if check and result.returncode:
        raise SystemExit('git ' + ' '.join(args) + ' failed: '
                         + (result.stderr or result.stdout).strip()[-500:])
    return result.stdout.strip()


def version_tuple(text):
    parts = []
    for piece in str(text).split('+')[0].split('.'):
        digits = ''
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ''


def fresh_version(name):
    """Version as a new interpreter sees it, after pip has changed the environment."""
    code = ('import importlib.metadata as m\n'
            'try: print(m.version(%r))\n'
            'except m.PackageNotFoundError: print("")' % name)
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=60).stdout.strip()


def source_supports_int8(comfy):
    for relative in SOURCE_FILES:
        path = comfy / relative
        try:
            if MARKER in path.read_text(encoding='utf-8', errors='replace'):
                return True
        except OSError:
            continue
    return False


def kitchen_ok(version):
    return bool(version) and version_tuple(version) >= MIN_KITCHEN


def ensure_constraints(comfy):
    """Use the controller's snapshot when present; otherwise write one from this venv."""
    path = comfy / CONSTRAINTS_NAME
    if not path.is_file() or not path.read_text().strip():
        lines = []
        for name in PROTECTED:
            version = installed(name)
            if version:
                lines.append(name + '==' + version)
            else:
                log('Not installed, so not pinned: ' + name)
        path.write_text('\n'.join(lines) + '\n')
        log('Wrote ' + str(path) + ' from installed versions: ' + ', '.join(lines))
    else:
        log('Using existing constraints ' + str(path))
    return path


def write_backup(comfy, head, branch, stashed, torch_version):
    path = comfy / BACKUP_NAME
    entry = {'head': head, 'branch': branch, 'stash': stashed, 'torch': torch_version,
             'target_tag': TARGET_TAG, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = {}
    data.setdefault('original', entry)
    data['latest'] = entry
    path.write_text(json.dumps(data, indent=2) + '\n')
    log('Previous state recorded in ' + str(path) + ' (HEAD ' + head[:8] + ').')


def install_requirements(comfy, constraints):
    env = dict(os.environ, PIP_CONSTRAINT=str(constraints))
    run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
         '--timeout', '30', '--retries', '3', '-r', comfy / 'requirements.txt'],
        env=env, timeout=2100)


def _terminated(signum, frame):
    # SIGTERM (boot-time timeout, pod stop) becomes an exception so the rollback runs.
    raise KeyboardInterrupt('terminated by signal %d' % signum)


def main():
    signal.signal(signal.SIGTERM, _terminated)
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-dir', default=os.environ.get('COMFYUI_DIR',
                        '/workspace/runpod-slim/ComfyUI'))
    parser.add_argument('--stash', action='store_true',
                        help='stash local changes instead of refusing to touch a dirty checkout')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    comfy = Path(args.comfy_dir).resolve()
    if not (comfy / 'requirements.txt').is_file():
        raise SystemExit('Not a ComfyUI directory: ' + str(comfy))

    has_source = source_supports_int8(comfy)
    kitchen = installed('comfy-kitchen')
    log('ComfyUI source has %s: %s; comfy-kitchen: %s (need >= %s)' % (
        MARKER, has_source, kitchen or 'missing', '.'.join(map(str, MIN_KITCHEN))))
    if has_source and kitchen_ok(kitchen):
        log('INT8 support already present. Nothing to do.')
        return
    if args.check_only:
        raise SystemExit('INT8 support is missing.')
    if not (comfy / '.git').exists():
        raise SystemExit('ComfyUI is not a Git checkout; cannot move it to ' + TARGET_TAG + '.')

    torch_before = installed('torch')
    constraints = ensure_constraints(comfy)

    if not has_source:
        head = git(comfy, 'rev-parse', 'HEAD')
        branch = git(comfy, 'symbolic-ref', '--short', '-q', 'HEAD', check=False)
        dirty = git(comfy, 'status', '--porcelain', '--untracked-files=no')
        stashed = ''
        if dirty:
            if not args.stash:
                raise SystemExit('ComfyUI has local changes; refusing to switch version:\n' + dirty[:500])
            git(comfy, 'stash', 'push', '-m', 'selfism-int8-' + head[:8])
            stashed = git(comfy, 'rev-parse', '-q', '--verify', 'refs/stash')
            log('Local changes stashed (git stash list shows selfism-int8-' + head[:8] + ').')
        write_backup(comfy, head, branch, stashed, torch_before)
        log('Fetching tags from ' + UPSTREAM)
        git(comfy, 'fetch', '--force', '--tags', UPSTREAM, timeout=900)
        tag_sha = git(comfy, 'rev-parse', '--verify', 'refs/tags/' + TARGET_TAG + '^{commit}')
        if tag_sha != TARGET_SHA:
            raise SystemExit('Tag %s resolves to %s, expected %s; refusing.' % (TARGET_TAG, tag_sha, TARGET_SHA))
        git(comfy, 'checkout', '--detach', 'refs/tags/' + TARGET_TAG)
        log('Checked out %s (%s).' % (TARGET_TAG, tag_sha[:8]))
        if not source_supports_int8(comfy):
            raise SystemExit(MARKER + ' still missing after checkout; restore with: git -C %s checkout %s' % (comfy, head))

    try:
        install_requirements(comfy, constraints)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        log('Installing requirements failed: %s' % exc)
        if not has_source:
            log('Rolling ComfyUI back to ' + (branch or head)[:40])
            git(comfy, 'checkout', branch or head, check=False)
        raise SystemExit('Requirements install failed; ComfyUI source was ' +
                         ('rolled back.' if not has_source else 'left unchanged.'))

    kitchen_after = fresh_version('comfy-kitchen')
    torch_after = fresh_version('torch')
    if torch_before and torch_after != torch_before:
        raise SystemExit('PyTorch changed from %s to %s; repair it before using ComfyUI.' % (torch_before, torch_after))
    if not kitchen_ok(kitchen_after):
        raise SystemExit('comfy-kitchen is %s after install; need >= %s.' % (kitchen_after or 'missing', '.'.join(map(str, MIN_KITCHEN))))
    if not source_supports_int8(comfy):
        raise SystemExit(MARKER + ' missing from ComfyUI source.')
    log('Verified: %s present, comfy-kitchen %s, torch %s unchanged.' % (MARKER, kitchen_after, torch_after))
    log('ComfyUI INT8 support ready. Restart ComfyUI to load it.')


if __name__ == '__main__':
    main()
