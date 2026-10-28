"""Inventory of a Selfism image, run inside it: python3.12 image_inventory.py OUT.json

Records what "nothing lost or changed" is checked against: pip packages (name + version),
apt packages, ComfyUI checkout, custom nodes and baked node packs, and every file
(type, mode, owner, size, symlink target, sha256). Used by compare-images.yml.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

SKIP = {'/proc', '/sys', '/dev', '/inv', '/out', '/etc/hosts', '/etc/hostname', '/etc/resolv.conf',
        '/.dockerenv', '/workspace'}


def run(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=600).stdout
    except Exception as exc:  # noqa: BLE001 - an inventory records failures instead of dying
        return 'ERROR %s' % exc


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def files():
    out = {}
    for top, dirs, names in os.walk('/', topdown=True):
        dirs[:] = [d for d in dirs if os.path.join(top, d) not in SKIP]
        for name in dirs + names:
            path = os.path.join(top, name)
            if path in SKIP:
                continue
            try:
                info = os.lstat(path)
            except OSError:
                continue
            entry = {'mode': oct(stat.S_IMODE(info.st_mode)), 'uid': info.st_uid, 'gid': info.st_gid}
            if stat.S_ISLNK(info.st_mode):
                entry.update(type='link', target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                entry.update(type='dir')
            elif stat.S_ISREG(info.st_mode):
                entry.update(type='file', size=info.st_size)
                try:
                    entry['sha256'] = sha256(path)
                except OSError as exc:
                    entry['sha256'] = 'ERROR %s' % exc
            else:
                entry.update(type='other')
            out[path] = entry
    return out


def node_dirs(root):
    result = {}
    root = Path(root)
    if not root.is_dir():
        return result
    for child in sorted(root.iterdir()):
        marker = child / '.selfism-source.json'
        result[child.name] = json.loads(marker.read_text()) if marker.is_file() else None
    return result


def main():
    packages = sorted({(d.metadata['Name'], d.version, str(Path(d._path).parent))  # noqa: SLF001
                       for d in metadata.distributions()})
    baked = '/opt/comfyui-baked'
    data = {
        'pip_packages': [list(p) for p in packages],
        'pip_freeze': run(sys.executable, '-m', 'pip', 'freeze', '--all', '--disable-pip-version-check').splitlines(),
        'dpkg': run('dpkg-query', '-W', '-f=${Package} ${Version} ${Architecture}\n').splitlines(),
        'python': run('python3.12', '--version').strip(),
        'which_python': run('bash', '-lc', 'command -v python3.12; readlink -f "$(command -v python3.12)"').split(),
        'comfyui_head': run('git', '-c', 'safe.directory=*', '-C', baked, 'rev-parse', 'HEAD').strip(),
        'comfyui_tag': run('git', '-c', 'safe.directory=*', '-C', baked, 'describe', '--tags', '--exact-match').strip(),
        'comfyui_custom_nodes': sorted(os.listdir(baked + '/custom_nodes')),
        'node_cache': node_dirs('/opt/10sorlabs/node_cache'),
        'ld_so_conf': run('bash', '-c', 'cat /etc/ld.so.conf.d/*.conf'),
        'files': files(),
    }
    Path(sys.argv[1]).write_text(json.dumps(data, sort_keys=True))
    print('inventory:', len(data['files']), 'paths,', len(packages), 'python distributions')


if __name__ == '__main__':
    main()
