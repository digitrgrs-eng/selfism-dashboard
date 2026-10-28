"""Compare two image inventories (scripts/image_inventory.py): OLD.json NEW.json [REPORT.md]

Exits 1 when a python or apt package, custom node, node pack, ComfyUI checkout, image
config value or file path is missing or has a different version in NEW, other than the
expected additions (llama-cpp-python, diskcache). File content differences are reported
by area; files that are rebuilt on every build (bytecode, pip RECORD files, git objects,
logs) are expected to differ and are listed separately.
"""
import collections
import json
import re
import sys

EXPECTED_NEW_PACKAGES = {'llama_cpp_python', 'diskcache'}
VOLATILE = re.compile(
    r'(\.pyc$|/__pycache__(/|$)|\.dist-info/(RECORD|direct_url\.json|INSTALLER|REQUESTED)$'
    r'|^/opt/comfyui-baked/\.git/|^/var/log/|^/var/cache/|^/var/lib/apt/|^/var/lib/dpkg/'
    r'|^/etc/ld\.so\.cache$|^/tmp/|^/root/\.cache/|^/root/\.config/|\.selfism-source\.json$|^/etc/machine-id$)')
EXPECTED_NEW_PATH = re.compile(r'^/usr/local/lib/python3\.12/dist-packages/(llama_cpp|diskcache|bin)'
                               r'|^/usr/local/lib/python3\.12/dist-packages/(llama_cpp_python|diskcache)-'
                               r'|^/usr/local/bin/(llama|pip)')


def norm(name):
    return re.sub(r'[-_.]+', '_', name).lower()


def area(path):
    parts = path.split('/')
    if path.startswith('/usr/local/lib/python3.12/dist-packages/'):
        return '/'.join(parts[:7])
    return '/'.join(parts[:3])


def main():
    old, new = (json.load(open(p)) for p in sys.argv[1:3])
    lines, problems = [], []
    say = lines.append

    def packages(inv):
        return {norm(n): (v, loc) for n, v, loc in inv['pip_packages']}
    po, pn = packages(old), packages(new)
    missing = sorted(set(po) - set(pn))
    changed = sorted(n for n in set(po) & set(pn) if po[n] != pn[n])
    added = sorted(set(pn) - set(po))
    say('## Python packages: %d old, %d new' % (len(po), len(pn)))
    say('- missing in new: %s' % (missing or 'none'))
    say('- version/location changed: %s' % ([(n, po[n], pn[n]) for n in changed] or 'none'))
    say('- added in new: %s' % (added or 'none'))
    problems += ['pip missing ' + n for n in missing] + ['pip changed ' + n for n in changed]
    problems += ['pip unexpected new ' + n for n in added if n not in EXPECTED_NEW_PACKAGES]

    def freeze(inv):
        # A local wheel's freeze line names the random mktemp dir it was installed from.
        return {re.sub(r'file:///tmp/tmp\.[^/]+/', 'file:///tmp/<tmp>/', l) for l in inv['pip_freeze']
                if not l.lower().startswith(('llama', 'diskcache'))}
    fo, fn = freeze(old), freeze(new)
    say('- pip freeze --all lines: old %d, new %d, only-old %s, only-new %s'
        % (len(old['pip_freeze']), len(new['pip_freeze']), sorted(fo - fn) or 'none', sorted(fn - fo) or 'none'))
    problems += ['freeze line ' + l for l in sorted(fo ^ fn)]

    do, dn = set(old['dpkg']), set(new['dpkg'])
    say('## apt packages: %d old, %d new; only-old %s; only-new %s'
        % (len(do), len(dn), sorted(do - dn) or 'none', sorted(dn - do) or 'none'))
    problems += ['dpkg ' + l for l in sorted(do ^ dn)]

    for key in ('python', 'which_python', 'comfyui_head', 'comfyui_tag', 'comfyui_custom_nodes', 'node_cache', 'ld_so_conf'):
        same = old.get(key) == new.get(key)
        say('## %s: %s' % (key, 'identical' if same else 'DIFFERENT old=%r new=%r' % (old.get(key), new.get(key))))
        if not same:
            problems.append(key)
    if 'config' in old and 'config' in new:
        for key in sorted(set(old['config']) | set(new['config'])):
            if key == 'Labels':
                continue
            if old['config'].get(key) != new['config'].get(key):
                say('## image config %s DIFFERENT: %r -> %r' % (key, old['config'].get(key), new['config'].get(key)))
                problems.append('config ' + key)
        say('## image config (ENV, ENTRYPOINT, WORKDIR, ports, healthcheck, user): compared')

    fo, fn = old['files'], new['files']
    gone = sorted(p for p in fo if p not in fn)
    extra = sorted(p for p in fn if p not in fo)
    meta = collections.Counter(); content = collections.Counter(); volatile = collections.Counter()
    meta_examples, content_examples = [], []
    for path in sorted(set(fo) & set(fn)):
        a, b = fo[path], fn[path]
        if VOLATILE.search(path):
            if a != b:
                volatile[area(path)] += 1
            continue
        if {k: a.get(k) for k in ('type', 'mode', 'uid', 'gid', 'target')} != {k: b.get(k) for k in ('type', 'mode', 'uid', 'gid', 'target')}:
            meta[area(path)] += 1
            meta_examples.append((path, a, b))
        elif a.get('sha256') != b.get('sha256'):
            content[area(path)] += 1
            content_examples.append((path, a.get('size'), b.get('size')))
    gone_real = [p for p in gone if not VOLATILE.search(p)]
    extra_real = [p for p in extra if not VOLATILE.search(p) and not EXPECTED_NEW_PATH.search(p)]
    say('## files: old %d, new %d' % (len(fo), len(fn)))
    say('- missing in new (excluding rebuilt bytecode/logs/git objects): %d %s' % (len(gone_real), gone_real[:60]))
    say('- unexpected extra in new: %d %s' % (len(extra_real), extra_real[:60]))
    say('- type/mode/owner/link differences: %d %s' % (sum(meta.values()), meta_examples[:30]))
    say('- content differences (same path, different sha256): %d by area %s' % (sum(content.values()), content.most_common(40)))
    say('  examples: %s' % content_examples[:40])
    say('- rebuilt-every-build files that differ (bytecode, RECORD, git objects, logs): %d by area %s'
        % (sum(volatile.values()), volatile.most_common(15)))
    problems += ['file missing ' + p for p in gone_real] + ['file meta ' + p for p, _, _ in meta_examples]
    say('')
    say('RESULT: %s' % ('OK - nothing lost or changed in version' if not problems else 'DIFFERENCES: %d (first: %s)' % (len(problems), problems[:20])))
    report = '\n'.join(lines) + '\n'
    print(report)
    if len(sys.argv) > 3:
        open(sys.argv[3], 'w').write(report)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
