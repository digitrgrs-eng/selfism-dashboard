"""Boot-time ComfyUI update, run by docker/entrypoint.sh before ComfyUI starts.

The RunPod base image launches ComfyUI from its own script (runpod-base-start.sh),
which this image cannot edit.  Instead ``entrypoint.sh`` calls this module *before*
that script, so the upgrade happens first and ComfyUI starts exactly once, already on
the pinned release.  It runs ``selfism_int8.py`` (an idempotent no-op when ComfyUI
already supports ``int8_tensorwise`` and comfy-kitchen is new enough).

Guarantees: never raises and always exits 0, so a failure can never block the boot;
bounded by a timeout; everything is appended to /workspace/selfism-boot.log.

Environment:
  SELFISM_AUTO_COMFY_UPDATE=0          disable entirely (default 1; "force" ignores the retry limit)
  SELFISM_AUTO_COMFY_UPDATE_TIMEOUT=N  seconds for the whole update (default 1800)
  COMFYUI_DIR                          default /workspace/runpod-slim/ComfyUI
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_COMFY = '/workspace/runpod-slim/ComfyUI'
DEFAULT_BAKED = '/opt/comfyui-baked'
DEFAULT_LOG = '/workspace/selfism-boot.log'
STATE_NAME = '.selfism-boot-state.json'
MAX_ATTEMPTS = 2          # consecutive failed attempts per target tag before giving up
LOG_LIMIT = 1_000_000     # bytes; older log is rotated to .1 beyond this
SEED_TIMEOUT = 600
MIN_TIMEOUT = 30
CURRENT = {'process': None}


def settings(env=None):
    env = os.environ if env is None else env
    mode = env.get('SELFISM_AUTO_COMFY_UPDATE', '1').strip().lower()
    try:
        timeout = int(env.get('SELFISM_AUTO_COMFY_UPDATE_TIMEOUT', '1800'))
    except ValueError:
        timeout = 1800
    return {
        'mode': 'off' if mode in ('0', 'false', 'no', 'off') else ('force' if mode == 'force' else 'on'),
        'timeout': max(MIN_TIMEOUT, timeout),
        'comfy': Path(env.get('COMFYUI_DIR', DEFAULT_COMFY)),
        'baked': Path(env.get('SELFISM_BAKED_COMFYUI', DEFAULT_BAKED)),
        'log': Path(env.get('SELFISM_BOOT_LOG', DEFAULT_LOG)),
        'script': Path(env.get('SELFISM_INT8_SCRIPT', str(HERE / 'selfism_int8.py'))),
    }


class Log:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file() and self.path.stat().st_size > LOG_LIMIT:
                self.path.replace(self.path.with_name(self.path.name + '.1'))
            self.handle = self.path.open('a', encoding='utf-8', errors='replace')
        except OSError:
            self.handle = None

    def write(self, text):
        sys.stdout.write(text); sys.stdout.flush()
        if self.handle:
            try:
                self.handle.write(text); self.handle.flush()
            except OSError:
                pass

    def line(self, message):
        self.write('[selfism-boot %s] %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), message))


def target_tag():
    try:
        sys.path.insert(0, str(HERE.parent))
        from launcher import selfism_int8
        return selfism_int8.TARGET_TAG
    except Exception:
        return 'unknown'


def read_state(comfy):
    try:
        return json.loads((comfy / STATE_NAME).read_text())
    except (OSError, ValueError):
        return {}


def write_state(comfy, state):
    try:
        (comfy / STATE_NAME).write_text(json.dumps(state, indent=2) + '\n')
    except OSError:
        pass


def run_logged(log, command, timeout, cwd=None):
    """Run a command streaming its output to the log. Returns the exit code (124 = timed out)."""
    process = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors='replace', cwd=cwd,
                               start_new_session=True)
    CURRENT['process'] = process
    def pump():
        for chunk in process.stdout:
            log.write(chunk)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.line('Timed out after %ds; asking the update to stop and roll back.' % timeout)
        try:
            process.send_signal(signal.SIGTERM)       # the script rolls back on SIGTERM
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except OSError: pass
            process.wait()
        code = 124
    reader.join(timeout=5)
    CURRENT['process'] = None
    return code


def prepare_environment(cfg, log):
    """Create what the base script would create on a first start, so we can upgrade first.

    Returns the venv python path, or None when we must leave first-time setup to the
    base script (it then starts ComfyUI as shipped; the dashboard can still upgrade).
    """
    comfy = cfg['comfy']
    venv = comfy / '.venv-cu128'
    python = venv / 'bin/python'
    if (venv / 'bin/activate').is_file() and python.is_file() and (comfy / 'requirements.txt').is_file():
        return python
    if (comfy / '.venv').is_dir() and not venv.is_dir():
        log.line('Old CUDA 12.4 venv present; leaving its migration to the base image. Skipping.')
        return None
    if not (comfy / 'requirements.txt').is_file():
        if comfy.exists() or not (cfg['baked'] / 'requirements.txt').is_file():
            log.line('No usable ComfyUI at %s and no baked copy; skipping.' % comfy)
            return None
        log.line('First start: copying baked ComfyUI to %s (same as the base image does).' % comfy)
        comfy.parent.mkdir(parents=True, exist_ok=True)
        if run_logged(log, ['cp', '-r', cfg['baked'], comfy], SEED_TIMEOUT):
            log.line('Copy failed; removing the partial copy so the base image can redo it.')
            subprocess.run(['rm', '-rf', '--', str(comfy)])
            return None
    if not venv.is_dir():
        log.line('First start: creating the ComfyUI virtual environment.')
        failed = run_logged(log, ['python3.12', '-m', 'venv', '--system-site-packages', venv], SEED_TIMEOUT)
        if not failed:
            failed = run_logged(log, [venv / 'bin/python', '-m', 'ensurepip'], SEED_TIMEOUT)
        if failed:
            log.line('Virtual environment creation failed; removing it so the base image can redo it.')
            subprocess.run(['rm', '-rf', '--', str(venv)])
            return None
    return python if (venv / 'bin/activate').is_file() and python.is_file() else None


def forward_sigterm(signum, frame):
    """Pod stop: pass SIGTERM to the running update so it rolls back instead of dying half-way."""
    process = CURRENT.get('process')
    if process is not None and process.poll() is None:
        try: process.send_signal(signal.SIGTERM)
        except OSError: pass


def main():
    try: signal.signal(signal.SIGTERM, forward_sigterm)
    except ValueError: pass   # not the main thread (tests)
    cfg = settings()
    if cfg['mode'] == 'off':
        print('[selfism-boot] SELFISM_AUTO_COMFY_UPDATE=0; skipping ComfyUI update.', flush=True)
        return 0
    log = Log(cfg['log'])
    started = time.monotonic()
    try:
        tag = target_tag()
        log.line('Checking ComfyUI (target %s, timeout %ds, log %s).' % (tag, cfg['timeout'], cfg['log']))
        if not cfg['script'].is_file():
            log.line('Missing %s; skipping.' % cfg['script'])
            return 0
        state = read_state(cfg['comfy'])
        if (cfg['mode'] != 'force' and state.get('tag') == tag
                and state.get('failures', 0) >= MAX_ATTEMPTS):
            log.line('The update to %s failed %d boots in a row (%s). Not retrying on boot; '
                     'fix the cause and set SELFISM_AUTO_COMFY_UPDATE=force, or use the dashboard.'
                     % (tag, state['failures'], state.get('last_error', 'see log')))
            return 0
        python = prepare_environment(cfg, log)
        if python is None:
            return 0
        remaining = max(MIN_TIMEOUT, cfg['timeout'] - int(time.monotonic() - started))
        code = run_logged(log, [python, '-u', cfg['script'], '--comfy-dir', cfg['comfy'], '--stash'],
                          remaining)
        if code == 0:
            if state.get('failures'):
                log.line('Update succeeded after earlier failures; retry counter cleared.')
            write_state(cfg['comfy'], {'tag': tag, 'failures': 0, 'last_ok': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
            log.line('ComfyUI check finished OK in %.0fs.' % (time.monotonic() - started))
        else:
            failures = state.get('failures', 0) + 1 if state.get('tag') == tag else 1
            write_state(cfg['comfy'], {'tag': tag, 'failures': failures, 'last_error': 'exit code %s' % code,
                                       'last_fail': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
            log.line('ComfyUI update did not complete (exit code %s, failure %d of %d allowed). '
                     'Continuing the boot with ComfyUI as it is.' % (code, failures, MAX_ATTEMPTS))
    except BaseException as exc:  # never fail the boot
        try: log.line('Unexpected error, continuing boot: %s: %s' % (type(exc).__name__, exc))
        except Exception: pass
    return 0


if __name__ == '__main__':
    try:
        main()
    finally:
        sys.exit(0)
