import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

boot = importlib.import_module('launcher.selfism_boot')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    comfy = tmp_path/'ComfyUI'
    (comfy/'.venv-cu128/bin').mkdir(parents=True)
    (comfy/'requirements.txt').write_text('')
    (comfy/'.venv-cu128/bin/activate').write_text('')
    py = comfy/'.venv-cu128/bin/python'
    py.symlink_to(sys.executable)
    monkeypatch.setenv('COMFYUI_DIR', str(comfy))
    monkeypatch.setenv('SELFISM_BOOT_LOG', str(tmp_path/'boot.log'))
    monkeypatch.setenv('SELFISM_BAKED_COMFYUI', str(tmp_path/'baked'))
    monkeypatch.delenv('SELFISM_AUTO_COMFY_UPDATE', raising=False)
    monkeypatch.delenv('SELFISM_AUTO_COMFY_UPDATE_TIMEOUT', raising=False)
    return tmp_path, comfy


def fake_script(tmp_path, monkeypatch, body):
    script = tmp_path/'fake_int8.py'
    script.write_text(body)
    monkeypatch.setenv('SELFISM_INT8_SCRIPT', str(script))
    return script


def log_text(tmp_path):
    return (tmp_path/'boot.log').read_text()


def test_settings_switch_values():
    for value in ('0', 'false', 'No', 'OFF'):
        assert boot.settings({'SELFISM_AUTO_COMFY_UPDATE': value})['mode'] == 'off'
    assert boot.settings({})['mode'] == 'on'
    assert boot.settings({'SELFISM_AUTO_COMFY_UPDATE': 'force'})['mode'] == 'force'
    assert boot.settings({'SELFISM_AUTO_COMFY_UPDATE_TIMEOUT': 'junk'})['timeout'] == 1800


def test_disabled_does_nothing(env, monkeypatch):
    tmp_path, comfy = env
    marker = tmp_path/'ran'
    fake_script(tmp_path, monkeypatch, 'open(%r,"w").write("x")' % str(marker))
    monkeypatch.setenv('SELFISM_AUTO_COMFY_UPDATE', '0')
    assert boot.main() == 0
    assert not marker.exists() and not (tmp_path/'boot.log').exists()


def test_runs_script_with_comfy_dir_and_stash_and_logs(env, monkeypatch):
    tmp_path, comfy = env
    fake_script(tmp_path, monkeypatch, 'import sys; print("ARGS", sys.argv[1:])')
    assert boot.main() == 0
    text = log_text(tmp_path)
    assert "ARGS ['--comfy-dir', %r, '--stash']" % str(comfy) in text
    assert 'finished OK' in text
    assert json.loads((comfy/'.selfism-boot-state.json').read_text())['failures'] == 0


def test_failure_never_fails_boot_and_is_counted_then_limited(env, monkeypatch):
    tmp_path, comfy = env
    marker = tmp_path/'count'
    fake_script(tmp_path, monkeypatch,
                'open(%r,"a").write("x"); raise SystemExit(3)' % str(marker))
    for _ in range(4):
        assert boot.main() == 0
    assert marker.read_text() == 'x'*boot.MAX_ATTEMPTS
    assert 'Not retrying on boot' in log_text(tmp_path)
    # 'force' ignores the limit.
    monkeypatch.setenv('SELFISM_AUTO_COMFY_UPDATE', 'force')
    assert boot.main() == 0
    assert marker.read_text() == 'x'*(boot.MAX_ATTEMPTS+1)


def test_success_clears_failure_counter(env, monkeypatch):
    tmp_path, comfy = env
    boot.write_state(comfy, {'tag': boot.target_tag(), 'failures': 1})
    fake_script(tmp_path, monkeypatch, 'pass')
    boot.main()
    assert boot.read_state(comfy)['failures'] == 0


def test_new_target_tag_resets_the_retry_limit(env, monkeypatch):
    tmp_path, comfy = env
    boot.write_state(comfy, {'tag': 'v0.0.1', 'failures': 9})
    marker = tmp_path/'ran'
    fake_script(tmp_path, monkeypatch, 'open(%r,"w").write("x")' % str(marker))
    boot.main()
    assert marker.exists()


def test_timeout_terminates_and_still_returns_zero(env, monkeypatch):
    tmp_path, comfy = env
    original = boot.settings
    monkeypatch.setattr(boot, 'settings', lambda: dict(original(), timeout=1))
    monkeypatch.setattr(boot, 'MIN_TIMEOUT', 1)
    fake_script(tmp_path, monkeypatch, 'import time; time.sleep(60)')
    assert boot.main() == 0
    assert 'Timed out' in log_text(tmp_path) and 'exit code 124' in log_text(tmp_path)


def test_missing_comfyui_is_skipped_quietly(env, monkeypatch):
    tmp_path, comfy = env
    import shutil; shutil.rmtree(comfy)
    marker = tmp_path/'ran'
    fake_script(tmp_path, monkeypatch, 'open(%r,"w").write("x")' % str(marker))
    assert boot.main() == 0
    assert not marker.exists() and 'skipping' in log_text(tmp_path)


def test_unexpected_exception_is_swallowed(env, monkeypatch):
    tmp_path, comfy = env
    monkeypatch.setattr(boot, 'prepare_environment', lambda *a: 1/0)
    fake_script(tmp_path, monkeypatch, 'pass')
    assert boot.main() == 0
    assert 'Unexpected error' in log_text(tmp_path)


def test_module_exits_zero_even_when_cli_run_fails(env):
    tmp_path, comfy = env
    result = subprocess.run([sys.executable, '-m', 'launcher.selfism_boot'], cwd=ROOT,
                            env=dict(os.environ, SELFISM_INT8_SCRIPT=str(tmp_path/'nope.py')),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert 'Missing' in result.stdout


def test_first_start_seeds_comfyui_and_venv_before_updating(tmp_path, monkeypatch):
    baked = tmp_path/'baked'; baked.mkdir()
    (baked/'requirements.txt').write_text('')
    comfy = tmp_path/'ws/ComfyUI'
    monkeypatch.setenv('COMFYUI_DIR', str(comfy))
    monkeypatch.setenv('SELFISM_BOOT_LOG', str(tmp_path/'boot.log'))
    monkeypatch.setenv('SELFISM_BAKED_COMFYUI', str(baked))
    monkeypatch.delenv('SELFISM_AUTO_COMFY_UPDATE', raising=False)
    calls = []
    def fake_run(log, command, timeout, cwd=None):
        calls.append([str(c) for c in command])
        if command[0] == 'cp':
            (comfy.parent).mkdir(parents=True, exist_ok=True)
            import shutil; shutil.copytree(command[2], command[3])
        elif command[1:3] == ['-m', 'venv']:
            (Path(command[-1])/'bin').mkdir(parents=True)
            (Path(command[-1])/'bin/python').symlink_to(sys.executable)
        elif command[1:3] == ['-m', 'ensurepip']:
            (comfy/'.venv-cu128/bin/activate').write_text('')
        return 0
    monkeypatch.setattr(boot, 'run_logged', fake_run)
    monkeypatch.setenv('SELFISM_INT8_SCRIPT', str(tmp_path/'x.py'))
    (tmp_path/'x.py').write_text('')
    boot.main()
    assert [c[0] for c in calls][:1] == ['cp']
    assert any(c[1:3] == ['-m', 'venv'] for c in calls)
    assert calls[-1][1:3] == ['-u', str(tmp_path/'x.py')]


def test_old_cu124_venv_is_left_to_the_base_image(tmp_path, monkeypatch):
    comfy = tmp_path/'ComfyUI'; (comfy/'.venv').mkdir(parents=True)
    (comfy/'requirements.txt').write_text('')
    monkeypatch.setenv('COMFYUI_DIR', str(comfy))
    monkeypatch.setenv('SELFISM_BOOT_LOG', str(tmp_path/'boot.log'))
    marker = tmp_path/'ran'
    fake_script(tmp_path, monkeypatch, 'open(%r,"w").write("x")' % str(marker))
    assert boot.main() == 0 and not marker.exists()
    assert 'migration' in (tmp_path/'boot.log').read_text()


def test_int8_script_rolls_back_on_sigterm():
    src = (ROOT/'launcher/selfism_int8.py').read_text()
    assert 'signal.SIGTERM' in src and 'KeyboardInterrupt' in src


def test_entrypoint_runs_hook_before_base_script_and_dockerfile_enables_it():
    entry = (ROOT/'docker/entrypoint.sh').read_text()
    assert entry.index('launcher.selfism_boot') < entry.index('runpod-base-start.sh &')
    assert 'BASE_PID=$!' in entry and entry.index('wait "$BOOT_PID"') < entry.index('runpod-base-start.sh &')
    assert subprocess.run(['bash', '-n', str(ROOT/'docker/entrypoint.sh')]).returncode == 0
    assert 'SELFISM_AUTO_COMFY_UPDATE=1' in (ROOT/'Dockerfile').read_text()
    assert 'apply_preview_defaults.py' in entry
    assert entry.index('apply_preview_defaults.py') < entry.index('runpod-base-start.sh')
    assert 'docker/comfyui_defaults/' in (ROOT/'Dockerfile').read_text()


def test_preview_defaults_pin_args_settings_and_manager_config(tmp_path, monkeypatch):
    import importlib.util
    script = ROOT/'docker/comfyui_defaults/apply_preview_defaults.py'
    spec = importlib.util.spec_from_file_location('apply_preview_defaults', script)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    work = tmp_path/'runpod-slim'; comfy = work/'ComfyUI'
    (comfy/'user'/'default').mkdir(parents=True)
    monkeypatch.setenv('SELFISM_WORKSPACE', str(work))
    monkeypatch.setenv('COMFYUI_DIR', str(comfy))
    # first apply creates everything
    assert mod.main() == 0
    args = (work/'comfyui_args.txt').read_text()
    assert '--preview-method none' in args and args.count('--preview-method') == 1
    settings = json.loads((comfy/'user'/'default'/'comfy.settings.json').read_text())
    assert settings['Comfy.Execution.PreviewMethod'] == 'none'
    mgr = (comfy/'user'/'__manager'/'config.ini').read_text()
    assert 'preview_method = none' in mgr
    # a leftover auto/taesd setting is overwritten; other settings and args are kept
    (work/'comfyui_args.txt').write_text('# keep me\n--preview-method auto\n--highvram\n')
    (comfy/'user'/'default'/'comfy.settings.json').write_text(json.dumps({
        'Comfy.Execution.PreviewMethod': 'taesd', 'Comfy.Other': True})+'\n')
    (comfy/'user'/'__manager'/'config.ini').write_text('[default]\npreview_method = auto\nchannel_url = x\n')
    assert mod.main() == 0
    args = (work/'comfyui_args.txt').read_text()
    assert '--preview-method none' in args and '--highvram' in args and '# keep me' in args
    assert args.count('--preview-method') == 1
    settings = json.loads((comfy/'user'/'default'/'comfy.settings.json').read_text())
    assert settings == {'Comfy.Execution.PreviewMethod': 'none', 'Comfy.Other': True}
    assert 'preview_method = none' in (comfy/'user'/'__manager'/'config.ini').read_text()
    assert 'channel_url = x' in (comfy/'user'/'__manager'/'config.ini').read_text()


def test_bake_script_seeds_preview_none_into_the_baked_user_tree():
    bake = (ROOT/'docker/bake_comfyui.sh').read_text()
    assert 'Comfy.Execution.PreviewMethod' in bake and 'preview_method = none' in bake
