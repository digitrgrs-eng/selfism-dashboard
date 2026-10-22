"""Custom-node git installs survive GitHub refusing lazy blob fetches."""

import asyncio
import json

from launcher import app as launcher_app

REPO = "https://github.com/kijai/ComfyUI-KJNodes"
REF = "1289b52fbb6d64a339a4047b9ea74cf7758ccf1e"
NODE = {"name": "ComfyUI-KJNodes", "repo": REPO, "ref": REF, "install_requirements": False}
REFUSED = (
    "fatal: could not read Username for 'https://github.com': No such device or address\n"
    f"fatal: could not fetch {REF} from promisor remote\n"
    "warning: Clone succeeded, but checkout failed."
)


def setup(tmp_path, monkeypatch):
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv")
    return launcher_app.JobController(), custom_nodes / "ComfyUI-KJNodes"


def fake_archive(calls):
    async def download(repo, ref, destination):
        calls.append((repo, ref))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / launcher_app.SOURCE_MARKER).write_text(
            json.dumps({"repo": repo, "ref": ref, "source": "github-tarball"})
        )

    return download


def test_every_git_call_cannot_prompt(tmp_path, monkeypatch) -> None:
    controller, destination = setup(tmp_path, monkeypatch)
    seen = []

    async def run(*command, **bounds):
        command = tuple(str(part) for part in command)
        seen.append((command, bounds))
        if "clone" in command:
            destination.mkdir()
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node(NODE))

    git_calls = [(c, b) for c, b in seen if c[0] == "git"]
    assert git_calls
    for command, bounds in git_calls:
        assert command[1:3] == ("-c", "credential.helper=")
        assert bounds["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert bounds["env"]["GCM_INTERACTIVE"] == "never"
        assert "--filter=blob:none" not in command


def test_failed_shallow_fetch_retries_then_succeeds(tmp_path, monkeypatch) -> None:
    """A refused one-commit fetch is retried, still without a partial clone."""
    monkeypatch.setattr(launcher_app, "GIT_RETRY_DELAYS", (0.0,))
    controller, _destination = setup(tmp_path, monkeypatch)
    fetches = []

    async def run(*command, **_bounds):
        command = tuple(str(part) for part in command)
        if "fetch" in command:
            fetches.append(command)
            if len(fetches) == 1:
                return 128, REFUSED
            return 0, ""
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node(NODE))

    assert len(fetches) == 2
    assert all("--depth" in command and "1" in command for command in fetches)
    assert all("--filter=blob:none" not in command for command in fetches)
    assert controller.state.restart_required is True


def test_checkout_failure_after_clone_falls_back_to_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher_app, "GIT_RETRY_DELAYS", (0.0,))
    controller, _destination = setup(tmp_path, monkeypatch)
    archives = []
    monkeypatch.setattr(launcher_app, "download_github_tarball", fake_archive(archives))

    async def run(*command, **_bounds):
        command = tuple(str(part) for part in command)
        if "clone" in command:
            destination.mkdir(exist_ok=True)
            return 0, ""
        if "checkout" in command:
            return 128, REFUSED
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node(NODE))
    assert archives == [(REPO, REF)]


def test_fetch_failure_retries_then_falls_back_to_archive(tmp_path, monkeypatch) -> None:
    controller, destination = setup(tmp_path, monkeypatch)
    destination.mkdir()
    archives = []
    monkeypatch.setattr(launcher_app, "download_github_tarball", fake_archive(archives))
    fetches = []
    unset = []

    async def run(*command, **_bounds):
        command = tuple(str(part) for part in command)
        if "get-url" in command:
            return 0, REPO + ".git\n"
        if "--get" in command:
            return 0, "true\n"  # partial clone from an earlier release
        if "--unset" in command:
            unset.append(command[-1])
            return 0, ""
        if "fetch" in command:
            fetches.append(command)
            return 128, REFUSED
        if "cat-file" in command:
            return 1, ""
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node(NODE))

    assert unset == ["remote.origin.promisor", "remote.origin.partialclonefilter"]
    assert all("--refetch" in f for f in fetches)
    assert len(fetches) == 1 + len(launcher_app.GIT_RETRY_DELAYS)
    assert archives == [(REPO, REF)]
    assert (destination / launcher_app.SOURCE_MARKER).exists()

    # The next run recognises the archive install and does not touch git at all.
    calls = []

    async def no_git(*command, **_bounds):
        calls.append(command)
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", no_git)
    asyncio.run(controller._install_custom_node(NODE))
    assert calls == []


def test_tarball_url_is_the_pinned_commit() -> None:
    assert launcher_app.github_tarball_url(REPO + ".git", REF) == (
        f"https://codeload.github.com/kijai/ComfyUI-KJNodes/tar.gz/{REF}"
    )


def _cache_pack(tmp_path, monkeypatch, *, ref=REF, requirements_baked=True):
    controller, destination = setup(tmp_path, monkeypatch)
    cache = tmp_path / "node_cache" / "ComfyUI-KJNodes"
    cache.mkdir(parents=True)
    (cache / "FROM_CACHE").write_text("baked", encoding="utf-8")
    (cache / ".git").mkdir()
    (cache / "requirements.txt").write_text("not-a-real-package\n", encoding="utf-8")
    (cache / launcher_app.SOURCE_MARKER).write_text(
        json.dumps(
            {
                "repo": REPO + ".git",
                "ref": ref,
                "source": "image-cache",
                "requirements_baked": requirements_baked,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher_app, "NODE_CACHE_DIR", tmp_path / "node_cache")
    return controller, destination


def test_a_matching_baked_pack_is_copied_and_skips_git_and_pip(tmp_path, monkeypatch, capsys) -> None:
    controller, destination = _cache_pack(tmp_path, monkeypatch)
    calls = []

    async def run(*command, **_bounds):
        calls.append(tuple(str(part) for part in command))
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node({**NODE, "install_requirements": True}))

    assert (destination / "FROM_CACHE").read_text() == "baked"
    assert not (destination / ".git").exists()
    assert calls == []
    assert "copied in " in capsys.readouterr().out


def test_a_baked_pack_without_the_dep_marker_still_installs_requirements(
    tmp_path, monkeypatch
) -> None:
    controller, destination = _cache_pack(tmp_path, monkeypatch, requirements_baked=False)
    pip = []

    async def run(*command, **_bounds):
        command = tuple(str(part) for part in command)
        if "pip" in command:
            pip.append(command)
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node({**NODE, "install_requirements": True}))

    assert (destination / "FROM_CACHE").is_file()
    assert pip and "-r" in pip[0]
    assert any(part.endswith("requirements.txt") for part in pip[0])


def test_a_ref_mismatch_falls_through_to_a_shallow_fetch(tmp_path, monkeypatch) -> None:
    controller, destination = _cache_pack(tmp_path, monkeypatch, ref="b" * 40)
    fetches = []

    async def run(*command, **_bounds):
        command = tuple(str(part) for part in command)
        if "fetch" in command:
            fetches.append(command)
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", run)
    asyncio.run(controller._install_custom_node(NODE))

    assert fetches and "--depth" in fetches[0] and REF in fetches[0]
    assert "--filter=blob:none" not in fetches[0]
    assert not (destination / "FROM_CACHE").exists()
