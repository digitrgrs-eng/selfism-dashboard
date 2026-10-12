import importlib.metadata
import subprocess
import sys

from launcher import selfism_runtime as rt


def test_repair_installs_llama_cpp_runtime_deps_before_verifying(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(sys, "argv", ["selfism_runtime"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(rt.platform, "machine", lambda: "x86_64")
    lib = tmp_path / "nvidia" / "x" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcublas.so.12").touch()
    (lib / "libnccl.so.2").touch()
    monkeypatch.setattr(rt, "cuda_folders", lambda: [str(lib)])
    monkeypatch.setattr(rt, "Path", lambda p: tmp_path / "ld.conf" if str(p).startswith("/etc") else __import__("pathlib").Path(p))
    monkeypatch.setattr(rt, "run", lambda command, **_k: calls.append(list(map(str, command))))
    monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    monkeypatch.setattr(rt.importlib.metadata, "version",
                        lambda name: {"torch": "2.9.0+cu130"}.get(name, "0.4.1+cu128"))

    rt.main()

    deps = next(i for i, c in enumerate(calls) if "diskcache" in c)
    wheel = next(i for i, c in enumerate(calls) if rt.WHEEL in c)
    verify = max(i for i, c in enumerate(calls) if "-c" in c and rt.VERIFY in c)
    assert wheel < deps < verify
    for name in ("numpy", "jinja2", "typing-extensions"):
        assert name in calls[deps]
    assert "--upgrade" not in calls[deps] and "--force-reinstall" not in calls[deps]
