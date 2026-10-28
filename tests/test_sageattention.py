"""SageAttention is baked into the image for every card whose workflow selects it.

The Hearmeman video graphs call KJNodes' sageattn_qk_int8_pv_fp8_cuda++ option.
That imports sageattn_qk_int8_pv_fp8_cuda and passes pv_accum_dtype="fp32+fp16".
Those cards must not opt into the old runtime profile: that profile used to
compile on the pod and replace onnxruntime-gpu 1.30.0 with 1.22.0 plus
opencv-contrib-python.
"""
import asyncio
import json
from pathlib import Path

import pytest

import launcher.app as launcher_app

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "selfism_workflows"
# File stem is the Selfism profile that installs the graph.
SAGE_CARDS = {
    "hearmeman_wan_v5": "sageattn_qk_int8_pv_fp8_cuda++",
    "hearmeman_scail2_v2": "sageattn_qk_int8_pv_fp8_cuda++",
}
# SeedVR2 loaders set to sageattn_2 (from sageattention import sageattn_varlen, in the
# baked SageAttention 2.2.0 wheel). Workflow file -> Selfism profile.
SEEDVR2_SAGE_CARDS = {
    "qwen21_auto_carousel_10sor": "qwen21_auto_carousel",
}


def _nodes(data):
    nodes = list(data.get("nodes") or [])
    for group in (data.get("definitions") or {}).get("subgraphs") or []:
        nodes.extend(group.get("nodes") or [])
    return [node for node in nodes if isinstance(node, dict)]


def sage_attention_values(data):
    """Widget values that select a SageAttention kernel.

    ComfyUI stores the selected combo in widgets_values. Named widgets cover
    loaders that also keep attention_mode beside the value list.
    """
    found = []

    def consider(node, key, value):
        if not isinstance(value, str):
            return
        token = value.strip()
        if " " in token or len(token) > 80 or not token.lower().startswith("sage"):
            return
        found.append((node.get("id"), node.get("type"), key, token))

    for node in _nodes(data):
        named = node.get("widgets_values_named") or {}
        if isinstance(named, dict):
            for key, value in named.items():
                if key in {"sage_attention", "attention_mode", "attention"} or (
                    isinstance(value, str) and value.lower().startswith("sage")
                ):
                    consider(node, key, value)
        values = node.get("widgets_values")
        if isinstance(values, list):
            for value in values:
                consider(node, "widgets_values", value)
        elif isinstance(values, dict):
            for key, value in values.items():
                consider(node, key, value)
    return found


def test_every_sage_workflow_is_a_card_the_image_bake_covers():
    catalog = json.loads((ROOT / "catalog/selfism.json").read_text(encoding="utf-8"))
    discovered = {}
    for path in sorted(WORKFLOWS.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        values = sage_attention_values(data)
        if not values:
            continue
        discovered[path.relative_to(ROOT).as_posix()] = values

    for stem, profile in SEEDVR2_SAGE_CARDS.items():
        values = discovered.pop(f"selfism_workflows/{stem}.json")
        assert {(item[1], item[3]) for item in values} == {("SeedVR2LoadDiTModel", "sageattn_2")}
        assert "ComfyUI-SeedVR2_VideoUpscaler" in catalog[profile + "_nodes"]
    assert set(discovered) == {f"selfism_workflows/{name}.json" for name in SAGE_CARDS}
    for name, expected in SAGE_CARDS.items():
        values = discovered[f"selfism_workflows/{name}.json"]
        assert {(item[1], item[3]) for item in values} == {("DiffusionModelLoaderKJ", expected)}
        assert 877 in {item[0] for item in values}
        assert name + "_files" in catalog
        assert "ComfyUI-KJNodes" in catalog[name + "_nodes"]

    # GOD Mode's WanVideo loader stays on sdpa. It is not a sage card.
    god = json.loads((WORKFLOWS / "wan22_animate_god_mode.json").read_text(encoding="utf-8"))
    assert sage_attention_values(god) == []
    named = [
        (node.get("widgets_values_named") or {}).get("attention_mode")
        for node in _nodes(god)
        if (node.get("widgets_values_named") or {}).get("attention_mode")
    ]
    assert named == ["sdpa"]

    bundled = json.loads((ROOT / "catalog/workflows.json").read_text(encoding="utf-8"))
    assert all(not item.get("runtime_profile") for item in bundled["workflows"])
    # Selfism installs build their payload in code. None of them select the profile.
    assert "runtime_profile" not in (ROOT / "launcher/selfism.py").read_text(encoding="utf-8")


def test_runtime_profile_only_checks_the_baked_package(tmp_path, monkeypatch):
    source = (ROOT / "launcher/app.py").read_text(encoding="utf-8")
    body = source.split("async def _install_runtime_profile", 1)[1].split(
        "\n    async def ", 1
    )[0]
    assert "from sageattention import sageattn_qk_int8_pv_fp8_cuda" in body
    assert 'metadata.version("onnxruntime-gpu") != "1.30.0"' in body
    assert "CUDAExecutionProvider" in body
    assert "opencv-contrib-python" in body and "is installed" in body
    for forbidden in (
        "onnxruntime-gpu==1.22.0",
        "opencv-contrib-python==",
        "apt-get",
        '"pip"',
        "cuda-nvcc",
        "TORCH_CUDA_ARCH_LIST",
    ):
        assert forbidden not in body
    assert launcher_app.SAGEATTENTION_VERSION == "2.2.0"
    arches = set(launcher_app.SAGEATTENTION_ARCHITECTURES.split(";"))
    assert {"8.9", "9.0", "12.0"} <= arches

    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", venv)
    controller = launcher_app.JobController()
    calls = []

    async def fake_process(*command, **kwargs):
        calls.append(tuple(str(part) for part in command))
        script = command[2]
        assert "from sageattention import sageattn_qk_int8_pv_fp8_cuda" in script
        assert "1.30.0" in script
        assert "opencv-contrib-python" in script
        return 0, "sageattn_qk_int8_pv_fp8_cuda 2.2.0+cu130torch2.10.0.cg1\n"

    monkeypatch.setattr(controller, "_run_process", fake_process)
    asyncio.run(controller._install_runtime_profile(launcher_app.SAGEATTENTION_PROFILE))
    assert calls == [(str(python), "-c", calls[0][2])]
    assert controller.state.restart_required is False
    blob = " ".join(calls[0])
    assert "apt-get" not in blob and " pip " not in f" {blob} "


def test_runtime_profile_refuses_to_compile_when_the_bake_is_missing(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", venv)
    controller = launcher_app.JobController()
    calls = []

    async def fake_process(*command, **kwargs):
        calls.append(tuple(str(part) for part in command))
        return 1, "sageattention fp8 CUDA extension is not loaded\n"

    monkeypatch.setattr(controller, "_run_process", fake_process)
    with pytest.raises(RuntimeError, match="SageAttention runtime verification failed"):
        asyncio.run(controller._install_runtime_profile(launcher_app.SAGEATTENTION_PROFILE))
    assert len(calls) == 1
    assert "pip" not in " ".join(calls[0])
    assert controller.state.restart_required is False
