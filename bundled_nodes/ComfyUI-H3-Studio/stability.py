"""Check ComfyUI's active weight loader before H3 starts model work."""


def _open_readonly_mapped(*args, **kwargs):
    from .mapped_safetensors import ReadOnlyMappedSafetensors
    return ReadOnlyMappedSafetensors(*args, **kwargs)


def _load_readonly_safetensors(filename):
    with _open_readonly_mapped(filename, framework='pt', device='cpu') as source:
        state_dict = {name: source.get_tensor(name) for name in source.keys()}
        metadata = source.metadata()
    return state_dict, {} if metadata is None else metadata


class _CpuMappedSafetensors:
    """Limit the CPU loader override to ComfyUI's utility-module reference."""

    _h3_cpu_mapped_installed = True

    def __init__(self, original):
        self._original_safetensors = original
        self._original_safe_open = original.safe_open

    def __getattr__(self, name):
        return getattr(self._original_safetensors, name)

    def safe_open(self, *args, **kwargs):
        device = kwargs.get('device', args[2] if len(args) > 2 else 'cpu')
        framework = kwargs.get('framework', args[1] if len(args) > 1 else None)
        if (len(args) <= 3 and 'backend' not in kwargs and (device is None or str(device) == 'cpu')
                and framework in ('pt', 'torch', 'pytorch')):
            return _open_readonly_mapped(*args, **kwargs)
        return self._original_safe_open(*args, **kwargs)


def _ensure_windows_cpu_loader(import_module):
    import sys

    if sys.platform != 'win32':
        return

    import inspect
    from packaging.version import InvalidVersion, Version

    utils = import_module('comfy.utils')
    safetensors = getattr(utils, 'safetensors', None)

    if getattr(safetensors, '_h3_cpu_mapped_installed', False) is not True:
        try:
            supported = (
                Version(safetensors.__version__) >= Version('0.8.0')
                and 'backend' in inspect.signature(safetensors.safe_open).parameters
            )
        except (AttributeError, InvalidVersion, TypeError, ValueError):
            supported = False
        if not supported:
            raise RuntimeError(
                'MiniMax H3 Studio stopped before loading models because the Windows '
                'CPU model loader needs safetensors>=0.8.0 with backend selection. '
                'Its default mmap path can crash inside torch storage while opening '
                'this model; --disable-mmap only copies tensors after that operation. '
                'Close ComfyUI, install the update using its Python environment '
                '(python -m pip install "safetensors>=0.8.0"), then restart ComfyUI. '
                'If ComfyUI does not expose its Safetensors loader, update ComfyUI too. '
                'The Studio uses read-only mapped CPU buffers to avoid keeping a '
                'complete extra copy of every model in RAM.'
            )

        utils.safetensors = _CpuMappedSafetensors(safetensors)

    original_loader = getattr(utils, 'load_safetensors', None)
    if getattr(original_loader, '_h3_mapped_loader_installed', False) is not True:
        def mapped_loader(filename):
            return _load_readonly_safetensors(filename)

        mapped_loader._h3_mapped_loader_installed = True
        mapped_loader._h3_original_load_safetensors = original_loader
        # Dynamic VRAM calls this entry point directly. Its tensors deliberately
        # omit _comfy_tensor_file_slice so the ordinary RAM-copy path is used.
        utils.load_safetensors = mapped_loader


def ensure_stable_loading():
    """Check loader state and configure CPU reads without opening model files."""
    from importlib import import_module

    try:
        memory_management = import_module('comfy.memory_management')
    except ModuleNotFoundError as error:
        # Older ComfyUI cores predate this module and its dynamic VRAM loader.
        # A missing dependency inside an installed module is still an error.
        if error.name != 'comfy.memory_management':
            raise
        memory_management = None

    _ensure_windows_cpu_loader(import_module)
