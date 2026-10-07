"""Model-local PyTorch attention policy and optional Sage index safety guard.

The optional size guard targets native MiniMax's pre-shaped HND tensors. The
stable node defaults to PyTorch at every size. Neither policy changes weights,
allocates contiguous copies, or recovers an already-failed CUDA context.
Background: https://github.com/thu-ml/SageAttention/issues/386
"""

from functools import wraps
import logging
from threading import Lock


log = logging.getLogger(__name__)
INT32_MAX = (1 << 31) - 1
SEQUENCE_BLOCK = 128


def padded_offset_bound(shape, strides, *, sequence_block=SEQUENCE_BLOCK):
    """Conservative largest element offset for a positive-stride HND view.

    Include masked lanes in the last sequence tile: pointer arithmetic happens
    before the mask is applied. Batch/head contributions count too. Python ints
    keep this check from overflowing itself. Offsets are relative to data_ptr(),
    so storage_offset() must not be added. Dimension padding follows Sage's
    64/128/256 head widths; retaining the original strides is conservative when
    Sage would allocate a padded copy.
    """
    shape = tuple(int(value) for value in shape)
    strides = tuple(int(value) for value in strides)
    if len(shape) != 4 or len(strides) != 4:
        raise ValueError('Expected four-dimensional HND shape and strides')
    if sequence_block < 1:
        raise ValueError('Sequence block must be positive')
    if any(size < 1 for size in shape) or any(stride < 0 for stride in strides):
        raise ValueError('Expected non-empty tensors with non-negative strides')
    batch, heads, sequence, width = shape
    padded_sequence = ((sequence + sequence_block - 1) // sequence_block) * sequence_block
    padded_width = max(64, 1 << (width - 1).bit_length())
    return ((batch - 1) * strides[0] + (heads - 1) * strides[1]
            + (padded_sequence - 1) * strides[2] + (padded_width - 1) * strides[3])


def sage_index_risk(q, k, v, *, skip_reshape=False):
    """Return offending tensor metadata, or None for the guarded native layout.

    MiniMax supplies HND tensors with skip_reshape=True. Other input layouts are
    deliberately left to the original backend instead of guessing its reshape.
    """
    if not skip_reshape:
        return None
    for name, tensor in zip(('q', 'k', 'v'), (q, k, v)):
        shape = tuple(int(size) for size in tensor.shape)
        strides = tuple(int(stride) for stride in tensor.stride())
        if len(shape) != 4 or any(size < 1 for size in shape) or any(s < 0 for s in strides):
            continue
        bound = padded_offset_bound(shape, strides)
        if bound > INT32_MAX:
            return {'tensor': name, 'shape': shape, 'strides': strides,
                    'padded_offset': bound, 'limit': INT32_MAX}
    return None


class _SageIndexGuard:
    def __init__(self, sage, pytorch, previous=None, *, force_pytorch=False):
        self.sage_functions = (sage, getattr(sage, '__wrapped__', sage))
        self.pytorch = pytorch
        self.previous = previous
        self.force_pytorch = force_pytorch
        self._logged = False
        self._log_lock = Lock()

    def _dispatch(self, original, q, k, v, *args, **kwargs):
        risk = None
        if not self.force_pytorch:
            if not any(original is sage for sage in self.sage_functions):
                return original(q, k, v, *args, **kwargs)
            # q,k,v,heads,mask,attn_precision,skip_reshape: support positional
            # calls and MiniMax's keyword form without changing arguments.
            skip_reshape = kwargs.get('skip_reshape', args[3] if len(args) > 3 else False)
            risk = sage_index_risk(q, k, v, skip_reshape=skip_reshape)
            if risk is None:
                return original(q, k, v, *args, **kwargs)
        with self._log_lock:
            if not self._logged:
                self._logged = True
                shapes = tuple(tuple(t.shape) for t in (q, k, v))
                if self.force_pytorch:
                    log.info('[H3 Attention] Model stability profile selects PyTorch '
                             'attention at every size; q/k/v shapes=%s.', shapes)
                else:
                    log.warning(
                        '[H3 Attention] Using PyTorch for Sage int32 offset risk: '
                        '%s shape=%s strides=%s padded_offset=%s > %s; '
                        'q/k/v shapes=%s; sequence block=%s.',
                        risk['tensor'], risk['shape'], risk['strides'],
                        risk['padded_offset'], risk['limit'], shapes, SEQUENCE_BLOCK,
                    )
        return self.pytorch(q, k, v, *args, **kwargs)

    def __call__(self, original, q, k, v, *args, **kwargs):
        if self.previous is None:
            return self._dispatch(original, q, k, v, *args, **kwargs)

        @wraps(original)
        def guarded_original(*inner_args, **inner_kwargs):
            return self._dispatch(original, *inner_args, **inner_kwargs)

        # Keep an existing override's transformations and explicit replacement.
        # Apply our policy if that override delegates to the original backend.
        return self.previous(guarded_original, q, k, v, *args, **kwargs)


def patch_model_attention(model, *, force_pytorch=False):
    """Clone a ModelPatcher and apply the requested attention policy.

    By default, only risky native Sage calls switch to PyTorch. force_pytorch
    selects PyTorch regardless of size or original backend for this model.
    Existing ordinary overrides are composed. An override owning attention
    containers is preserved unchanged: its private ownership protocol cannot be
    safely composed using the plain-tensor override hook.
    """
    patched = model.clone()
    patched.model_options = dict(patched.model_options)
    options = dict(patched.model_options.get('transformer_options', {}))
    patched.model_options['transformer_options'] = options
    previous = options.get('optimized_attention_override')
    if isinstance(previous, _SageIndexGuard):
        previous = previous.previous
    if previous is not None and hasattr(previous, 'container_function'):
        log.warning('[H3 Attention] Existing container attention override preserved; '
                    'attention safety policy was not installed on this model clone.')
        return patched
    # Importing this helper or running its arithmetic tests does not import
    # ComfyUI, Torch, Sage, or initialize CUDA.
    from importlib import import_module
    attention = import_module('comfy.ldm.modules.attention')
    options['optimized_attention_override'] = _SageIndexGuard(
        attention.attention_sage, attention.attention_pytorch, previous,
        force_pytorch=force_pytorch)
    return patched


class H3StudioAttentionSafety:
    """Stable H3 profile; opt out to retain Sage with the size-based guard."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'force_pytorch': ('BOOLEAN', {'default': True}),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def patch(self, model, force_pytorch=True):
        return (patch_model_attention(model, force_pytorch=force_pytorch),)
