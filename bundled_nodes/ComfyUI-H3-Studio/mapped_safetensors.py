"""Read-only CPU tensor views without PyTorch's file-storage slicing path.

Returned tensors share the read-only model mapping. A consumer that changes
weights in place must first clone them, as with ComfyUI's native mapped reader.
No returned tensors are cached here; their buffer references own the mapping.
"""

import json
import math
import mmap
import os
import struct
import sys
import warnings


_MAX_HEADER_BYTES = 100_000_000
_DTYPES = {
    'F64': ('float64', 8), 'F32': ('float32', 4),
    'F16': ('float16', 2), 'BF16': ('bfloat16', 2),
    'I64': ('int64', 8), 'I32': ('int32', 4),
    'I16': ('int16', 2), 'I8': ('int8', 1),
    'U64': ('uint64', 8), 'U32': ('uint32', 4),
    'U16': ('uint16', 2), 'U8': ('uint8', 1),
    'BOOL': ('bool', 1), 'C64': ('complex64', 8),
    'F8_E4M3': ('float8_e4m3fn', 1),
    'F8_E5M2': ('float8_e5m2', 1),
    'F8_E4M3FNUZ': ('float8_e4m3fnuz', 1),
    'F8_E5M2FNUZ': ('float8_e5m2fnuz', 1),
    'F8_E8M0': ('float8_e8m0fnu', 1),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate Safetensors header key: {key!r}')
        result[key] = value
    return result


def _validate_header(header, data_size):
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be a JSON object')
    metadata = header.get('__metadata__')
    if '__metadata__' in header and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items())
    ):
        raise ValueError('Safetensors metadata must contain only strings')

    tensors = {}
    spans = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(info, dict):
            raise ValueError(f'Invalid tensor record: {name!r}')
        dtype = info.get('dtype')
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError(f'Unsupported Safetensors dtype for {name!r}: {dtype!r}')
        shape = info.get('shape')
        if (not isinstance(shape, list)
                or any(type(dim) is not int or not 0 <= dim <= sys.maxsize for dim in shape)):
            raise ValueError(f'Invalid tensor shape: {name!r}')
        offsets = info.get('data_offsets')
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise ValueError(f'Invalid tensor offsets: {name!r}')
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f'Tensor offsets extend beyond the data: {name!r}')
        if math.prod(shape) * _DTYPES[dtype][1] != end - start:
            raise ValueError(f'Tensor shape and byte size disagree: {name!r}')
        tensors[name] = (dtype, tuple(shape), start, end)
        spans.append((start, end, name))

    previous_end = 0
    for start, end, name in sorted(spans):
        if start != previous_end:
            raise ValueError(f'Tensor offsets overlap or leave a gap: {name!r}')
        previous_end = end
    if previous_end != data_size:
        raise ValueError('Safetensors data contains unclaimed bytes')
    return tensors, metadata


class ReadOnlyMappedSafetensors:
    """Small safe_open-compatible surface used by comfy.utils.load_torch_file."""

    def __init__(self, filename, framework='pt', device='cpu'):
        if framework not in ('pt', 'torch', 'pytorch'):
            raise ValueError('The mapped reader supports PyTorch only')
        if device is not None and str(device) != 'cpu':
            raise ValueError('The mapped reader supports CPU tensors only')
        if sys.byteorder != 'little':
            raise ValueError('The mapped reader requires a little-endian host')

        self._mapping = None
        with open(os.fspath(filename), 'rb') as source:
            size = os.fstat(source.fileno()).st_size
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError('Incomplete Safetensors header length')
            header_size = struct.unpack('<Q', prefix)[0]
            if header_size > _MAX_HEADER_BYTES:
                raise ValueError('Safetensors header exceeds the size limit')
            self._data_base = 8 + header_size
            if self._data_base > size:
                raise ValueError('Incomplete Safetensors header')
            try:
                header = json.loads(source.read(header_size).decode('utf-8'),
                                    object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError('Invalid Safetensors JSON header') from error
            self._tensors, self._metadata = _validate_header(header, size - self._data_base)

            import torch
            self._torch = torch
            self._torch_dtypes = {}
            for dtype, _, _, _ in self._tensors.values():
                attribute = _DTYPES[dtype][0]
                resolved = getattr(torch, attribute, None)
                if resolved is None:
                    raise ValueError(f'Installed PyTorch does not support {dtype}')
                self._torch_dtypes[dtype] = resolved
            # ACCESS_READ does not reserve private writable copies of the file.
            self._mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)

    def _require_open(self):
        if self._mapping is None:
            raise ValueError('Safetensors file is closed')

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        # Do not close exported mappings: tensors retain their own buffer refs.
        # Dropping this reference closes an unused mapping immediately, or lets
        # the last surviving tensor release it later without a global cache.
        self._mapping = None

    def keys(self):
        self._require_open()
        return sorted(self._tensors)

    def metadata(self):
        self._require_open()
        return None if self._metadata is None else dict(self._metadata)

    def get_tensor(self, name):
        self._require_open()
        dtype_name, shape, start, end = self._tensors[name]
        dtype = self._torch_dtypes[dtype_name]
        if start == end:
            return self._torch.empty(shape, dtype=dtype, device='cpu')
        view = memoryview(self._mapping)[self._data_base + start:self._data_base + end]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable')
            tensor = self._torch.frombuffer(view, dtype=dtype).reshape(shape)
        # frombuffer itself retains its buffer. Keep explicit storage ownership
        # as well, including when callers replace the Tensor with a Parameter.
        tensor.untyped_storage()._h3_mapped_safetensors_refs = (self._mapping, view)
        return tensor
