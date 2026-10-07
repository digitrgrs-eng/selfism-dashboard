"""Content-cached SAM3 masks and conservative preserve-mask cleanup.

The cache is independent of generation prompts, seeds, LoRAs, and output size.
Feed the same fixed-resolution source frames to both detectors.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)
CACHE_VERSION = 1


class H3CharacterCachedSAM3:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            'required': {
                'frames': ('IMAGE', {'tooltip': 'Stable source frames before inversion. Use a fixed analysis size so output-quality changes can reuse masks.'}),
                'text': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'The SAM3 target or preserve description. Changing it invalidates this mask cache.'}),
                'checkpoint_name': (folder_paths.get_filename_list('checkpoints'), {'tooltip': 'Must match the connected SAM3 checkpoint loader. Its file identity is part of the cache key.'}),
                'threshold': ('FLOAT', {'default': .5, 'min': 0., 'max': 1., 'step': .01, 'tooltip': 'Minimum detection confidence. Preserve should reject weak guesses; 0 accepts the highest-scoring guess even if the requested object is absent.'}),
                'refine_iterations': ('INT', {'default': 2, 'min': 0, 'max': 5, 'tooltip': 'Native SAM3 refinement passes. Changing this recomputes the mask.'}),
                'use_cache': ('BOOLEAN', {'default': True, 'tooltip': 'Reuse a saved mask when source pixels, description, SAM3 model, and detection settings match. Turn off to recompute without reading or updating the cache.'}),
            },
            'optional': {
                'model': ('MODEL', {'lazy': True, 'tooltip': 'SAM3 model; not requested on a valid cache hit.'}),
                'clip': ('CLIP', {'lazy': True, 'tooltip': 'SAM3 text encoder from the same checkpoint; not requested on a valid cache hit.'}),
            },
        }

    RETURN_TYPES = ('MASK',)
    RETURN_NAMES = ('masks',)
    FUNCTION = 'detect'
    CATEGORY = 'MiniMax Safe/Character'

    @classmethod
    def IS_CHANGED(cls, use_cache=True, **kwargs):
        # Comfy's in-memory output cache otherwise skips the node before our
        # disk-cache switch is evaluated. Off means recompute on every run.
        return 'reuse-sam3-mask-v1' if use_cache else float('nan')

    @staticmethod
    def _cache_directory():
        import folder_paths
        return Path(folder_paths.get_user_directory()) / 'default' / 'h3_character_mask_cache'

    @staticmethod
    def _checkpoint_identity(checkpoint_name):
        import folder_paths
        path = folder_paths.get_full_path('checkpoints', checkpoint_name)
        if not path or not Path(path).is_file():
            raise ValueError(f'SAM3 checkpoint does not exist: {checkpoint_name}')
        path = Path(path).resolve()
        stat = path.stat()
        return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    @staticmethod
    def _detector_identity():
        import folder_paths
        root = Path(folder_paths.__file__).resolve().parent
        paths = [Path(__file__).resolve(), root / 'comfy_extras' / 'nodes_sam3.py']
        paths.extend(sorted((root / 'comfy' / 'ldm' / 'sam3').rglob('*.py')))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(str(path).encode('utf-8'))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _key(self, frames, text, checkpoint_name, threshold, refine_iterations):
        if (not isinstance(frames, torch.Tensor) or frames.ndim != 4
                or frames.shape[-1] not in (3, 4) or any(size <= 0 for size in frames.shape)
                or not frames.is_floating_point()):
            raise ValueError('SAM3 analysis frames must be non-empty floating-point RGB/RGBA images [frames, height, width, channels].')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Enter a SAM3 description, or leave the optional preserve control blank to skip its branch.')
        if not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('SAM3 threshold must be between 0 and 1.')
        if isinstance(refine_iterations, bool) or not isinstance(refine_iterations, int) or not 0 <= refine_iterations <= 5:
            raise ValueError('SAM3 refinement must be an integer between 0 and 5.')
        metadata = {
            'version': CACHE_VERSION, 'shape': list(frames.shape), 'dtype': str(frames.dtype),
            'text': text, 'checkpoint': self._checkpoint_identity(checkpoint_name),
            'detector': self._detector_identity(), 'threshold': float(threshold),
            'refine_iterations': refine_iterations, 'individual_masks': False,
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        # One frame at a time bounds temporary CPU memory and includes all
        # pixels, not just filenames or sparse samples of the video.
        for frame in frames:
            cpu = frame.detach().to(device='cpu').contiguous()
            if not bool(torch.isfinite(cpu).all()):
                raise ValueError('SAM3 analysis frames contain non-finite pixel values.')
            digest.update(cpu.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _validate_mask(mask, expected_shape):
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(f'SAM3 mask does not match the analysis frames; expected {tuple(expected_shape)}.')
        if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError('SAM3 mask contains non-finite values or values outside 0–1.')

    def _read_cache(self, key, expected_shape):
        path = self._cache_directory() / (key + '.npz')
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {'mask', 'metadata'}:
                    raise ValueError('Unexpected cache fields')
                metadata = json.loads(str(archive['metadata'].item()))
                array = archive['mask']
                if (metadata.get('version') != CACHE_VERSION or metadata.get('key') != key
                        or list(array.shape) != list(expected_shape)
                        or array.dtype not in (np.dtype('uint8'), np.dtype('float32'))):
                    raise ValueError('Cache identity, shape or dtype differs')
                checksum = hashlib.sha256(array.tobytes(order='C')).hexdigest()
                if checksum != metadata.get('checksum'):
                    raise ValueError('Cache content checksum differs')
                mask = torch.from_numpy(array.copy()).float()
                self._validate_mask(mask, expected_shape)
                return mask
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
            log.warning('[H3 Character] Ignoring invalid SAM3 cache %s: %s', path.name, error)
            return None

    def _write_cache(self, key, mask):
        directory = self._cache_directory()
        directory.mkdir(parents=True, exist_ok=True)
        array = mask.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
        # Native detection outputs binary masks. Retain fractional values if a
        # future compatible detector returns soft masks.
        if np.logical_or(array == 0, array == 1).all():
            array = array.astype(np.uint8)
        metadata = json.dumps({'version': CACHE_VERSION, 'key': key,
                               'checksum': hashlib.sha256(array.tobytes(order='C')).hexdigest()})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=directory, prefix=key + '.', suffix='.pending', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, mask=array, metadata=np.array(metadata))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / (key + '.npz'))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def check_lazy_status(self, frames, text, checkpoint_name, threshold, refine_iterations, use_cache=True, model=None, clip=None):
        key = self._key(frames, text, checkpoint_name, threshold, refine_iterations)
        cached = self._read_cache(key, frames.shape[:3]) if use_cache else None
        self._pending_cache = (key, cached) if cached is not None else None
        if cached is not None:
            return []
        return [name for name, value in [('model', model), ('clip', clip)] if value is None]

    @staticmethod
    def _run_detector(frames, text, threshold, refine_iterations, model, clip):
        import nodes
        from comfy_extras.nodes_sam3 import SAM3_Detect
        conditioning = nodes.CLIPTextEncode().encode(clip, text)[0]
        return SAM3_Detect.execute(model=model, image=frames, conditioning=conditioning,
                                  threshold=threshold, refine_iterations=refine_iterations,
                                  individual_masks=False)[0]

    def detect(self, frames, text, checkpoint_name, threshold, refine_iterations, use_cache=True, model=None, clip=None):
        key = self._key(frames, text, checkpoint_name, threshold, refine_iterations)
        pending = getattr(self, '_pending_cache', None)
        self._pending_cache = None
        if use_cache:
            cached = pending[1] if pending is not None and pending[0] == key else self._read_cache(key, frames.shape[:3])
            if cached is not None:
                log.info('[H3 Character] Reusing SAM3 mask cache %s', key[:12])
                return (cached,)
        if model is None or clip is None:
            raise ValueError('Connect both MODEL and CLIP from the SAM3 checkpoint loader. They are skipped only for valid cache hits.')
        mask = self._run_detector(frames, text, threshold, refine_iterations, model, clip)
        self._validate_mask(mask, frames.shape[:3])
        mask = mask.detach().to(device='cpu', dtype=torch.float32).clone()
        if use_cache:
            try:
                self._write_cache(key, mask)
                log.info('[H3 Character] Saved SAM3 mask cache %s', key[:12])
            except OSError as error:
                log.warning('[H3 Character] SAM3 mask completed, but its cache could not be saved: %s', error)
        return (mask,)


class H3CharacterPreserveCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mask': ('MASK',),
            'enabled': ('BOOLEAN', {'default': False, 'tooltip': 'Optional cleanup of tiny enclosed holes and isolated specks in the preserve mask. It cannot recover an undetected part or fix the wrong object.'}),
            'max_region_pixels': ('INT', {'default': 16, 'min': 0, 'max': 1024, 'tooltip': 'Largest isolated foreground speck or enclosed background hole to remove/fill at the analysis resolution. Keep small to retain narrow details.'}),
        }}

    RETURN_TYPES = ('MASK',)
    RETURN_NAMES = ('cleaned_preserve_mask',)
    FUNCTION = 'clean'
    CATEGORY = 'MiniMax Safe/Character'

    def clean(self, mask, enabled=False, max_region_pixels=16):
        if not enabled or max_region_pixels == 0:
            return (mask,)
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3) or any(v <= 0 for v in mask.shape):
            raise ValueError('Preserve cleanup expects a non-empty mask [frames, height, width].')
        if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError('Preserve mask values must be finite and between 0 and 1.')
        if isinstance(max_region_pixels, bool) or not isinstance(max_region_pixels, int) or not 0 <= max_region_pixels <= 1024:
            raise ValueError('Cleanup size must be an integer between 0 and 1024 pixels.')
        batch = mask[None] if mask.ndim == 2 else mask
        output = batch.detach().to(device='cpu', dtype=torch.float32).clone()
        for index, frame in enumerate(batch):
            binary = (frame.detach().to(device='cpu').numpy() >= .5).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] <= max_region_pixels:
                    output[index][torch.from_numpy(labels == component)] = 0
                    binary[labels == component] = 0
            # Never fill background connected to the image border: doing so
            # would manufacture missing subject parts or an entire object.
            count, holes, stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=8)
            border = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
            for component in range(1, count):
                if component not in border and stats[component, cv2.CC_STAT_AREA] <= max_region_pixels:
                    output[index][torch.from_numpy(holes == component)] = 1
        return (output[0] if mask.ndim == 2 else output,)


MASK_CONTROL_NODES = {c.__name__: c for c in (H3CharacterCachedSAM3, H3CharacterPreserveCleanup)}
