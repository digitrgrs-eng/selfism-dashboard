"""Local media, cached SAM3 guidance, and temporary previews for H3 Studio."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import folder_paths
import nodes
from comfy_execution.graph_utils import GraphBuilder
from .stability import ensure_stable_loading


MODES = ['Reference to video', 'First + last frame', 'Video inpainting']
MASK_MODES = ['No mask', 'SAM3', 'Rectangle']
TREATMENTS = ['Original', 'Isolate selection', 'Invert selected colors']
SAM_CHECKPOINT = 'sam3.1_multiplex_fp16.safetensors'
RECTANGLE = '[0.25, 0.2, 0.75, 0.8]'


def input_file(name):
    """Resolve only files inside ComfyUI input, including through symlinks."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Choose or upload the media required by the selected mode.')
    if name.endswith((' [input]', ' [output]', ' [temp]')):
        raise ValueError('Choose an uploaded input filename without a storage annotation.')
    root = Path(folder_paths.get_input_directory()).resolve()
    relative = Path(name)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Choose media from the ComfyUI input folder.')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'The selected input media is missing or outside the input folder: {name}')
    return path


def file_identity(name):
    path = input_file(name)
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def media_identity(name):
    if not name:
        return None
    try:
        return file_identity(name)
    except (OSError, ValueError):
        # Unused selectors may retain a deleted upload. The selected branch
        # validates its files before decoding them.
        return {'unavailable': name}


def analysis_images(images):
    height, width = images.shape[1:3]
    scale = min(1., math.sqrt(500_000 / (height * width)))
    if scale == 1:
        return images
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return F.interpolate(images.movedim(-1, 1), size=size, mode='nearest-exact').movedim(1, -1)


def mask_keys(source):
    if source['kind'] == 'video':
        return ['frames']
    return ['reference', 'last'] if source['last'] is not None else ['reference']


class H3StudioMedia:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'ref_image': ('STRING', {'default': '', 'tooltip': 'Reference image, or first frame. Upload with the media card.'}),
            'last_image': ('STRING', {'default': '', 'tooltip': 'Used only in First + last frame mode.'}),
            'source_video': ('STRING', {'default': '', 'tooltip': 'Used only in Video inpainting mode.'}),
            'start_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 1e6, 'step': .01}),
            'end_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 1e6, 'step': .01,
                                      'tooltip': 'Video trim end; zero uses the end of the source video.'}),
        }}

    RETURN_TYPES = ('H3_STUDIO_MEDIA',)
    RETURN_NAMES = ('media',)
    FUNCTION = 'select'
    CATEGORY = 'MiniMax H3 Studio'

    @classmethod
    def IS_CHANGED(cls, ref_image='', last_image='', source_video='', start_seconds=0., end_seconds=0.):
        return (media_identity(ref_image), media_identity(last_image), media_identity(source_video),
                start_seconds, end_seconds)

    def select(self, ref_image='', last_image='', source_video='', start_seconds=0., end_seconds=0.):
        return ({'ref_image': ref_image, 'last_image': last_image, 'source_video': source_video,
                 'start_seconds': start_seconds, 'end_seconds': end_seconds},)


class H3StudioSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'media': ('H3_STUDIO_MEDIA',), 'mode': (MODES, {'default': MODES[0]})}}

    RETURN_TYPES = ('H3_STUDIO_SOURCE',)
    RETURN_NAMES = ('source',)
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def load(self, media, mode):
        ensure_stable_loading()
        if mode not in MODES:
            raise ValueError('Choose one of the three H3 Studio modes.')
        source = {'mode': mode, 'kind': 'video' if mode == 'Video inpainting' else 'image',
                  'reference': None, 'last': None, 'frames': None, 'audio': None, 'fps': 24,
                  'identity': {'mode': mode}}
        fields = ['source_video'] if source['kind'] == 'video' else ['ref_image']
        if mode == 'First + last frame':
            fields.append('last_image')
        if source['kind'] == 'video' and media.get('ref_image'):
            fields.append('ref_image')
        for name in fields:
            source['identity'][name] = file_identity(media.get(name, ''))
        for name, key in [('ref_image', 'reference'), ('last_image', 'last')]:
            if name in fields:
                source[key] = nodes.NODE_CLASS_MAPPINGS['LoadImage']().load_image(media[name])[0][:1].clone()
        if source['kind'] == 'video':
            start, end = media.get('start_seconds', 0.), media.get('end_seconds', 0.)
            video = nodes.NODE_CLASS_MAPPINGS['H3CharacterVideoInput']().load(media['source_video'], start, end)[0]
            prepare_node = nodes.NODE_CLASS_MAPPINGS.get('BiggerFishyH3StudioVideoPrepare')
            if prepare_node is None:
                raise RuntimeError('Update both ComfyUI-H3-Studio and ComfyUI-H3-Studio-Support from the same release, then restart ComfyUI. The updated Studio video decoder is missing.')
            source['frames'], source['audio'] = prepare_node().prepare(video, .8, 0)
            source['identity']['trim'] = [start, end]
        return (source,)


class H3StudioAnalysisImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('H3_STUDIO_SOURCE',), 'source_key': (['reference', 'last', 'frames'],)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def prepare(self, source, source_key):
        return (analysis_images(source[source_key]),)


class H3StudioMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('H3_STUDIO_SOURCE',),
            'mask_mode': (MASK_MODES, {'default': 'No mask'}),
            'sam3_target': ('STRING', {'default': 'shirt', 'multiline': True}),
            'preserve_text': ('STRING', {'default': '', 'multiline': True,
                'tooltip': 'Optional SAM3 region excluded from edits. This guides the input; generated pixels are not locked.'}),
            'preserve_refinement': ('INT', {'default': 3, 'min': 0, 'max': 5}),
            'preserve_cleanup': ('BOOLEAN', {'default': True}),
            'reuse_sam3_masks': ('BOOLEAN', {'default': True}),
            'rectangle': ('STRING', {'default': RECTANGLE}),
            'strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': .01,
                'tooltip': 'Input treatment strength. It does not set a generated-video pixel lock.'}),
            'feather': ('INT', {'default': 0, 'min': 0, 'max': 32}),
            'treatment': (TREATMENTS, {'default': 'Original',
                'tooltip': 'Original uses masks for preview and prompt guidance. Isolate dims the surroundings. Color inversion is experimental for still references.'}),
        }}

    RETURN_TYPES = ('H3_STUDIO_GUIDES',)
    RETURN_NAMES = ('guides',)
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def prepare(self, source, mask_mode='No mask', sam3_target='shirt', preserve_text='',
                preserve_refinement=3, preserve_cleanup=True, reuse_sam3_masks=True,
                rectangle=RECTANGLE, strength=1., feather=0, treatment='Original'):
        ensure_stable_loading()
        settings = {'mask_mode': mask_mode, 'sam3_target': sam3_target, 'preserve_text': preserve_text,
                    'preserve_refinement': preserve_refinement, 'preserve_cleanup': preserve_cleanup,
                    'reuse_sam3_masks': reuse_sam3_masks, 'rectangle': rectangle, 'strength': strength,
                    'feather': feather, 'treatment': treatment}
        if mask_mode not in MASK_MODES or treatment not in TREATMENTS:
            raise ValueError('Choose a supported mask mode and input treatment.')
        if mask_mode == 'Rectangle':
            rectangle_values(rectangle)
        if mask_mode == 'SAM3' and not sam3_target.strip():
            raise ValueError('Describe the SAM3 target, or choose No mask.')
        graph = GraphBuilder()
        finish = graph.node('H3StudioApplyMasks', source=source, settings=settings)
        if mask_mode == 'SAM3' or preserve_text.strip():
            checkpoint = graph.node('H3CharacterSAMCheckpoint', checkpoint_name=SAM_CHECKPOINT)
            loader = graph.node('CheckpointLoaderSimple', ckpt_name=checkpoint.out(0))
            for key in mask_keys(source):
                frames = graph.node('H3StudioAnalysisImages', source=source, source_key=key)
                if mask_mode == 'SAM3':
                    target = graph.node('H3CharacterCachedSAM3', frames=frames.out(0), text=sam3_target,
                        checkpoint_name=checkpoint.out(0), threshold=0., refine_iterations=2,
                        use_cache=reuse_sam3_masks, model=loader.out(0), clip=loader.out(1))
                    finish.set_input(key + '_target', target.out(0))
                if preserve_text.strip():
                    preserve = graph.node('H3CharacterCachedSAM3', frames=frames.out(0), text=preserve_text,
                        checkpoint_name=checkpoint.out(0), threshold=.5, refine_iterations=preserve_refinement,
                        use_cache=reuse_sam3_masks, model=loader.out(0), clip=loader.out(1))
                    cleaned = graph.node('H3CharacterPreserveCleanup', mask=preserve.out(0),
                        enabled=preserve_cleanup, max_region_pixels=64)
                    finish.set_input(key + '_preserve', cleaned.out(0))
        return {'result': (finish.out(0),), 'expand': graph.finalize()}


def rectangle_values(rectangle):
    try:
        values = json.loads(rectangle)
    except (TypeError, ValueError) as error:
        raise ValueError('Draw a rectangle or enter [left, top, right, bottom].') from error
    if (not isinstance(values, list) or len(values) != 4 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)):
        raise ValueError('Rectangle requires four finite normalized coordinates.')
    left, top, right, bottom = [min(1., max(0., float(x))) for x in values]
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    if left == right or top == bottom:
        raise ValueError('The rectangle needs nonzero width and height.')
    return left, top, right, bottom


def resize_mask(mask, height, width):
    if mask.shape[-2:] == (height, width):
        return mask
    return F.interpolate(mask[:, None], size=(height, width), mode='bilinear', align_corners=False)[:, 0].clamp(0, 1)


def selected_mask(images, settings, detected=None):
    shape = images.shape[:3]
    if settings['mask_mode'] == 'SAM3':
        if detected is None or tuple(detected.shape) != tuple(shape):
            raise ValueError('SAM3 target does not match its source analysis images.')
        mask = detected.detach().to(device='cpu', dtype=torch.float32).clone()
    else:
        mask = torch.ones(shape, dtype=torch.float32)
        if settings['mask_mode'] == 'Rectangle':
            left, top, right, bottom = rectangle_values(settings['rectangle'])
            height, width = shape[1:]
            mask.zero_()
            mask[:, math.floor(top * height):math.ceil(bottom * height),
                 math.floor(left * width):math.ceil(right * width)] = 1
    feather = settings['feather']
    if feather:
        for index in range(len(mask)):
            mask[index] = torch.from_numpy(cv2.GaussianBlur(mask[index].numpy(), (0, 0), float(feather)))
    return mask.clamp(0, 1)


def preview_images(images):
    height, width = images.shape[1:3]
    scale = min(1., 640 / max(height, width))
    # Even dimensions allow native H.264 encoding for video previews.
    height, width = max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2)
    if images.shape[1:3] != (height, width):
        images = F.interpolate(images.movedim(-1, 1), size=(height, width), mode='bilinear', align_corners=False).movedim(1, -1)
    return images[..., :3].detach().to(device='cpu').clone()


def overlay(images, mask, color):
    result = preview_images(images)
    mask = resize_mask(mask, *result.shape[1:3])
    tint = result.new_tensor(color)
    for index in range(len(result)):
        result[index].lerp_(tint, mask[index, ..., None].to(result.dtype) * .45)
    return result


def apply_treatment(images, target, preserve, treatment, strength):
    if treatment == 'Original':
        return images
    height, width = images.shape[1:3]
    selected = resize_mask(target, height, width)
    keep = resize_mask(preserve, height, width)
    edit = selected * (1 - keep)
    result = images.clone()
    for index in range(len(result)):
        rgb = images[index, ..., :3]
        if treatment == 'Invert selected colors':
            result[index, ..., :3] = torch.lerp(rgb, 1 - rgb, (edit[index, ..., None] * strength).to(rgb.device))
        else:
            retained = (edit[index] + keep[index]).clamp(0, 1)[..., None].to(rgb.device)
            result[index, ..., :3] = torch.lerp(rgb, torch.full_like(rgb, .5), (1 - retained) * strength)
    return result


class H3StudioApplyMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('H3_STUDIO_SOURCE',), 'settings': ('H3_STUDIO_MASK_SETTINGS',)},
                'optional': {key + '_' + kind: ('MASK',) for key in ('reference', 'last', 'frames')
                             for kind in ('target', 'preserve')}}

    RETURN_TYPES = ('H3_STUDIO_GUIDES',)
    FUNCTION = 'apply'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def apply(self, source, settings, **masks):
        guides = dict(source)
        guides.update(mask_settings=settings, target_masks={}, preserve_masks={}, effective_masks={},
                      target_overlays={}, preserve_overlays={}, input_previews={})
        for key in ('reference', 'last', 'frames'):
            guides['prepared_' + key] = source[key]
        for key in mask_keys(source):
            original = source[key]
            analysis = analysis_images(original)
            target = selected_mask(analysis, settings, masks.get(key + '_target'))
            preserve = masks.get(key + '_preserve')
            if preserve is None:
                if settings['preserve_text'].strip():
                    raise ValueError('The requested SAM3 preserve mask was not computed.')
                preserve = torch.zeros_like(target)
            if tuple(preserve.shape) != tuple(target.shape):
                raise ValueError('SAM3 preserve and target must use the same source analysis dimensions.')
            preserve = preserve.detach().to(device='cpu', dtype=torch.float32).clamp(0, 1)
            effective = target * (1 - preserve) * float(settings['strength'])
            guides['target_masks'][key] = target
            guides['preserve_masks'][key] = preserve
            guides['effective_masks'][key] = effective
            guides['prepared_' + key] = apply_treatment(original, target, preserve, settings['treatment'], float(settings['strength']))
            guides['target_overlays'][key] = overlay(analysis, target * (1 - preserve), [0.1, 1., .3])
            guides['preserve_overlays'][key] = overlay(analysis, preserve, [.15, .5, 1.])
            if settings['treatment'] != 'Original':
                guides['input_previews'][key] = preview_images(guides['prepared_' + key])
        return (guides,)


class H3StudioMaskPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'guides': ('H3_STUDIO_GUIDES',)}}

    RETURN_TYPES = ()
    FUNCTION = 'preview'
    OUTPUT_NODE = True
    CATEGORY = 'MiniMax H3 Studio'

    @staticmethod
    def _save(images, fps, video, label):
        digest = hashlib.sha256(('h3-studio-preview-v1:' + str(fps) + ':' + label).encode())
        digest.update(str(tuple(images.shape)).encode())
        for frame in images:
            digest.update(frame.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy().tobytes())
        extension = '.mp4' if video else '.png'
        root = Path(folder_paths.get_temp_directory()) / 'h3_studio_previews'
        root.mkdir(parents=True, exist_ok=True)
        path = root / (digest.hexdigest() + extension)
        if not path.is_file() or path.stat().st_size == 0:
            temporary = root / (path.stem + '.' + uuid.uuid4().hex + '.pending' + extension)
            try:
                if video:
                    from comfy_extras.nodes_video import CreateVideo
                    clip = CreateVideo.execute(images=images, fps=float(fps)).result[0]
                    clip.save_to(str(temporary), format='mp4', codec='h264', crf=23)
                else:
                    array = (images[0].detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(array).save(temporary, format='PNG')
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return {'filename': path.name, 'subfolder': 'h3_studio_previews', 'type': 'temp',
                'label': label, 'mime': 'video/mp4' if video else 'image/png'}

    def preview(self, guides):
        items = []
        video = guides['kind'] == 'video'
        names = {'reference': 'First frame' if guides['last'] is not None else 'Reference',
                 'last': 'Last frame', 'frames': 'Source video'}
        for key in mask_keys(guides):
            items.append(self._save(guides['target_overlays'][key], guides['fps'], video,
                                   names[key] + ' · edit selection in green'))
            items.append(self._save(guides['preserve_overlays'][key], guides['fps'], video,
                                   names[key] + ' · preserve in blue'))
            if key in guides['input_previews']:
                items.append(self._save(guides['input_previews'][key], guides['fps'], video,
                                       names[key] + ' · model input'))
        return {'ui': {'h3_studio_previews': items}, 'result': ()}


MEDIA_NODES = {cls.__name__: cls for cls in (H3StudioMedia, H3StudioSource, H3StudioAnalysisImages,
    H3StudioMasks, H3StudioApplyMasks, H3StudioMaskPreview)}
