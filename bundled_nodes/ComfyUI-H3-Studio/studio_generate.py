"""Mode-specific native MiniMax H3 generation for the compact Studio graph."""
import math
from fractions import Fraction

import torch
import torch.nn.functional as F

from comfy_execution.graph_utils import GraphBuilder
from .stability import ensure_stable_loading
from .attention_safety import H3StudioAttentionSafety
from .studio_loras import parse_lora_rows, resolve_enabled_loras, append_lora_triggers
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, align_frame_count
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest import Types


FL_MODEL = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'
REF_MODEL = 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
FL_TURBO = 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'
REF_TURBO = 'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'
VIDEO_VAE = 'minimax_h3_video_vae_int8_convrot.safetensors'
AUDIO_VAE = 'minimax_h3_audio_vae_fp32.safetensors'
TEXT_ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'


def canvas_size(image, megapixels, aspect_ratio):
    height, width = image.shape[1:3]
    ratio = width / height if aspect_ratio == 'Match input' else {
        '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1, '4:3': 4 / 3, '3:4': 3 / 4,
    }[aspect_ratio]
    area = float(megapixels) * 1_000_000
    return max(32, round(math.sqrt(area * ratio) / 32) * 32), max(32, round(math.sqrt(area / ratio) / 32) * 32)


def fit_images(images, width, height, fit):
    """Use the same geometry for both endpoint images, without stretching either."""
    old_h, old_w = images.shape[1:3]
    if (old_w, old_h) == (width, height):
        return images[..., :3]
    scale = max(width / old_w, height / old_h) if fit == 'Crop to fill' else min(width / old_w, height / old_h)
    new_w, new_h = max(1, round(old_w * scale)), max(1, round(old_h * scale))
    output = torch.empty((len(images), height, width, 3), dtype=images.dtype, device=images.device)
    for start in range(0, len(images), 8):
        batch = F.interpolate(images[start:start + 8, ..., :3].movedim(-1, 1), size=(new_h, new_w),
                              mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        if fit == 'Crop to fill':
            left, top = (new_w - width) // 2, (new_h - height) // 2
            batch = batch[:, :, top:top + height, left:left + width]
        else:
            left, top = (width - new_w) // 2, (height - new_h) // 2
            batch = F.pad(batch, (left, width - new_w - left, top, height - new_h - top), mode='replicate')
        output[start:start + len(batch)] = batch.movedim(1, -1)
    return output


def generation_prompt(guides, prompt, include_original_video, reference_description=''):
    mode = guides['mode']
    if mode == 'Reference to video':
        heading = '[reference generation] Use <Picture 1> as the visual reference for the subject and its appearance. Generate the scene and motion described below.'
    elif mode == 'First + last frame':
        heading = 'Begin with <Picture 1> and end with <Picture 2>. Generate a coherent continuous transition and the motion described below between these endpoint frames.'
    else:
        heading = '[video editing] Edit <Video 1> according to the instructions below. Preserve the source timing, actions, expressions, camera movement and all unrelated scene details.'
        if guides.get('reference') is not None:
            heading += ' <Picture 1> is an optional appearance reference; apply only the appearance changes requested below.'
        if guides['mask_settings'].get('treatment') == 'Invert selected colors':
            heading += ' The altered colors in the selected source area identify where to regenerate; render natural colors there, not photographic-negative colors.'
        if include_original_video:
            heading += ' <Video 2> is the unmodified source, provided for motion and context; apply the requested edit rather than copying its old appearance.'
    settings = guides.get('mask_settings', {})
    selected = settings.get('sam3_target', '').strip()
    preserve = settings.get('preserve_text', '').strip()
    if settings.get('mask_mode') == 'SAM3' and selected:
        heading += f' Focus the requested changes on: {selected}.'
    if preserve:
        heading += f' Preserve the appearance and coherent motion of: {preserve}.'
    text = heading + '\n\n' + prompt.strip()
    if guides.get('reference') is not None and reference_description.strip():
        text += ('\n\nReference character:\n'
                 '<Picture 1> defines the character\'s identity and visible appearance. '
                 'Use the following appearance details with that image while following the scene and action instructions above:\n'
                 + reference_description.strip())
    return text


class H3StudioGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'guides': ('H3_STUDIO_GUIDES',),
            'prompt': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'Describe the desired action, camera and appearance. For video inpainting, say precisely what changes, such as Make the shirt deep red while preserving its folds and lighting.'}),
            'target_megapixels': ('FLOAT', {'default': .8, 'min': .05, 'max': 2., 'step': .05, 'tooltip': 'Output detail and memory cost. 0.8 is the normal default; 0.2 is useful for a quick test.'}),
            'duration_seconds': ('FLOAT', {'default': 5., 'min': .2, 'max': 60., 'step': .1, 'tooltip': 'New-video duration, rounded up to the H3 frame grid. Video inpainting uses the selected source trim instead. The model is trained around 4–15 seconds.'}),
            'aspect_ratio': (['Match input', '16:9', '9:16', '1:1', '4:3', '3:4'], {'default': 'Match input', 'tooltip': 'Match input preserves the reference or source aspect ratio. Other choices use the image-fit setting.'}),
            'fit': (['Crop to fill', 'Fit with edge padding'], {'default': 'Crop to fill', 'tooltip': 'Both endpoint images use identical aspect-preserving fitting. Crop fills the frame; padding retains the whole image.'}),
            'noise_seed': ('INT', {'default': 314159, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True, 'tooltip': 'Fixed repeats the same settings; Randomize explores a new result.'}),
            'steps': ('INT', {'default': 8, 'min': 1, 'max': 50, 'tooltip': 'The included mode-specific Turbo LoRAs are trained for 8 steps. More steps are not automatically better.'}),
            'include_original_video': ('BOOLEAN', {'default': False, 'tooltip': 'Video inpainting only: also provide the unmodified video as a second reference. It may retain more source appearance.'}),
        }, 'optional': {
            'reference_description': ('STRING', {'multiline': True, 'default': '',
                'tooltip': 'Character appearance saved with the reference image. Kept separate from scene and action instructions; ignored when no reference image is supplied.'}),
            'loras_json': ('STRING', {'default': '', 'multiline': True, 'tooltip': 'Ordered LoRAs from the green settings panel. Enabled rows apply once and append their trigger words. An empty list disables all LoRAs; older graphs without this setting keep their matching Turbo.'}),
        }}

    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('video',)
    FUNCTION = 'generate'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def generate(self, guides, prompt, target_megapixels, duration_seconds, aspect_ratio, fit, noise_seed, steps, include_original_video, reference_description='', loras_json=''):
        ensure_stable_loading()
        if loras_json.strip():
            lora_rows = parse_lora_rows(loras_json)
            resolve_enabled_loras(lora_rows)
        first_last = guides['mode'] == 'First + last frame'
        graph = GraphBuilder()
        model = graph.node('UNETLoader', unet_name=FL_MODEL if first_last else REF_MODEL, weight_dtype='default')
        if loras_json.strip():
            stack = graph.node('H3StudioLoraStack', model=model.out(0), loras_json=loras_json, prompt=prompt)
            selected_model, selected_prompt = stack.out(0), append_lora_triggers(prompt, lora_rows)
        else:
            # Backwards-compatible loading for existing graphs without the new control.
            turbo = graph.node('LoraLoaderModelOnly', model=model.out(0), lora_name=FL_TURBO if first_last else REF_TURBO, strength_model=1.)
            selected_model, selected_prompt = turbo.out(0), prompt
        shifted = graph.node('MiniMaxH3SigmaShift', model=selected_model, shift_video=6., shift_audio=3.)
        attention_safe = graph.node('H3StudioAttentionSafety', model=shifted.out(0), force_pytorch=True)
        memory = graph.node('H3SafeMemory', model=attention_safe.out(0), chunks=4)
        clip = graph.node('CLIPLoader', clip_name=TEXT_ENCODER, type='minimax', device='default')
        vae = graph.node('VAELoader', vae_name=VIDEO_VAE)
        condition = graph.node('H3StudioCondition', guides=guides, clip=clip.out(0), vae=vae.out(0), prompt=selected_prompt,
                               target_megapixels=target_megapixels, duration_seconds=duration_seconds,
                               aspect_ratio=aspect_ratio, fit=fit, include_original_video=include_original_video,
                               reference_description=reference_description)
        noise = graph.node('RandomNoise', noise_seed=noise_seed)
        sampler = graph.node('KSamplerSelect', sampler_name='res_multistep')
        schedule = graph.node('BasicScheduler', model=memory.out(0), scheduler='simple', steps=steps, denoise=1.)
        guider = graph.node('BasicGuider', model=memory.out(0), conditioning=condition.out(0))
        sampled = graph.node('SamplerCustomAdvanced', noise=noise.out(0), guider=guider.out(0), sampler=sampler.out(0),
                             sigmas=schedule.out(0), latent_image=condition.out(1))
        decoded = graph.node('VAEDecodeTiled', samples=sampled.out(1), vae=vae.out(0),
                             tile_size=512, overlap=64, temporal_size=32, temporal_overlap=8)
        finish = graph.node('H3StudioFinish', images=decoded.out(0), guides=guides)
        if guides['mode'] != 'Video inpainting':
            audio_vae = graph.node('VAELoader', vae_name=AUDIO_VAE)
            audio = graph.node('VAEDecodeAudio', samples=sampled.out(1), vae=audio_vae.out(0))
            finish.set_input('generated_audio', audio.out(0))
        return {'result': (finish.out(0),), 'expand': graph.finalize()}


class H3StudioCondition:
    @classmethod
    def INPUT_TYPES(cls):
        generation_schema = H3StudioGenerate.INPUT_TYPES()
        schema = generation_schema['required']
        return {'required': {name: value for name, value in schema.items() if name not in ('noise_seed', 'steps')}
                | {'clip': ('CLIP',), 'vae': ('VAE',)}, 'optional': {k:v for k,v in generation_schema['optional'].items() if k != 'loras_json'}}

    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    FUNCTION = 'condition'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def condition(self, guides, clip, vae, prompt, target_megapixels, duration_seconds, aspect_ratio, fit, include_original_video, reference_description=''):
        video_mode = guides['mode'] == 'Video inpainting'
        basis = guides['frames'] if video_mode else guides['reference']
        width, height = canvas_size(basis, target_megapixels, aspect_ratio)
        length = align_frame_count(max(5, len(basis) if video_mode else round(duration_seconds * 24)))
        text = generation_prompt(guides, prompt, include_original_video, reference_description)
        if guides['mode'] == 'First + last frame':
            first = fit_images(guides['prepared_reference'], width, height, fit)
            last = fit_images(guides['prepared_last'], width, height, fit)
            return tuple(MiniMaxH3ImageToVideo.execute(clip, vae, text, width, height, length, first, last).result)
        references = {}
        if guides.get('prepared_reference') is not None:
            references['ref_image_1'] = guides['prepared_reference']
        videos = {}
        if video_mode:
            frames = fit_images(guides['prepared_frames'], width, height, fit)
            if len(frames) < length:
                frames = torch.cat((frames, frames[-1:].expand(length - len(frames), -1, -1, -1)), dim=0)
            videos['ref_video_1'] = frames
            if include_original_video:
                original = fit_images(guides['frames'], width, height, fit)
                if len(original) < length:
                    original = torch.cat((original, original[-1:].expand(length - len(original), -1, -1, -1)), dim=0)
                videos['ref_video_2'] = original
        return tuple(MiniMaxH3ReferenceToVideo.execute(clip, text, width, height, length, ref_image_size='match',
                                                     vae=vae, ref_images=references, ref_videos=videos).result)


class H3StudioFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'guides': ('H3_STUDIO_GUIDES',)},
                'optional': {'generated_audio': ('AUDIO',)}}

    RETURN_TYPES = ('VIDEO',)
    FUNCTION = 'finish'
    CATEGORY = 'MiniMax H3 Studio/internal'

    def finish(self, images, guides, generated_audio=None):
        if guides['mode'] == 'Video inpainting':
            images = images[:len(guides['frames'])]
            audio = guides.get('audio')
        else:
            audio = generated_audio
        if audio is not None:
            audio = dict(audio, waveform=audio['waveform'][..., :round(len(images) / 24 * audio['sample_rate'])])
        return (VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(24, 1))),)


GENERATE_NODES = {cls.__name__: cls for cls in (H3StudioGenerate, H3StudioCondition, H3StudioFinish, H3StudioAttentionSafety)}
