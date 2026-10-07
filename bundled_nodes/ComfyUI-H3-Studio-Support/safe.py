"""Original H3 Studio memory patch and bounded video preparation."""
import logging
import math
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

class H3SafeMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'chunks': ('INT', {'default': 4, 'min': 1, 'max': 16})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'MiniMax Safe'

    def patch(self, model, chunks):
        if chunks == 1:
            return (model,)
        patched = model.clone()
        diffusion = model.get_model_object('diffusion_model')
        paths = [f'diffusion_model.blocks.{i}.mlp.forward' for i in range(len(diffusion.blocks))]
        paths += [f'diffusion_model.token_refiner.blocks.{i}.mlp.forward' for i in range(len(diffusion.token_refiner.blocks))]
        def wrap(original):
            def forward(x):
                if x.shape[0] < 4096:
                    return original(x)
                output = torch.empty_like(x)
                offset = 0
                for part in x.chunk(chunks, dim=0):
                    end = offset + part.shape[0]
                    output[offset:end].copy_(original(part))
                    offset = end
                return output
            return forward
        for path in paths:
            patched.add_object_patch(path, wrap(patched.get_model_object(path)))
        log.info('[H3 Safe] Feed-forward chunking enabled: %s chunks, %s layers', chunks, len(paths))
        return (patched,)

class H3SafePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video': ('VIDEO',), 'target_megapixels': ('FLOAT', {'default': .8, 'min': .05, 'max': 2, 'step': .05}), 'max_seconds': ('FLOAT', {'default': 0, 'min': 0, 'max': 600, 'step': 1})}}
    RETURN_TYPES = ('IMAGE', 'AUDIO')
    RETURN_NAMES = ('frames_24fps', 'original_audio')
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax Safe'

    def prepare(self, video, target_megapixels, max_seconds):
        if hasattr(video, 'get_stream_source') and isinstance(video.get_stream_source(), str):
            from .legacy_loader import prepare_file
            return prepare_file(video, target_megapixels, max_seconds)
        comp = video.get_components()
        source = comp.images
        fps = float(comp.frame_rate)
        length = max(1, round(len(source) / fps * 24))
        if max_seconds > 0:
            length = min(length, max(1, round(max_seconds * 24)))
        h, w = source.shape[1:3]
        scale = math.sqrt(target_megapixels * 1000000 / (h*w))
        width, height = max(32, round(w*scale/32)*32), max(32, round(h*scale/32)*32)
        frames = torch.empty((length, height, width, 3), dtype=source.dtype, device='cpu')
        for start in range(0, length, 24):
            indexes = [min(len(source)-1, round(i*fps/24)) for i in range(start,min(length,start+24))]
            batch = source[indexes,:,:,:3].movedim(-1,1)
            batch = F.interpolate(batch, size=(height,width), mode='bicubic', align_corners=False, antialias=True)
            frames[start:start+len(indexes)] = batch.movedim(1,-1).clamp(0,1).cpu()
        audio = comp.audio
        if audio is None:
            audio = {'waveform': torch.zeros(1,2,round(length/24*44100)), 'sample_rate':44100}
        else:
            audio = {'waveform':audio['waveform'][...,:round(length/24*audio['sample_rate'])].clone(), 'sample_rate':audio['sample_rate']}
        log.info('[H3 Safe] Prepared %s frames at 24 fps, %sx%s; duration %.3fs', length,width,height,length/24)
        return (frames,audio)


class BiggerFishyH3StudioVideoPrepare(H3SafePrepare):
    """Studio-owned decoder, independent of older optional support packs."""

    CATEGORY = 'MiniMax H3 Studio/internal'

# Compatibility for direct Python imports only; the generic node id is not registered.
H3StudioPrepare = BiggerFishyH3StudioVideoPrepare
