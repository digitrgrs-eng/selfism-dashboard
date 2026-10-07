"""Original non-destructive file-backed trim implementation."""
import math
import av
from comfy_api.latest._input_impl.video_types import VideoFromFile

def _has_duration_metadata(video):
    # Native get_duration otherwise falls back to scanning/decoding a whole file.
    with av.open(video.get_stream_source(), mode='r') as container:
        if container.duration is not None:
            return
        stream = next((s for s in container.streams if s.type == 'video'), None)
        if stream is not None and stream.frames and stream.average_rate:
            return
    raise ValueError('Video duration is missing from its metadata. Save the clip as MP4 before trimming.')

class H3TrimVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('VIDEO',),
            'start_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0,
                'max': 1e6, 'step': .01, 'tooltip': 'Where the selected clip starts, in seconds.'}),
            'end_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0,
                'max': 1e6, 'step': .01, 'tooltip': 'Where the selected clip ends. Zero keeps everything to the end.'}),
        }}

    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('VIDEO',)
    FUNCTION = 'trim'
    CATEGORY = 'MiniMax Safe'

    def trim(self, video, start_seconds=0.0, end_seconds=0.0):
        start, end = float(start_seconds), float(end_seconds)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
            raise ValueError('Trim start and end must be finite, nonnegative seconds')
        if end and end <= start:
            raise ValueError('Trim end must be later than trim start (or zero for the end of the video)')
        if start == 0 and end == 0:
            return (video,)
        if not isinstance(video, VideoFromFile):
            raise ValueError('Connect Load Video to Trim Video; generated or combined video inputs are not supported')
        _has_duration_metadata(video)
        duration = float(video.get_duration())
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('The selected video has no usable duration')
        if start >= duration:
            raise ValueError(f'Trim start must be before the video ends ({duration:.3f} seconds)')
        end = min(end, duration) if end else duration
        if start == 0 and end == duration:
            return (video,)

        # Explicit duration keeps an existing trimmed input from expanding again.
        trimmed = video.as_trimmed(start_time=start, duration=end - start, strict_duration=False)
        if trimmed is None:
            raise ValueError('The selected video trim could not be applied')
        expected_start = video.get_active_trim_window()[0] + start
        actual_start, actual_duration = trimmed.get_active_trim_window()
        if not math.isclose(actual_start, expected_start, abs_tol=1e-8) or not math.isclose(actual_duration, end - start, abs_tol=1e-8):
            raise ValueError('This existing trim cannot be extended safely. Connect an untrimmed Load Video input.')
        return (trimmed,)
