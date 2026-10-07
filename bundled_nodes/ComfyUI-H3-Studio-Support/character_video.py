"""Source video loader used by the compact Studio media node."""
import os
from pathlib import Path

class H3CharacterVideoInput:
    """File-backed video plus non-destructive trim; pixels decode downstream."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root = Path(folder_paths.get_input_directory())
        extensions = {'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.mpeg', '.mpg'}
        files = sorted(str(path.relative_to(root)).replace('\\', '/')
                       for path in root.rglob('*') if path.is_file() and path.suffix.lower() in extensions)
        return {'required': {
            'file': (files, {'tooltip': 'Upload or choose your source video. Trim video below the preview chooses the section to generate.'}),
            'start_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1e6, 'step': .01, 'tooltip': 'First second of the selected clip.'}),
            'end_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1e6, 'step': .01, 'tooltip': 'Last second of the clip. Zero means the end of the video.'}),
        }}

    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('video',)
    FUNCTION = 'load'
    CATEGORY = 'MiniMax Safe/Character'

    @classmethod
    def VALIDATE_INPUTS(cls, file, start_seconds=0.0, end_seconds=0.0):
        import folder_paths
        return True if folder_paths.exists_annotated_filepath(file) else f'Video file is missing: {file}'

    @classmethod
    def IS_CHANGED(cls, file, start_seconds=0.0, end_seconds=0.0):
        import folder_paths
        path = folder_paths.get_annotated_filepath(file)
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_size, start_seconds, end_seconds)

    def load(self, file, start_seconds=0.0, end_seconds=0.0):
        import folder_paths
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        from .workflow_controls import H3TrimVideo
        source = VideoFromFile(folder_paths.get_annotated_filepath(file))
        return H3TrimVideo().trim(source, start_seconds, end_seconds)
