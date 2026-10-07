"""Original disk-backed preparation with portable tool and temp locations."""
import hashlib
import json
import logging
import math
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import torch
import folder_paths

log = logging.getLogger(__name__)


def _resolve_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    command = [sys.executable, '-s', '-m', 'pip', 'install', '-r', str(Path(__file__).with_name('requirements.txt'))]
    repair = subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as error:
        if error.name != 'imageio_ffmpeg':
            raise
        raise RuntimeError(
            'FFmpeg is not on PATH, and imageio-ffmpeg is missing from this ComfyUI Python: '
            + sys.executable + '. Open Install-MiniMax-H3-Studio.bat, or run: '
            + repair + '. Do not use --user; restart ComfyUI after repair. '
            + 'ffprobe is also required separately; imageio-ffmpeg does not supply it.'
        ) from error
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (RuntimeError, OSError) as error:
        raise RuntimeError(
            'The imageio-ffmpeg fallback could not locate a usable FFmpeg executable. '
            + 'Install FFmpeg and ffprobe on PATH, or repair this Python environment with: '
            + repair + '. Restart ComfyUI after repair.'
        ) from error


FF = _resolve_ffmpeg()
_probe_name = 'ffprobe.exe' if Path(FF).suffix.lower() == '.exe' else 'ffprobe'
_probe_sibling = Path(FF).with_name(_probe_name)
PROBE = shutil.which('ffprobe') or (str(_probe_sibling) if _probe_sibling.is_file() else 'ffprobe')

def job_key(graph):
    # ComfyUI injects transient is_changed fingerprints during execution.
    graph={k:{'class_type':v.get('class_type'),'inputs':json.loads(json.dumps(v.get('inputs',{})))} for k,v in graph.items()}
    for node in graph.values():
        node.get('inputs',{}).pop('recovery_directory',None)
    return hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest()

def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def run(args):
    try:
        p = subprocess.run(args, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError as error:
        if Path(str(args[0])).name.lower() in ('ffprobe', 'ffprobe.exe'):
            raise RuntimeError('ffprobe is missing. Install FFmpeg AND ffprobe on PATH, then restart ComfyUI. The imageio-ffmpeg Python package supplies FFmpeg only, not ffprobe. See https://ffmpeg.org/download.html.') from error
        raise RuntimeError('The FFmpeg executable could not be started. Install FFmpeg and ffprobe on PATH, then restart ComfyUI. See https://ffmpeg.org/download.html.') from error
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return p.stdout

def probe(path):
    return json.loads(run([PROBE, '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)]))

def _check_interrupted():
    # Import lazily: inspecting this module need not initialize ComfyUI's device.
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


class _DecoderPipeReader:
    """Bounded pipe reads with cancellation even while FFmpeg produces no data."""
    def __init__(self, pipe):
        self.pipe = pipe
        self.queue = queue.Queue(maxsize=2)
        self.stopped = threading.Event()
        self.pending = b''
        self.eof = False
        self.thread = threading.Thread(target=self._pump, name='h3-video-decoder', daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                pass

    def _pump(self):
        try:
            while not self.stopped.is_set():
                chunk = self.pipe.read(65536)
                self._put((chunk, None))
                if not chunk:
                    return
        except Exception as error:
            self._put((b'', error))

    def read_exact(self, size):
        result = bytearray()
        while len(result) < size:
            _check_interrupted()
            if self.pending:
                take = min(size - len(result), len(self.pending))
                result.extend(self.pending[:take])
                self.pending = self.pending[take:]
                continue
            if self.eof:
                break
            try:
                chunk, error = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            if error is not None:
                raise RuntimeError(f'Video decoder output pipe failed: {error}') from error
            if not chunk:
                self.eof = True
            else:
                self.pending = chunk
        return result

    def stop(self):
        self.stopped.set()

    def join(self):
        # The owner terminates/reaps the process before joining this reader.
        self.thread.join(timeout=2)


def _decoder_stderr(errors):
    errors.seek(0, 2)
    errors.seek(max(0, errors.tell() - 3000))
    text = errors.read().decode(errors='replace').strip()
    return f' FFmpeg: {text}' if text else ''


def _wait_decoder(process):
    while True:
        _check_interrupted()
        try:
            return process.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            pass


def _decode_rgb_batches(source, count, batch_size, start_frame=None):
    """Decode once; frame-count metadata is a strict bound, never a padding hint."""
    w, h = source['width'], source['height']
    for name, value in (('width', w), ('height', h), ('count', count), ('batch_size', batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Video {name} must be a positive integer')
    if start_frame is not None and (isinstance(start_frame, bool)
                                   or not isinstance(start_frame, int) or start_frame < 0):
        raise ValueError('Video start frame must be a nonnegative integer')
    frame_bytes = w * h * 3
    args = [FF, '-v', 'error', '-threads', '2', '-i', str(source['path']),
            '-map', '0:v:0', '-an', '-sn', '-dn']
    if start_frame is not None:
        # Count decoded frames, rather than assuming a timestamp seek picks the
        # same frame in every container/timebase. Preserve each selected frame.
        args += ['-vf', f'trim=start_frame={start_frame}:end_frame={start_frame + count},setpts=PTS-STARTPTS',
                 '-frames:v', str(count)]
    args += ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-threads', '2', 'pipe:1']
    _check_interrupted()
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors,
                                       bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except FileNotFoundError as error:
            raise RuntimeError('Install FFmpeg on PATH, then restart ComfyUI to prepare a source video.') from error
        reader = None
        received = 0
        try:
            reader = _DecoderPipeReader(process.stdout)
            for offset in range(0, count, batch_size):
                size = min(batch_size, count - offset)
                raw = reader.read_exact(size * frame_bytes)
                if len(raw) != size * frame_bytes:
                    code = _wait_decoder(process)
                    complete, partial = divmod(len(raw), frame_bytes)
                    actual = received + complete
                    raise RuntimeError(
                        f'Video decoder returned an incomplete section: expected {count} frames, '
                        f'received {actual}; missing {count - actual}. '
                        f'Block starts at frame {(start_frame or 0) + offset}; '
                        f'{partial} trailing RGB bytes; FFmpeg exit code {code}.'
                        + _decoder_stderr(errors))
                batch = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(size, h, w, 3).copy())
                batch = batch.float().div_(255)
                received += size
                yield batch
                del batch, raw
            # Read through EOF to detect a stale/incorrect frame count. Full
            # videos are uncapped; an explicitly selected window ends at count.
            extra = reader.read_exact(frame_bytes)
            if extra:
                detail = (f'at least {count + 1} frames' if len(extra) == frame_bytes
                          else f'{count} frames and a partial extra frame ({len(extra)}/{frame_bytes} RGB bytes)')
                raise RuntimeError(f'Video decoder returned extra data: expected {count} frames, received {detail}.'
                                   + _decoder_stderr(errors))
            code = _wait_decoder(process)
            if code:
                raise RuntimeError(f'FFmpeg video decoder exited with code {code} after {received}/{count} frames.'
                                   + _decoder_stderr(errors))
        finally:
            # Covers cancellation, generator.close(), tensor conversion errors,
            # malformed output, and successful exhaustion. No orphan decoder.
            if reader is not None:
                reader.stop()
            try:
                if process.poll() is None:
                    process.kill()
                process.wait()
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                if reader is not None:
                    reader.join()


def iter_rgb_frames(source, batch_size=8):
    """Yield the complete normalized video in bounded CPU RGB tensor batches."""
    yield from _decode_rgb_batches(source, source['length'], batch_size)


def read_window(source, start, count):
    # Exhaust the decoder before returning, so late FFmpeg errors are checked.
    batches = list(_decode_rgb_batches(source, count, count, start_frame=start))
    return batches[0]

class H3StreamPrepare:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A cached directory can contain output made with a different downstream
        # seed/prompt/reference. Reevaluate to allocate a fresh job directory or
        # validate the full graph against explicitly requested recovery settings.
        return float('nan')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'video':('VIDEO',),'target_megapixels':('FLOAT',{'default':.8,'min':.05,'max':2,'step':.05}),
                            'max_seconds':('FLOAT',{'default':0,'min':0,'max':600,'step':1})},
                'optional':{'recovery_directory':('STRING',{'default':''})},'hidden':{'prompt':'PROMPT'}}
    RETURN_TYPES=('H3_DISK_SOURCE','H3_AUDIO_SOURCE')
    RETURN_NAMES=('frames_24fps','original_audio')
    FUNCTION='prepare'
    CATEGORY='MiniMax Safe'
    def prepare(self,video,target_megapixels,max_seconds,recovery_directory='',prompt=None):
        src=video.get_stream_source()
        if not isinstance(src,(str,Path)):
            raise ValueError('Low-RAM workflow requires a video file from Load Video')
        if getattr(video,'_VideoFromFile__crop',None):
            raise ValueError('Use an uncropped Load Video input for the low-RAM workflow')
        start,duration=video.get_active_trim_window()
        if max_seconds>0:duration=min(duration,max_seconds) if duration else max_seconds
        identity={'job_key':job_key(prompt or {}),'source_size':Path(src).stat().st_size,'source_mtime_ns':Path(src).stat().st_mtime_ns}
        if recovery_directory.strip():
            directory=Path(recovery_directory).resolve()
            saved=json.loads((directory/'recovery.json').read_text())
            if saved['identity']!=identity:raise ValueError('Recovery settings or source file changed; use the original inputs to resume.')
            source=saved['source'];source['directory']=str(directory);source['path']=str(directory/'source.mkv')
            if not Path(source['path']).is_file():raise ValueError('Recovery source is missing')
            log.info('[H3 Recovery] Reusing disk source and completed sections from %s',directory)
            return source,saved['audio']
        w,h=video.get_dimensions()
        scale=math.sqrt(target_megapixels*1000000/(w*h))
        w,h=max(32,round(w*scale/32)*32),max(32,round(h*scale/32)*32)
        recovery_root=Path(folder_paths.get_temp_directory())/'h3_studio_prepare'
        recovery_root.mkdir(parents=True, exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix='h3_stream_',dir=recovery_root))
        target=directory/'source.mkv'
        args=[FF,'-v','error','-y','-threads','2','-ss',str(start),'-i',str(src)]
        if duration:args+=['-t',str(duration)]
        args+=['-map','0:v:0','-an','-vf',f'fps=24,scale={w}:{h}:flags=bicubic,setsar=1',
               '-c:v','ffv1','-level','3','-threads','2',str(target)]
        run(args)
        stream=next(s for s in probe(target)['streams'] if s['codec_type']=='video')
        length=int(stream['nb_read_frames'])
        if length<1:raise ValueError('Input video has no frames')
        log.info('[H3 Stream] Disk-backed input: %s frames, %sx%s. No full-video tensor.',length,w,h)
        source={'path':str(target),'directory':str(directory),'width':w,'height':h,'length':length}
        audio={'path':str(src),'start':start,'duration':length/24}
        atomic_json(directory/'recovery.json',{'identity':identity,'source':source,'audio':audio})
        return source,audio
