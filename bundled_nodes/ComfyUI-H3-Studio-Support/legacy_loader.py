"""Memory-safe file loading for the original H3SafePrepare IMAGE/AUDIO API."""
from pathlib import Path
import numpy as np
import torch
from .streaming import H3StreamPrepare,iter_rgb_frames,run,FF,log

def prepare_file(video,target_megapixels,max_seconds):
    # Decode/resample/resize on disk before allocating the final small tensor.
    source,audio_source=H3StreamPrepare().prepare(video,target_megapixels,max_seconds)
    directory=Path(source['directory'])
    try:
        length=source['length']
        frames=torch.empty((length,source['height'],source['width'],3),dtype=torch.float32)
        batches=iter_rgb_frames(source,batch_size=8)
        start=0
        try:
            for batch in batches:
                end=start+len(batch)
                if end>length:
                    raise RuntimeError(f'Video decoder produced more than the expected {length} frames')
                frames[start:end]=batch
                start=end
            if start!=length:
                raise RuntimeError(f'Video decoder returned {start} frames; expected {length}')
        finally:
            batches.close()
        from .streaming import PROBE
        import json
        info=json.loads(run([PROBE,'-v','error','-select_streams','a:0','-show_entries','stream=sample_rate,channels','-of','json',audio_source['path']]))
        streams=info.get('streams',[])
        if streams:
            rate=int(streams[0]['sample_rate']);channels=int(streams[0]['channels'])
            raw=run([FF,'-v','error','-ss',str(audio_source['start']),'-i',audio_source['path'],
                '-t',str(length/24),'-map','0:a:0','-vn','-f','f32le','-acodec','pcm_f32le',
                '-ar',str(rate),'-ac',str(channels),'pipe:1'])
            samples=np.frombuffer(raw,dtype=np.float32).reshape(-1,channels).T.copy()
            audio={'waveform':torch.from_numpy(samples).unsqueeze(0),'sample_rate':rate}
        else:audio={'waveform':torch.zeros(1,2,round(length/24*44100)),'sample_rate':44100}
        log.info('[H3 Safe] Original workflow loader fixed: %s frames, %sx%s; full-resolution frames never stacked.',length,source['width'],source['height'])
        return frames,audio
    finally:
        # Delete only the two staging files created by this loader, never checkpoints from other jobs.
        for name in ['source.mkv','recovery.json']:(directory/name).unlink(missing_ok=True)
        directory.rmdir()
