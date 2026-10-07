H3 Studio portable support

Copy this folder into ComfyUI/custom_nodes next to ComfyUI-H3-Studio.
Update both packs from the same release.
Install requirements.txt using ComfyUI's Python, then restart ComfyUI.
Use a current ComfyUI build with native MiniMax H3, SAM3.1 and subgraphs.
The INT8 ConvRot models require a compatible PyTorch/CUDA build; the publisher
recommends PyTorch with CUDA 13.0. Keep ComfyUI's own requirements current.

Video loading needs FFmpeg AND ffprobe available on PATH. FFmpeg.org lists
platform packages: https://ffmpeg.org/download.html
The imageio-ffmpeg fallback supplies FFmpeg only, not ffprobe.
The support pack never downloads tools or models automatically.

No QwenVL captioner or external SAM3 plugin is needed. The workflow uses the
native SAM3.1 checkpoint and the H3 conditioning encoder listed in its downloader.
The shared nodes retain the original memory, trim, masking and cache code.
The Studio video loader reads prepared 24 fps frames sequentially in batches of
eight with one FFmpeg process, exact frame-count checks, and cancellation cleanup.
The existing final IMAGE tensor still contains the whole prepared clip. Audio
keeps the selected trim and prepared-video duration. Tool lookup and temporary
file locations are portable.

Saved characters use ComfyUI/user/default/h3_character_presets and durable images
under ComfyUI/input/Character_Presets. No personal library is included here.
The internal H3StudioPrepare node always uses this pack's updated decoder, even
when an older ComfyUI-MiniMax-Safe is present. Shared masks, memory settings and
character-library routes defer to that larger pack to avoid duplicate names and
routes. Do not install a partial or unrelated folder under that name.
