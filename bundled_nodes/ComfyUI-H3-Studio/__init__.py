"""Four-node MiniMax H3 Studio: reference video, endpoints, and video editing."""
from .studio_media import MEDIA_NODES
from .studio_generate import GENERATE_NODES
from .studio_loras import LORA_CONTROL_NODES
from .lora_upload import register_lora_upload_routes
from .model_downloads import DOWNLOAD_NODES, register_model_download_routes

NODE_CLASS_MAPPINGS = {**MEDIA_NODES, **GENERATE_NODES, **DOWNLOAD_NODES, **LORA_CONTROL_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3StudioMedia': 'MiniMax Studio · Media',
    'H3StudioSource': 'MiniMax Studio · Selected Media',
    'H3StudioMasks': 'MiniMax Studio · SAM3 Guidance',
    'H3StudioGenerate': 'MiniMax Studio · Generate',
    'H3StudioMaskPreview': 'MiniMax Studio · Mask Preview',
    'H3StudioModelDownloads': 'MiniMax Studio · Models and Downloads',
}
WEB_DIRECTORY = './web'
register_model_download_routes()

register_lora_upload_routes()

# Install before any queued graph can cache weights from the old reader.
# Requirements errors remain visible again as actionable node errors at Run.
from .stability import ensure_stable_loading
try:
    ensure_stable_loading()
except RuntimeError as error:
    import logging
    logging.warning("[H3 Studio loading preflight] %s", error)