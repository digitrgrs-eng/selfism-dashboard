"""Minimal portable dependencies for the MiniMax H3 Studio share package."""
import logging
from pathlib import Path
import folder_paths
from .safe import BiggerFishyH3StudioVideoPrepare

NODE_CLASS_MAPPINGS = {'BiggerFishyH3StudioVideoPrepare': BiggerFishyH3StudioVideoPrepare}

# The author's larger local pack owns these names and the same preset routes.
# Keep it authoritative when both packages have been installed.
_legacy_installed = any((Path(root) / 'ComfyUI-MiniMax-Safe' / '__init__.py').is_file()
                        for root in folder_paths.get_folder_paths('custom_nodes'))
if _legacy_installed:
    logging.info('[H3 Studio Support] Using the Studio decoder and existing MiniMax-Safe masks, memory settings, and character routes.')
else:
    from .safe import H3SafeMemory, H3SafePrepare
    from .character_video import H3CharacterVideoInput
    from .character_mask_frames import H3CharacterSAMCheckpoint
    from .mask_controls import H3CharacterCachedSAM3, H3CharacterPreserveCleanup
    from .character_presets import register_character_preset_routes

    NODE_CLASS_MAPPINGS.update({node.__name__: node for node in (
        H3SafeMemory, H3SafePrepare, H3CharacterVideoInput,
        H3CharacterSAMCheckpoint, H3CharacterCachedSAM3, H3CharacterPreserveCleanup,
    )})
    register_character_preset_routes()
