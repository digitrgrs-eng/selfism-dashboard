"""Local character descriptions with durable copies of their reference images."""
import asyncio
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import threading
import unicodedata
from urllib.parse import urlsplit
import uuid

from aiohttp import web
import folder_paths
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 100 * 1024 ** 2
MAX_REQUEST_BYTES = 256 * 1024
MAX_PROMPT_CHARACTERS = 20000
IMAGE_EXTENSIONS = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif',
                    'BMP': '.bmp', 'TIFF': '.tiff', 'ICO': '.ico', 'AVIF': '.avif'}
_save_lock = threading.Lock()


class PresetError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _same_origin(request):
    if request.headers.get('Sec-Fetch-Site', '').lower() == 'cross-site':
        return False
    origin = request.headers.get('Origin')
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (parsed.scheme.lower() == request.scheme.lower()
            and parsed.netloc.lower() == request.host.lower()
            and parsed.path in ('', '/') and not parsed.query and not parsed.fragment)


def _preset_directory():
    root = Path(folder_paths.get_user_directory()).resolve()
    directory = (root / 'default' / 'h3_character_presets').resolve()
    if not directory.is_relative_to(root):
        raise PresetError('The character preset folder must remain inside the ComfyUI user folder.', 500)
    return directory


def _image_directory():
    root = Path(folder_paths.get_input_directory()).resolve()
    directory = (root / 'Character_Presets').resolve()
    if not directory.is_relative_to(root):
        raise PresetError('The character image folder must remain inside the ComfyUI input folder.', 500)
    return directory


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{32}', value):
        raise PresetError('Choose a valid saved character before updating it.')
    return value


def _name(value):
    if not isinstance(value, str):
        raise PresetError('Enter a character name.')
    value = value.strip()
    if not 1 <= len(value) <= 100 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise PresetError('Use a character name between 1 and 100 characters, on one line.')
    return value


def _name_key(value):
    return unicodedata.normalize('NFKC', value).casefold()


def _prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise PresetError('Enter the character description to save with this reference image.')
    if len(value) > MAX_PROMPT_CHARACTERS or '\x00' in value:
        raise PresetError(f'Keep the character description below {MAX_PROMPT_CHARACTERS:,} characters without null characters.')
    return value


def _source_image(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(char) < 32 for char in value):
        raise PresetError('Choose a reference image from the ComfyUI input folder.')
    name, annotated_root = folder_paths.annotated_filepath(value)
    root = Path(folder_paths.get_input_directory()).resolve()
    if annotated_root is not None and Path(annotated_root).resolve() != root:
        raise PresetError('Character references must come from the ComfyUI input folder, not output or temporary files.')
    name = name.replace('\\', '/')
    if (name.startswith('/') or PureWindowsPath(name).drive or ':' in name
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise PresetError('Use an input image filename, not an absolute path or parent-folder path.')
    try:
        path = Path(folder_paths.get_annotated_filepath(value)).resolve()
    except ValueError as error:
        raise PresetError('The reference image path is invalid.') from error
    if not path.is_relative_to(root):
        raise PresetError('The reference image must remain inside the ComfyUI input folder.')
    if not path.is_file():
        raise PresetError('The reference image is missing. Upload it again before saving this character.')
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise PresetError('Character reference images must be 100 MiB or smaller.', 413)
    return path


def _read_presets():
    directory = _preset_directory()
    presets = []
    if not directory.exists():
        return presets
    for path in sorted(directory.glob('*.json')):
        if not path.resolve().is_relative_to(directory):
            raise PresetError('A character preset points outside its saved folder.', 500)
        try:
            if path.stat().st_size > MAX_REQUEST_BYTES:
                raise ValueError('oversized record')
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict) or set(record) != {'id', 'name', 'prompt', 'image'}:
                raise ValueError('invalid record')
            if _id(record['id']) != path.stem:
                raise ValueError('preset ID differs from filename')
            _name(record['name'])
            _prompt(record['prompt'])
            image = record['image']
            if not isinstance(image, str) or not re.fullmatch(r'Character_Presets/[0-9a-f]{32}\.(jpg|png|webp|gif|bmp|tiff|ico|avif)', image):
                raise ValueError('invalid saved image filename')
        except (ValueError, TypeError, KeyError) as error:
            raise PresetError('A saved character record is damaged. Your saved files have been left unchanged.', 500) from error
        presets.append(record)
    return sorted(presets, key=lambda p: (_name_key(p['name']), p['id']))


def _copy_image(source):
    directory = _image_directory()
    directory.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=directory, prefix='.', suffix='.pending', delete=False) as output:
            pending = Path(output.name)
            with source.open('rb') as original:
                total = 0
                while chunk := original.read(1024 ** 2):
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise PresetError('Character reference images must be 100 MiB or smaller.', 413)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        try:
            with Image.open(pending) as image:
                extension = IMAGE_EXTENSIONS.get(image.format)
                if not extension:
                    raise PresetError('Use a PNG, JPEG, WebP, GIF, BMP, TIFF, ICO, or AVIF reference image.')
                image.verify()
        except (UnidentifiedImageError, SyntaxError, OSError, Image.DecompressionBombError) as error:
            raise PresetError('The reference file is not a valid supported image. Upload a different image.') from error
        target = directory / (uuid.uuid4().hex + extension)
        # Publish without replacing any original or prior saved reference.
        os.link(pending, target)
        return 'Character_Presets/' + target.name
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _write_record(record):
    directory = _preset_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (record['id'] + '.json')
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, prefix='.', suffix='.pending', delete=False) as output:
            pending = Path(output.name)
            json.dump(record, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def save_preset(payload):
    if not isinstance(payload, dict):
        raise PresetError('Send a character name, description, and reference image.')
    name, prompt = _name(payload.get('name')), _prompt(payload.get('prompt'))
    ident = _id(payload['id']) if 'id' in payload else None
    source = _source_image(payload.get('image'))
    with _save_lock:
        presets = _read_presets()
        previous = next((p for p in presets if p['id'] == ident), None)
        if ident is not None and previous is None:
            raise PresetError('This character no longer exists. Save it as a new character instead.', 404)
        if any(p['id'] != ident and _name_key(p['name']) == _name_key(name) for p in presets):
            raise PresetError('A character with this name already exists. Choose another name or update that character.', 409)
        saved_image = _copy_image(source)
        record = {'id': ident or uuid.uuid4().hex, 'name': name, 'prompt': prompt, 'image': saved_image}
        _write_record(record)
        return record


async def list_character_presets(request):
    try:
        presets = await asyncio.to_thread(_read_presets)
        return web.json_response({'presets': presets})
    except PresetError as error:
        return web.json_response({'error': str(error)}, status=error.status)
    except OSError:
        return web.json_response({'error': 'Could not read saved characters. Check the preset folder access.'}, status=500)


async def save_character_preset(request):
    if not _same_origin(request):
        return web.json_response({'error': 'Save characters from this ComfyUI page.'}, status=403)
    if request.content_type != 'application/json':
        return web.json_response({'error': 'Send the character settings as JSON.'}, status=400)
    try:
        if request.content_length and request.content_length > MAX_REQUEST_BYTES:
            raise PresetError('The character settings are too large.', 413)
        chunks, size = [], 0
        async for chunk in request.content.iter_chunked(64 * 1024):
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise PresetError('The character settings are too large.', 413)
            chunks.append(chunk)
        payload = json.loads(b''.join(chunks))
        preset = await asyncio.to_thread(save_preset, payload)
        return web.json_response({'preset': preset}, status=200 if 'id' in payload else 201)
    except PresetError as error:
        return web.json_response({'error': str(error)}, status=error.status)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return web.json_response({'error': 'The character settings contain invalid JSON.'}, status=400)
    except OSError:
        return web.json_response({'error': 'Could not save this character. Check available disk space and folder access.'}, status=500)


def register_character_preset_routes():
    from server import PromptServer
    PromptServer.instance.routes.get('/minimax-safe/character-presets')(list_character_presets)
    PromptServer.instance.routes.post('/minimax-safe/character-presets')(save_character_preset)
