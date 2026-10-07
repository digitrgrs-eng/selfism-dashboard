"""Local, streamed safetensors uploads for the workflow's custom LoRA loader."""
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
from urllib.parse import urlsplit

from aiohttp import web
import folder_paths
from safetensors import SafetensorError, safe_open


MAX_UPLOAD_BYTES = 4 * 1024 ** 3
CHUNK_BYTES = 1024 ** 2


def _filename(name):
    if not name or '/' in name or '\\' in name or ':' in name or PureWindowsPath(name).is_absolute():
        raise ValueError('Choose a .safetensors file with a plain filename, not a path')
    if name in ('.', '..') or any(ord(char) < 32 for char in name):
        raise ValueError('The file has an invalid name')
    if Path(name).suffix.lower() != '.safetensors':
        raise ValueError('Only .safetensors LoRA files can be uploaded')
    stem = re.sub(r'[<>"|?*]', '_', Path(name).stem).strip(' .')[:150]
    if not stem:
        raise ValueError('The file has an invalid name')
    if stem.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        stem = '_' + stem
    return stem + '.safetensors'


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


def _upload_directory():
    models = Path(folder_paths.models_dir).resolve()
    directory = (models / 'loras' / 'UserUploads').resolve()
    if not directory.is_relative_to(models):
        raise ValueError('The LoRA upload folder must be inside the models folder')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_safetensors(path):
    # safe_open checks tensor offsets against the file without loading tensors.
    with safe_open(str(path), framework='numpy') as weights:
        names = list(weights.keys())
        if not names:
            raise ValueError('The safetensors file contains no weights')
        for name in names:
            weights.get_slice(name).get_shape()


def _publish(pending, filename):
    target = pending.parent / filename
    index = 0
    while True:
        try:
            # Same-directory hard linking publishes atomically without overwriting.
            os.link(pending, target)
            return target
        except FileExistsError:
            index += 1
            target = pending.parent / f'{Path(filename).stem} ({index}).safetensors'


async def upload_lora(request):
    if not _same_origin(request):
        return web.json_response({'error': 'Upload LoRAs from this ComfyUI page'}, status=403)
    if request.content_type != 'multipart/form-data':
        return web.json_response({'error': 'Choose a .safetensors file to upload'}, status=400)
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES + CHUNK_BYTES:
        return web.json_response({'error': 'LoRA uploads must be 4 GiB or smaller'}, status=413)
    pending = None
    try:
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != 'file' or not part.filename:
            raise ValueError('Upload one .safetensors file in the file field')
        filename = _filename(part.filename)
        directory = _upload_directory()
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.', suffix='.pending', delete=False) as output:
            pending = Path(output.name)
            total = 0
            while chunk := await part.read_chunk(size=CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    return web.json_response({'error': 'LoRA uploads must be 4 GiB or smaller'}, status=413)
                output.write(chunk)
        if await reader.next() is not None:
            raise ValueError('Upload one LoRA file at a time')
        _validate_safetensors(pending)
        target = _publish(pending, filename)
        folder_paths.filename_list_cache.pop('loras', None)
        folder_paths.get_filename_list('loras')
        return web.json_response({'filename': 'UserUploads/' + target.name})
    except (ValueError, SafetensorError):
        return web.json_response({'error': 'The upload must be one valid .safetensors file with a plain filename'}, status=400)
    except OSError:
        return web.json_response({'error': 'Could not save the LoRA. Check available disk space and folder access.'}, status=500)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def register_lora_upload_routes():
    from server import PromptServer
    PromptServer.instance.routes.post('/h3-studio/upload-lora')(upload_lora)
