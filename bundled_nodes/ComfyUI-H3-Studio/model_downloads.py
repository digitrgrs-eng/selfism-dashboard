"""Explicit, allowlisted model downloads for the Studio setup node."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from aiohttp import web
import folder_paths


CATALOG_PATH = Path(__file__).with_name('model_catalog.json')
ALLOWED_FOLDERS = {'diffusion_models', 'loras', 'vae', 'text_encoders', 'checkpoints'}
ACTIVE_STATES = {'starting', 'downloading', 'verifying', 'cancelling'}
BLOCK_SIZE = 1024 * 1024


class DownloadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class DownloadCancelled(Exception):
    pass


def same_origin(request):
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


def load_catalog():
    data = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    models = data['models']
    identifiers = set()
    for model in models:
        identifier, name = model['id'], model['filename']
        if not re.fullmatch(r'[a-z0-9_-]+', identifier) or identifier in identifiers:
            raise ValueError('Invalid or duplicated model catalog ID.')
        identifiers.add(identifier)
        if (model['folder'] not in ALLOWED_FOLDERS or not name or name in ('.', '..')
                or any(char in name for char in '/\\:\x00')):
            raise ValueError('Model catalog destinations must be fixed model filenames.')
        if model.get('manual'):
            continue
        url = urlsplit(model['download_url'])
        if (url.scheme != 'https' or url.hostname != 'huggingface.co' or url.username
                or url.password or not re.search(r'/resolve/[a-f0-9]{40}/', url.path)):
            raise ValueError('Automatic downloads require a pinned Hugging Face artifact URL.')
        if not isinstance(model['size_bytes'], int) or model['size_bytes'] <= 0:
            raise ValueError('Automatic downloads require the verified artifact size.')
        if not re.fullmatch(r'[a-f0-9]{64}', model.get('sha256', '')):
            raise ValueError('Automatic downloads require a verified SHA-256.')
    return models


def model_directories(folder):
    if folder not in ALLOWED_FOLDERS:
        raise DownloadError('Unknown model folder.')
    try:
        registered = folder_paths.get_folder_paths(folder)
    except KeyError:
        registered = []
    paths = [Path(path).resolve() for path in registered]
    default = (Path(folder_paths.models_dir) / folder).resolve()
    if default not in paths:
        paths.append(default)
    return paths


def model_location(model):
    directories = model_directories(model['folder'])
    for directory in directories:
        path = directory / model['filename']
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                return path, 'blocked'
            expected = model.get('size_bytes')
            return path, 'present' if not expected or path.stat().st_size == expected else 'size_mismatch'
    return directories[0] / model['filename'], 'missing'


def partial_paths(target):
    return target.with_name('.' + target.name + '.h3-download.part'), target.with_name('.' + target.name + '.h3-download.json')


def resume_fingerprint(model):
    return {key: model[key] for key in ('id', 'download_url', 'size_bytes', 'sha256')}


class ModelDownloads:
    def __init__(self, models=None, opener=urlopen):
        self.models = {model['id']: model for model in (models if models is not None else load_catalog())}
        self.opener = opener
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.job = None
        self.thread = None

    def snapshot(self):
        with self.lock:
            job = dict(self.job) if self.job else None
        rows = []
        for model in self.models.values():
            _, status = model_location(model)
            rows.append({**model, 'local_status': status,
                         'destination': f"models/{model['folder']}/{model['filename']}"})
        return {'models': rows, 'job': job}

    def update(self, **values):
        with self.lock:
            self.job.update(values)

    def start(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.models:
            raise DownloadError('Choose a model from this node’s catalog.')
        model = self.models[identifier]
        if model.get('manual'):
            raise DownloadError(model.get('note') or 'Open the model page and follow its download requirements.')
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise DownloadError('A model is already downloading. Wait for it or cancel it first.', 409)
            target, status = model_location(model)
            if status == 'present':
                return {'id': identifier, 'state': 'present', 'message': 'Already present; no download was started.'}
            if status != 'missing':
                raise DownloadError('An existing file has a different size or is a link. Check it manually; the downloader will not overwrite it.', 409)
            self.cancel_event = threading.Event()
            self.job = {'id': identifier, 'state': 'starting', 'downloaded_bytes': 0,
                        'total_bytes': model['size_bytes'], 'error': '', 'started_at': time.time(),
                        'message': 'Preparing download…'}
            self.thread = threading.Thread(target=self.run, args=(model, target), daemon=True,
                                           name='h3-model-download')
            self.thread.start()
            return dict(self.job)

    def cancel(self):
        with self.lock:
            if self.job and self.job['state'] in ACTIVE_STATES:
                self.cancel_event.set()
                self.job.update(state='cancelling', message='Cancelling after the current network read…')
            return dict(self.job) if self.job else None

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise DownloadCancelled()

    def run(self, model, target):
        try:
            self.transfer(model, target)
            self.update(state='complete', downloaded_bytes=model['size_bytes'],
                        message='Downloaded and SHA-256 verified. Ready to use.')
        except DownloadCancelled:
            self.update(state='cancelled', message='Cancelled. The matching partial download is kept for the next attempt.')
        except HTTPError as error:
            if error.code in (401, 403):
                message = 'The host requires access approval or a sign-in. Open the model page and download manually; this node never asks for credentials.'
            else:
                message = f'Download host returned HTTP {error.code}. Try again or use the model page.'
            self.update(state='error', error=message, message=message)
        except (DownloadError, OSError, URLError, ValueError, http.client.HTTPException) as error:
            self.update(state='error', error=str(error), message=str(error))

    def transfer(self, model, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        part, metadata = partial_paths(target)
        if part.is_symlink() or metadata.is_symlink():
            raise DownloadError('The download’s temporary path is a link. Remove that link manually before trying again.')
        fingerprint = resume_fingerprint(model)
        offset = 0
        if part.exists():
            try:
                previous = json.loads(metadata.read_text(encoding='utf-8'))
            except (FileNotFoundError, json.JSONDecodeError):
                raise DownloadError('A partial download has no valid ownership record. Check the .h3-download files manually.')
            if previous != fingerprint or not part.is_file():
                raise DownloadError('A different partial download already uses this filename. It will not be overwritten.')
            offset = part.stat().st_size
            if offset > model['size_bytes']:
                raise DownloadError('The partial download is larger than the published artifact. Remove the .h3-download files manually.')
        else:
            if metadata.exists():
                try:
                    previous = json.loads(metadata.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    raise DownloadError('An invalid partial-download ownership record already exists.')
                if previous != fingerprint:
                    raise DownloadError('A different download already owns this temporary filename.')
            else:
                with metadata.open('x', encoding='utf-8') as stream:
                    json.dump(fingerprint, stream)
        missing = model['size_bytes'] - offset
        if shutil.disk_usage(target.parent).free < missing + 64 * 1024 ** 2:
            raise DownloadError('Not enough free space for this model. Free disk space and try again.')
        self.check_cancelled()
        if offset < model['size_bytes']:
            headers = {'User-Agent': 'ComfyUI-H3-Studio-Model-Downloader', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = Request(model['download_url'], headers=headers)
            with self.opener(request, timeout=30) as response:
                status = response.status
                final_url = urlsplit(response.geturl())
                if final_url.scheme != 'https':
                    raise DownloadError('The download redirected to an insecure connection.')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise DownloadError('The download server returned an unexpected encoding.')
                if status == 206:
                    expected = f"bytes {offset}-{model['size_bytes'] - 1}/{model['size_bytes']}"
                    if response.headers.get('Content-Range') != expected:
                        raise DownloadError('The server returned an unexpected byte range. The partial model was not changed.')
                elif status == 200:
                    offset = 0
                else:
                    raise DownloadError(f'Unexpected download response ({status}).')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != model['size_bytes'] - offset:
                    raise DownloadError('The server’s file size differs from the published model. Nothing was installed.')
                self.update(state='downloading', downloaded_bytes=offset,
                            message='Resuming download…' if offset else 'Downloading…')
                with part.open('ab' if offset else 'wb') as output:
                    while True:
                        self.check_cancelled()
                        chunk = response.read(BLOCK_SIZE)
                        if not chunk:
                            break
                        if offset + len(chunk) > model['size_bytes']:
                            raise DownloadError('The server sent more bytes than the published model size.')
                        output.write(chunk)
                        offset += len(chunk)
                        self.update(downloaded_bytes=offset)
                    output.flush()
                    os.fsync(output.fileno())
        if not part.exists() or part.stat().st_size != model['size_bytes']:
            raise DownloadError('Download interrupted before the model was complete. Download again to resume.')
        self.check_cancelled()
        self.update(state='verifying', message='Checking SHA-256 before installing…')
        digest = hashlib.sha256()
        with part.open('rb') as source:
            while chunk := source.read(BLOCK_SIZE):
                self.check_cancelled()
                digest.update(chunk)
        if digest.hexdigest() != model['sha256']:
            # This file is ours, and cannot be used or resumed after a failed hash.
            part.unlink()
            metadata.unlink()
            raise DownloadError('The downloaded model failed SHA-256 verification. No model was installed; retry the download.')
        self.check_cancelled()
        # Hard-link publication is atomic and fails if another file appeared.
        # Unlike replace(), this never overwrites a model already on disk.
        os.link(part, target)
        part.unlink()
        metadata.unlink()


class H3StudioModelDownloads:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}

    RETURN_TYPES = ()
    FUNCTION = 'show'
    CATEGORY = 'MiniMax H3/Studio'
    DESCRIPTION = 'Model setup links and explicit download buttons. Opening or running a workflow never downloads models.'

    def show(self):
        return ()


DOWNLOAD_NODES = {'H3StudioModelDownloads': H3StudioModelDownloads}
_downloads = None


def manager():
    global _downloads
    if _downloads is None:
        _downloads = ModelDownloads()
    return _downloads


async def list_models(request):
    try:
        return web.json_response(manager().snapshot())
    except (ValueError, OSError) as error:
        return web.json_response({'error': str(error)}, status=500)


async def model_action(request):
    if not same_origin(request):
        return web.json_response({'error': 'Open this control from the same ComfyUI site.'}, status=403)
    if request.content_length is not None and request.content_length > 1024:
        return web.json_response({'error': 'Request is too large.'}, status=413)
    try:
        raw = b''
        while len(raw) <= 1024:
            chunk = await request.content.read(1025 - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) > 1024:
            raise DownloadError('Request is too large.', 413)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) - {'id', 'action'}:
            raise DownloadError('Only a catalog ID and download action are accepted.')
        action = payload.get('action', 'download')
        if action == 'cancel':
            return web.json_response({'job': manager().cancel()})
        if action != 'download':
            raise DownloadError('Unknown model action.')
        return web.json_response({'job': manager().start(payload.get('id'))}, status=202)
    except DownloadError as error:
        return web.json_response({'error': str(error)}, status=error.status)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return web.json_response({'error': 'Expected a JSON catalog ID and action.'}, status=400)
    except OSError as error:
        return web.json_response({'error': str(error)}, status=500)


def register_model_download_routes():
    from server import PromptServer
    PromptServer.instance.routes.get('/h3-studio/models')(list_models)
    PromptServer.instance.routes.post('/h3-studio/models')(model_action)
