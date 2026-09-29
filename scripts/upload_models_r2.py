"""Upload installed catalog models to a private R2 bucket. Never delete local files."""
import argparse
import getpass
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit


def inventory(catalog, root):
    found, missing = [], []
    seen = set()
    for spec in catalog['files'].values():
        dest = spec['destination']
        if dest in seen:
            continue
        seen.add(dest)
        path = (root / dest).resolve()
        if not path.is_relative_to(root.resolve()) or not dest.startswith('models/'):
            raise ValueError('Unsafe model destination')
        if path.is_file():
            found.append((path, spec))
        else:
            missing.append(dest)
    return found, missing


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Upload every available catalog model')
    parser.add_argument('--comfy', type=Path, default=Path('/workspace/runpod-slim/ComfyUI'))
    parser.add_argument('--catalog', type=Path, default=Path('/opt/10sorlabs/catalog/selfism.json'))
    parser.add_argument('--bucket', default='selfism-models')
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding='utf-8'))
    found, missing = inventory(catalog, args.comfy)
    if not args.all:
        found = [(p, s) for p, s in found if s.get('id') == 'encoder']
    if not found:
        raise SystemExit('No matching installed models. Complete the dashboard installation first.')
    print('Upload plan (existing catalog files only):', flush=True)
    for path, spec in found:
        print(f"  {spec['destination']}  {path.stat().st_size / 1e9:.2f} GB", flush=True)
    print(f'Total: {sum(p.stat().st_size for p, _ in found)/1e9:.2f} GB', flush=True)
    if missing:
        print('Not installed (includes unselected formats / optional models):', flush=True)
        for name in missing:
            print('  '+name, flush=True)
    print('Private Millie LoRA is not included in this catalog transfer.', flush=True)
    endpoint = input('S3 endpoint (https://ACCOUNT.r2.cloudflarestorage.com): ').strip().rstrip('/')
    parts = urlsplit(endpoint)
    if (parts.scheme != 'https' or not (parts.hostname or '').endswith('.r2.cloudflarestorage.com')
            or parts.username or parts.password or parts.port or parts.query or parts.fragment):
        raise SystemExit('Expected a Cloudflare R2 HTTPS endpoint.')
    # Bucket settings may copy an endpoint with /bucket appended.
    if parts.path not in ('', '/'+args.bucket):
        raise SystemExit('Unexpected endpoint path; use the S3 endpoint shown by Cloudflare.')
    endpoint = 'https://' + parts.hostname
    access = getpass.getpass('Access Key ID (hidden): ').strip()
    secret = getpass.getpass('Secret Access Key (hidden): ').strip()
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError
    client = boto3.client('s3', endpoint_url=endpoint, region_name='auto',
                          aws_access_key_id=access, aws_secret_access_key=secret,
                          config=Config(signature_version='s3v4', retries={'max_attempts':5}))
    client.list_objects_v2(Bucket=args.bucket, MaxKeys=1)
    config = TransferConfig(multipart_threshold=64*1024**2,
                            multipart_chunksize=64*1024**2, max_concurrency=4)
    for path, spec in found:
        key = spec['destination'].removeprefix('models/')
        size = path.stat().st_size
        print('Checking SHA256: '+key, flush=True)
        digest = digest_file(path)
        if spec.get('sha256') and digest != spec['sha256'].lower():
            raise SystemExit('Local checksum mismatch; upload stopped: '+key)
        if spec.get('size_bytes') and size != spec['size_bytes']:
            raise SystemExit('Local size mismatch; upload stopped: '+key)
        try:
            existing = client.head_object(Bucket=args.bucket, Key=key)
        except ClientError as exc:
            if str(exc.response['Error']['Code']) not in ('404', 'NoSuchKey', 'NotFound'):
                raise
            existing = None
        if existing:
            if existing['ContentLength'] == size and existing.get('Metadata', {}).get('sha256') == digest:
                print('Already uploaded: '+key, flush=True)
                continue
            raise SystemExit('Different object already exists; refusing overwrite: '+key)
        start = time.monotonic()
        print('Uploading: '+key+' (large files may take several minutes)', flush=True)
        client.upload_file(str(path), args.bucket, key,
                           ExtraArgs={'Metadata':{'sha256':digest}}, Config=config)
        remote = client.head_object(Bucket=args.bucket, Key=key)
        if remote['ContentLength'] != size or remote.get('Metadata', {}).get('sha256') != digest:
            raise SystemExit('Remote size/metadata verification failed: '+key)
        seconds = time.monotonic()-start
        print(f'Uploaded: {key}; {seconds:.1f}s; {size/1e6/max(seconds,.001):.1f} MB/s', flush=True)
    print('Finished. This measures upload speed, not future RunPod download speed.', flush=True)
    print('Objects remain private. Local models and credentials were not saved or deleted.', flush=True)


if __name__ == '__main__':
    main()
