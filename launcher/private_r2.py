"""Private R2 source selection; secrets and signed URLs never enter UI logs."""
import copy
import os
import re
from urllib.parse import urlsplit


def r2_client():
    endpoint=os.getenv('SELFISM_R2_ENDPOINT','').strip().rstrip('/')
    bucket=os.getenv('SELFISM_R2_BUCKET','').strip()
    access=os.getenv('SELFISM_R2_ACCESS_KEY_ID','').strip()
    secret=os.getenv('SELFISM_R2_SECRET_ACCESS_KEY','').strip()
    if not any((endpoint,bucket,access,secret)):
        return None, '', 'Private R2 not configured.'
    if not all((endpoint,bucket,access,secret)):
        return None, '', 'Private R2 settings incomplete; using fallback sources.'
    parts=urlsplit(endpoint)
    if (parts.scheme!='https' or not (parts.hostname or '').endswith('.r2.cloudflarestorage.com')
            or parts.path or parts.query or parts.fragment or parts.username or parts.password or parts.port):
        return None, '', 'Private R2 endpoint invalid; use the account S3 HTTPS endpoint without bucket path.'
    from boto3 import client
    from botocore.config import Config
    return client('s3',endpoint_url=endpoint,region_name='auto',
                  aws_access_key_id=access,aws_secret_access_key=secret,
                  config=Config(signature_version='s3v4',connect_timeout=5,read_timeout=10,
                                retries={'max_attempts':1})), bucket, ''


def select_private(files, client, bucket):
    selected={}
    messages=[]
    if client is None:
        return selected,messages
    for index, original in enumerate(files):
        digest=str(original.get('sha256','')).lower()
        name=original.get('name',original['destination'])
        if not re.fullmatch(r'[0-9a-f]{64}',digest):
            messages.append(f'{name}: private R2 skipped (no trusted source SHA256).')
            continue
        key=original['destination'].removeprefix('models/')
        try:
            obj=client.head_object(Bucket=bucket,Key=key)
            size=obj['ContentLength']
            if (obj.get('Metadata',{}).get('sha256','').lower()!=digest
                    or not isinstance(size,int) or size<=0
                    or (original.get('size_bytes') and original['size_bytes']!=size)):
                messages.append(f'{name}: private R2 verification mismatch; using fallback.')
                continue
            url=client.generate_presigned_url('get_object',Params={'Bucket':bucket,'Key':key},ExpiresIn=86400)
            result=copy.deepcopy(original)
            result.update(url=url,auth='none',parallel=True,verify=True,size_bytes=size)
            selected[index]=result
            messages.append(f'{name}: Private R2 (identical SHA256).')
        except Exception:
            messages.append(f'{name}: private R2 missing or inaccessible; using fallback.')
    return selected,messages
