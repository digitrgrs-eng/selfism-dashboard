import asyncio
import importlib
from launcher.private_r2 import select_private, r2_client

s=importlib.import_module('launcher.selfism')

def model():
    return dict(name='model',destination='models/vae/a',url='https://original/a',sha256='a'*64,size_bytes=12)

class Client:
    def head_object(self,**kwargs):
        assert kwargs=={'Bucket':'test','Key':'vae/a'}
        return {'ContentLength':12,'Metadata':{'sha256':'a'*64}}
    def generate_presigned_url(self,*args,**kwargs):
        return 'https://test.r2.cloudflarestorage.com/private?signature=secret'

def test_private_exact_match_and_no_url_in_logs():
    f=model(); chosen,logs=select_private([f],Client(),'test')
    assert chosen[0]['url']!=f['url'] and chosen[0]['verify'] is True
    assert chosen[0]['destination']==f['destination'] and chosen[0]['auth']=='none'
    assert 'secret' not in str(logs)

def test_mismatch_missing_and_unknown_digest():
    for mutation in ({'sha256':'b'*64},{'size_bytes':13},{'sha256':''}):
        f=model(); f.update(mutation)
        assert select_private([f],Client(),'test')[0]=={}
    class Missing(Client):
        def head_object(self,**kwargs): raise RuntimeError('secret error detail')
    chosen,logs=select_private([model()],Missing(),'test')
    assert not chosen and 'secret' not in str(logs)

def test_priority_only_unmatched_files_reach_rapidcache(monkeypatch):
    monkeypatch.setattr(s,'r2_client',lambda:(Client(),'test',''))
    calls=[]
    async def fallback(host,files):
        calls.append(files)
        return files,[]
    monkeypatch.setattr(s,'resolve_rapidcache',fallback)
    a=model(); b=dict(model(),sha256='b'*64)
    result,_=asyncio.run(s.resolve_sources(None,[a,b]))
    assert calls==[[b]] and result[1]==b and result[0]['url']!=a['url']

def test_incomplete_config_does_not_create_client(monkeypatch):
    for key in ('ENDPOINT','BUCKET','ACCESS_KEY_ID','SECRET_ACCESS_KEY'):
        monkeypatch.delenv('SELFISM_R2_'+key,raising=False)
    assert r2_client()[0] is None
    monkeypatch.setenv('SELFISM_R2_ENDPOINT','https://example.com')
    assert 'incomplete' in r2_client()[2]
