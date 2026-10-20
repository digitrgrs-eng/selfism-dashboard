import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_krast_org_catalog_and_workflow():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    files=[c['files'][k] for k in c['krast_org_files']]
    assert len(files)==12
    for f in files:
        assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and f['size_bytes']>0
    names={n['name'] for n in c['nodes']}
    assert set(c['krast_org_nodes'])<=names and len(c['krast_org_nodes'])==12
    wf=(ROOT/'selfism_workflows/krast_org.json').read_text()
    assert 'blonde_epoch_8' not in wf and 'KRASTINT8' not in wf and 'millie_000002750.safetensors' in wf
    dests={f['destination'].split('/',1)[1] for f in files}
    w=json.loads(wf); ns=list(w['nodes'])+[n for s in w['definitions']['subgraphs'] for n in s['nodes']]
    used=[v for n in ns if isinstance(n.get('widgets_values'),list) for v in n['widgets_values'] if isinstance(v,str) and v.endswith(('.safetensors','.pt','.pth'))]
    assert used
    for m in used:
        assert any(d.endswith(m) for d in dests), m
