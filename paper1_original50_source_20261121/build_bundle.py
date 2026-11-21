"""Build a reproducible portable bundle; raw recovered sources remain unchanged."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'historical50_source_bundle.zip'


def build():
    # Normalize only newly authored text; byte-preserved raw originals are exempt.
    root_files=['.gitattributes','README.md','completeness.md','dialogue.md','source_manifest.json',
                'remote_source_index.json','configuration_snapshot.json','parity_verification.json',
                'gain_interface_audit.json','initial_stacked_gains_regenerated.npz',
                'parity_matrices_regenerated.npz','requirements.txt','extract_config.py',
                'run_legacy.py','verify_package.py','build_bundle.py','verification.json']
    for name in root_files:
        p=HERE/name
        if p.suffix in {'.py','.md','.txt','.json'}:
            text=p.read_text(encoding='utf-8-sig').replace('\r\n','\n')
            p.write_text(text,encoding='utf-8',newline='\n')
    source=json.loads((HERE/'source_manifest.json').read_text())
    files=sorted([HERE/name for name in root_files]+[HERE/entry['path'] for entry in source['files']])
    assert all(p.is_file() for p in files)
    members=[]
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            name=HERE.name+'/'+p.relative_to(HERE).as_posix();data=p.read_bytes()
            info=zipfile.ZipInfo(name,date_time=(2026,11,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
            members.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert len(z.namelist())==len(files)
        for entry in members:
            assert '..' not in Path(entry['path']).parts
            assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    data=ARCHIVE.read_bytes();result=dict(archive=ARCHIVE.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
        scope='New portable archive of recovered historical source and explicitly regenerated defaults; not an original old-run ZIP.',members=members)
    (HERE/'bundle_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='members'},indent=2))


if __name__=='__main__':build()
