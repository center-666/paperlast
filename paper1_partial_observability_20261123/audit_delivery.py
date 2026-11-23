"""Refresh exact artifact manifests or verify Git's staged artifact bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent.parent
PACKAGES=('paper1_sscd_commonrandom_20261122','paper1_partial_observability_20261123')


def files(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.name!='artifact_manifest.json')


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh',action='store_true')
    p.add_argument('--check-stage',action='store_true');a=p.parse_args();count=0
    for package in PACKAGES:
        directory=ROOT/package
        if a.refresh:
            document={'scope':'Exact delivered package bytes; excludes bytecode and this manifest. Original ZIP, executed snapshots and user attachment are preserved without line-ending conversion.',
                'files':{str(f.relative_to(directory)).replace('\\','/'):{'bytes':f.stat().st_size,'sha256':digest(f.read_bytes())}
                         for f in files(directory)}}
            (directory/'artifact_manifest.json').write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8',newline='\n')
        manifest=json.loads((directory/'artifact_manifest.json').read_text(encoding='utf-8'))
        assert set(manifest['files'])=={str(f.relative_to(directory)).replace('\\','/') for f in files(directory)}
        for name,entry in manifest['files'].items():
            data=(directory/name).read_bytes();assert len(data)==entry['bytes'] and digest(data)==entry['sha256'],name
            count+=1
            if a.check_stage:
                staged=subprocess.run(['git','show',f':{package}/{name}'],cwd=ROOT,check=True,capture_output=True).stdout
                assert staged==data,f'Staged bytes differ: {package}/{name}'
        if a.check_stage:
            staged=subprocess.run(['git','show',f':{package}/artifact_manifest.json'],cwd=ROOT,check=True,capture_output=True).stdout
            assert staged==(directory/'artifact_manifest.json').read_bytes()
    print(json.dumps({'status':'PASS','artifact_files_checked':count,'staged_bytes_checked':a.check_stage},indent=2))


if __name__=='__main__':main()
