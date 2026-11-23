"""Losslessly split the original large C2 NPZ into GitHub-sized array shards."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

HERE=Path(__file__).resolve().parent
FOLDER=HERE/'results/C2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    canonical=np.ascontiguousarray(value)
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def split():
    source=FOLDER/'paths.npz'
    if not source.is_file():
        raise FileNotFoundError('Expected the executed unsplit C2 path archive.')
    if (FOLDER/'archive_manifest.json').exists():
        raise FileExistsError('C2 paths have already been sharded.')
    original_hash=digest(source)
    result= json.loads((FOLDER/'results.json').read_text(encoding='utf-8'))
    assert result['archive_sha256']==original_hash
    groups={'paths_core':{},'ewma_mean':{},'ewma_covariance_diagonal':{}}
    entries={}
    with np.load(source,allow_pickle=False) as original:
        for name in original.files:
            value=original[name]
            if 'mean_critical' in name:
                group='ewma_mean'
            elif 'covariance_diagonal_critical' in name:
                group='ewma_covariance_diagonal'
            else:
                group='paths_core'
            groups[group][name]=value.copy()
            entries[name]={'shard':group+'.npz','shape':list(value.shape),'dtype':value.dtype.str,
                           'array_sha256':array_digest(value)}
    assert len(entries)==result['array_count'] and len(groups['paths_core'])
    with TemporaryDirectory(dir=FOLDER) as temporary:
        temp=Path(temporary)
        shards={}
        for group,arrays in groups.items():
            path=temp/(group+'.npz')
            np.savez_compressed(path,**arrays)
            if path.stat().st_size >= 100_000_000:
                raise ValueError(f"{path.name} exceeds GitHub's 100 MB per-file limit")
            shards[path.name]={'bytes':path.stat().st_size,'sha256':digest(path),
                               'member_count':len(arrays)}
        manifest={'status':'PASS','original_execution_archive_sha256':original_hash,
                  'original_execution_archive_bytes':source.stat().st_size,
                  'storage':'Lossless extracted NPY arrays, grouped into independently checksummed NPZ shards.',
                  'arrays':entries,'shards':shards}
        manifest_path=temp/'archive_manifest.json'
        save_json(manifest_path,manifest)
        for name in shards:
            (temp/name).replace(FOLDER/name)
        manifest_path.replace(FOLDER/'archive_manifest.json')
    # Read every shard after rename and verify every dtype, shape and raw member hash.
    for name,details in shards.items():
        assert digest(FOLDER/name)==details['sha256']
        with np.load(FOLDER/name,allow_pickle=False) as check:
            for member in check.files:
                info=entries[member]
                value=check[member]
                assert list(value.shape)==info['shape'] and value.dtype.str==info['dtype']
                assert array_digest(value)==info['array_sha256'],member
    result['storage_archive']={'format':'three lossless NPZ shards',
        'original_execution_archive_sha256':original_hash,'manifest':'archive_manifest.json',
        'shards':shards}
    save_json(FOLDER/'results.json',result)
    execution=FOLDER/'execution_manifest.json'
    execution_data=json.loads(execution.read_text(encoding='utf-8'))
    execution_data['storage_shards']=shards
    execution_data['storage_manifest_sha256']=digest(FOLDER/'archive_manifest.json')
    save_json(execution,execution_data)
    source.unlink()
    print(json.dumps({'status':'PASS','source_bytes':manifest['original_execution_archive_bytes'],
                      'shards':shards,'arrays':len(entries),
                      'result_sha256':digest(FOLDER/'results.json')},indent=2))


def verify():
    manifest=json.loads((FOLDER/'archive_manifest.json').read_text(encoding='utf-8'))
    result=json.loads((FOLDER/'results.json').read_text(encoding='utf-8'))
    assert result['storage_archive']['original_execution_archive_sha256']==manifest['original_execution_archive_sha256']
    seen=set()
    for name,details in manifest['shards'].items():
        path=FOLDER/name
        assert path.stat().st_size<100_000_000 and digest(path)==details['sha256']
        with np.load(path,allow_pickle=False) as data:
            assert len(data.files)==details['member_count']
            for member in data.files:
                assert member not in seen
                seen.add(member)
                value=data[member]
                spec=manifest['arrays'][member]
                assert list(value.shape)==spec['shape'] and value.dtype.str==spec['dtype']
                assert array_digest(value)==spec['array_sha256']
    assert seen==set(manifest['arrays'])
    print(f'PASS: {len(seen)} exact arrays verified across {len(manifest["shards"])} shards.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--folder',type=Path,default=FOLDER)
    args=parser.parse_args()
    FOLDER=args.folder.resolve()
    if args.verify_only:
        verify()
    else:
        split()
