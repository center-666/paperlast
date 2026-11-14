"""Targeted final-version read of existing FPN16; library remains read-only."""
from pathlib import Path
import hashlib,json
import requests
import subprocess
H=Path(__file__).resolve().parent;D=H/'literature_work';D.mkdir(exist_ok=True)
url='https://pure.tudelft.nl/ws/portalfiles/portal/242463520/1-s2.0-S0005109825001931-main.pdf'
r=requests.get(url,timeout=35,proxies={'https':'http://127.0.0.1:7890','http':'http://127.0.0.1:7890'})
meta=dict(existing_library_id='FPN16',url=url,http_status=r.status_code,new_record_created=False,library_modified=False,
 purpose='read final publisher version of already indexed paper, not new publication',
 temporary_work_copy=True,not_uploaded=True)
if r.status_code==200 and r.content.startswith(b'%PDF'):
 p=D/'FPN16_publisher_final_work.pdf';p.write_bytes(r.content)
 out=D/'FPN16_final_all.txt'
 subprocess.run(['pdftotext','-layout',str(p),str(out)],check=True)
 pages=out.read_text(encoding='utf-8').split('\f')
 if not pages[-1].strip():pages.pop()
 meta.update(sha256=hashlib.sha256(r.content).hexdigest(),pages=len(pages))
 for k,t in enumerate(pages):(D/f'FPN16_final_p{k+1}.txt').write_text(t,encoding='utf-8')
 subprocess.run(['pdftoppm','-f','4','-l','5','-scale-to','1400','-png',str(p),str(D/'FPN16_final')],check=True)
 meta['read_scope']='targeted PDF4-6/printed3-5 and reference portions PDF7-8/printed6-7; text plus PDF4-5 images; not full paper audit'
else:meta['error']='No readable PDF returned; do not infer full-text access'
(H/'final_version_source.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print(json.dumps(meta,indent=2))
