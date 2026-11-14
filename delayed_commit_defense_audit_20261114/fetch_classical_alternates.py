from pathlib import Path
import requests,json,hashlib,subprocess
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent; W=H/'literature_work'
targets=[('checkpoint2018_hnu','https://esnl.hnu.edu.cn/ecps2018/2019.3.19.1.pdf'),
 ('moore1973_direct','https://users.cecs.anu.edu.au/~john/papers/JOUR/033.PDF')]
def get(item):
 name,url=item
 try:
  session=requests.Session();session.trust_env=False
  r=session.get(url,timeout=20)
  d=dict(name=name,url=url,final_url=r.url,status=r.status_code,bytes=len(r.content),is_pdf=r.content.startswith(b'%PDF'))
  if d['is_pdf']:
   f=W/(name+'.pdf');f.write_bytes(r.content);d['sha256']=hashlib.sha256(r.content).hexdigest()
   subprocess.run(['pdftotext','-layout',str(f),str(W/(name+'.txt'))],check=True)
  return d
 except Exception as e:return dict(name=name,url=url,error=str(e),is_pdf=False)
with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(get,targets))
(H/'classical_alternate_sources.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows))
