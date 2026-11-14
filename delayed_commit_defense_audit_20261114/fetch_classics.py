"""Finite primary-source retrieval for the first gate, no method development."""
from pathlib import Path
import hashlib,json,requests,subprocess
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent; W=H/'literature_work';W.mkdir(exist_ok=True)
targets=[('checkpoint2018','https://www.seas.upenn.edu/~weimerj/pdf/2018-ICCPS-CPScheckpointing.pdf'),
 ('moore1973','https://users.cecs.anu.edu.au/~john/papers/JOUR/033.PDF')]
def get(item):
 name,url=item
 try:
  r=requests.get(url,timeout=35,proxies={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'})
  d=dict(name=name,url=url,final_url=r.url,status=r.status_code,bytes=len(r.content),is_pdf=r.content.startswith(b'%PDF'))
  if d['is_pdf']:
   f=W/(name+'.pdf');f.write_bytes(r.content)
   d['sha256']=hashlib.sha256(r.content).hexdigest()
   subprocess.run(['pdftotext','-layout',str(f),str(W/(name+'.txt'))],check=True)
  return d
 except Exception as e:return dict(name=name,url=url,error=str(e),is_pdf=False)
with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(get,targets))
(H/'classical_sources.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows))
