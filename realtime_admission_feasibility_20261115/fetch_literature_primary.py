"""Finite primary-source attempts, no authentication bypass or library writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,hashlib,json,subprocess
H=Path(__file__).resolve().parent;W=H/'literature_work';W.mkdir(exist_ok=True)
targets=[
 ('FPN18_final','https://pure.tudelft.nl/ws/portalfiles/portal/249131961/1-s2.0-S0005109825003541-main.pdf'),
 ('FPN19_final','https://link.springer.com/content/pdf/10.1007/s11432-025-4567-3.pdf'),
 ('FPN31_ieee','https://doi.org/10.1109/TPWRS.2025.3586788'),
 ('FPN32_ieee','https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11316654')]
def get(item):
 name,url=item
 try:
  r=requests.get(url,timeout=25,proxies={'https':'http://127.0.0.1:7890','http':'http://127.0.0.1:7890'})
  d=dict(key=name,url=url,status=r.status_code,final_url=r.url,bytes=len(r.content),is_pdf=r.content.startswith(b'%PDF'))
  if d['is_pdf']:
   p=W/(name+'.pdf');p.write_bytes(r.content);d['sha256']=hashlib.sha256(r.content).hexdigest()
   subprocess.run(['pdftotext','-layout',str(p),str(W/(name+'.txt'))],check=True)
  return d
 except Exception as e:return dict(key=name,url=url,error=str(e),is_pdf=False)
with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(get,targets))
(H/'literature_fetch_primary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False))
