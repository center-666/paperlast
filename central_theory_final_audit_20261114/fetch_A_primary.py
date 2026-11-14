"""One finite attempt at the three already-registered direct neighbors.
Only official DOI/IEEE endpoints are requested, with no authentication bypass.
"""
from pathlib import Path
import hashlib,json,requests
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent;W=H/'literature_work';W.mkdir(exist_ok=True)
targets=[
 ('FPN07','https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8814300'),
 ('FPN08','https://doi.org/10.1109/TCYB.2024.3413969'),
 ('FPN11','https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9691881'),
]
def fetch(pair):
 rid,url=pair
 try:
  r=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0'})
  pdf=r.content.startswith(b'%PDF-')
  row=dict(id=rid,requested_url=url,final_url=r.url,status=r.status_code,
   bytes=len(r.content),content_type=r.headers.get('Content-Type',''),is_pdf=pdf,
   redirects=[dict(status=x.status_code,url=x.url)for x in r.history])
  if pdf:
   p=W/(rid+'_official_attempt.pdf');p.write_bytes(r.content)
   row.update(saved_path=str(p),sha256=hashlib.sha256(r.content).hexdigest())
  else:row['result']='No core PDF obtained; HTML/access error is not full-text review.'
  return row
 except Exception as exc:return dict(id=rid,requested_url=url,error=str(exc),is_pdf=False)
with ThreadPoolExecutor(max_workers=3)as pool:rows=list(pool.map(fetch,targets))
(H/'primary_fetch_A.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
