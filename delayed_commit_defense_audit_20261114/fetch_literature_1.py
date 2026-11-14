"""Finite primary-source retrieval for Li/Cheng/Xu; no library mutation."""
from pathlib import Path
import requests,subprocess,hashlib,json
H=Path(__file__).resolve().parent;W=H/'literature_work';W.mkdir(exist_ok=True)
proxies={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
targets=[('Li2023','https://people.kth.se/~kallej/papers/Secure_CDC2023_Li.pdf'),
('Cheng2023','https://doi.org/10.1109/TCYB.2021.3108884'),
('Xu2023','https://www.sciencedirect.com/science/article/pii/S0005109823003333/pdfft?isDTMRedir=true&download=true'),
('Xu_author','https://yzli925.github.io/publications')]
rows=[]
for key,url in targets:
 try:
  r=requests.get(url,timeout=35,proxies=proxies,headers={'User-Agent':'Mozilla/5.0'})
  row=dict(key=key,url=url,status=r.status_code,final_url=r.url,bytes=len(r.content),content_type=r.headers.get('Content-Type'),is_pdf=r.content.startswith(b'%PDF'))
  if row['is_pdf']:
   p=W/f'{key}_primary.pdf';p.write_bytes(r.content)
   row.update(file=str(p),sha256=hashlib.sha256(r.content).hexdigest())
   subprocess.run(['pdftotext','-layout',str(p),str(W/f'{key}_all.txt')],check=True)
   pages=(W/f'{key}_all.txt').read_text(encoding='utf-8').split('\f')
   if not pages[-1].strip():pages.pop()
   row['pages']=len(pages)
   for i,t in enumerate(pages): (W/f'{key}_p{i+1}.txt').write_text(t,encoding='utf-8')
  elif key.endswith('author'):
   (W/f'{key}.html').write_bytes(r.content)
  rows.append(row)
 except Exception as e:rows.append(dict(key=key,url=url,error=str(e),is_pdf=False))
(H/'literature_1_sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False,indent=2))
