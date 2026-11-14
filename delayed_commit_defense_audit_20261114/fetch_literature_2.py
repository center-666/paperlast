from pathlib import Path
import json,hashlib,requests,subprocess,re
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent; W=H/'literature_work'; W.mkdir(exist_ok=True)
targets=[('Chen','https://www.ieee-jas.com/en/article/doi/10.1109/JAS.2025.125900'),
 ('Chen_net','https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125900'),
 ('Bhattacharya','https://dl.acm.org/doi/pdf/10.1145/3777903'),
 ('Bhattacharya_html','https://dl.acm.org/doi/full/10.1145/3777903'),
 ('Zhang','https://www.sciencedirect.com/science/article/pii/S0016003226005107/pdfft?isDTMRedir=true&download=true'),
 ('Gupta','https://doi.org/10.1016/j.sigpro.2026.110516')]
def f(pair):
 key,url=pair
 try:
  r=requests.get(url,timeout=35,proxies={'https':'http://127.0.0.1:7890','http':'http://127.0.0.1:7890'},headers={'User-Agent':'Mozilla/5.0'})
  pdf=r.content.startswith(b'%PDF'); suffix='.pdf' if pdf else '.html'
  p=W/(key+suffix);p.write_bytes(r.content)
  row=dict(key=key,url=url,final_url=r.url,status=r.status_code,bytes=len(r.content),pdf=pdf,
   local_temporary=str(p),sha256=hashlib.sha256(r.content).hexdigest())
  if pdf:subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  if not pdf:row['links']=re.findall(r'(?:href|src|content)=[\"\']([^\"\']+)[\"\']',r.text)
  return row
 except Exception as e:return dict(key=key,url=url,error=str(e),pdf=False)
with ThreadPoolExecutor(max_workers=6)as pool:rows=list(pool.map(f,targets))
(H/'literature_2_fetch.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items()if k not in ['links','local_temporary']}for r in rows],ensure_ascii=False))
