from pathlib import Path
import json,requests,hashlib,subprocess
H=Path(__file__).resolve().parent; W=H/'literature_work';W.mkdir(exist_ok=True)
sess=requests.Session();sess.proxies={'https':'http://127.0.0.1:7890','http':'http://127.0.0.1:7890'}
sess.headers.update({'User-Agent':'Mozilla/5.0'})
rows=[]
base='https://www.ieee-jas.net/';article='c9a9d9b1-cd64-4aa4-9ad4-90ae38070701'
u=base+'article/checkArticlePdf?id='+article
try:
 r=sess.post(u,timeout=30); rows.append(dict(key='Chen_access',url=u,status=r.status_code,text=r.text[:500]))
 d=r.json()
 if d.get('access')is not False and d.get('result')=='true':
  u=base+'article/exportPdf?id='+article+'&language=en';r=sess.get(u,timeout=40)
  row=dict(key='Chen_final',url=u,final_url=r.url,status=r.status_code,bytes=len(r.content),pdf=r.content.startswith(b'%PDF'))
  if row['pdf']:
   p=W/'Chen_final.pdf';p.write_bytes(r.content);row['sha256']=hashlib.sha256(r.content).hexdigest()
   subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  rows.append(row)
except Exception as e:rows.append(dict(key='Chen_access',error=str(e)))
(H/'literature_2_followup.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False))
