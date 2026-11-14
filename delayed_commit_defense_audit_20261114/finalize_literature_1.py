"""Attach accurate read scopes/metadata; no reference library writes."""
from pathlib import Path
import json,hashlib,requests
H=Path(__file__).resolve().parent
path=H/'literature_1_sources.json';rows=json.loads(path.read_text(encoding='utf-8'))
metadata={
 'Li2023':dict(title='Secure State Estimation with Asynchronous Measurements against Malicious Measurement-Data and Time-Stamp Manipulation',
 authors=['Zishuo Li','Anh Tung Nguyen','Andre M. H. Teixeira','Yilin Mo','Karl H. Johansson'],
 doi='10.1109/CDC49753.2023.10383571',year=2023,venue='CDC',printed_pages='7073-7080',
 read_pdf_pages=[2,3,4,5,6,8],visual_pdf_pages=[3,5],
 exact_read_scope='targeted model, attack definition, buffering paragraph, assumptions1-5, equations12-17, Algorithm1, Theorems2-3 and printed references; not all referenced proof chains',
 core_theorems=['Theorem2 at PDF5-6/printed7077-7078','Theorem3 at PDF6/printed7078'],
 buffer_evidence='PDF3/printed7075; length d, timestamp ordering and discard over-d late data, not attack-evidence commit',
 new_candidate_fully_covered=False,source_read_status='primary fulltext acquired; specified pages actually read'),
 'Cheng2023':dict(title='Security Analysis for Dynamic State Estimation of Power Systems With Measurement Delays',
 authors=['Zhijian Cheng','Hongru Ren','Jiahu Qin','Renquan Lu'],doi='10.1109/TCYB.2021.3108884',year=2023,online_year=2021,
 venue='IEEE Transactions on Cybernetics',volume=53,issue=4,printed_pages='2087-2096',read_pdf_pages=[],
 core_theorems=None,source_read_status='core fulltext missing; publisher202 empty and JavaScript verification, indexed author abstract is a lead only',
 equivalence_or_novelty_clearance=False),
 'Xu2023':dict(title='Integrity attacks on remote estimation with spatial-temporal information sources',
 authors=['Haoyuan Xu','Yake Yang','Jun Shang','Jun Fu','Yuzhe Li'],doi='10.1016/j.automatica.2023.111172',year=2023,
 venue='Automatica',volume=155,article_number='111172',read_pdf_pages=[],core_theorems=None,
 source_read_status='core fulltext missing; publisher abstract/snippets and author publication list used only for identification',
 equivalence_or_novelty_clearance=False)}
for row in rows:
 if row['key']in metadata:
  row.update(metadata[row['key']]);row.update(unique_library_id_at_initial_lookup=None,initial_library_lookup='doi/title case-insensitive search found no record; root handles any later registration')
url='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8815824'
try:
 r=requests.get(url,timeout=20,proxies={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'},headers={'User-Agent':'Mozilla/5.0'})
 row=dict(key='Li_reference10_Uribe2020',doi='10.1109/TAC.2019.2937850',url=url,status=r.status_code,final_url=r.url,
          bytes=len(r.content),is_pdf=r.content.startswith(b'%PDF'),content_type=r.headers.get('Content-Type'))
 if row['is_pdf']:
  p=H/'literature_work/Uribe2020_primary.pdf';p.write_bytes(r.content);row.update(file=str(p),sha256=hashlib.sha256(r.content).hexdigest(),read_status='obtained but not yet read')
 else:row.update(read_status='only citation traced; no original PDF obtained; WSEAS2019 is not substituted')
 rows.append(row)
except Exception as e:rows.append(dict(key='Li_reference10_Uribe2020',doi='10.1109/TAC.2019.2937850',url=url,is_pdf=False,error=str(e),read_status='original fulltext not read'))
path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(key=x['key'],is_pdf=x['is_pdf'],status=x.get('status'))for x in rows],ensure_ascii=False))
