"""Source mapping for eight real review items; no manuscript/library mutation."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;P=H.parent;ROOT=H.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=ROOT/'研究规划/第一篇小论文/secure distributed safety.tex'
attachment=Path('E:/Apps/Codex/data/attachments/b64223d7-dfc7-4003-b7bf-761f7c412ced/已粘贴的文本.txt')
lines=attachment.read_text(encoding='utf-8-sig').splitlines()
items=[('R1.2',51,['30','31','32','33','34'],'independent contribution/strong watermark comparison'),
       ('R2.1',67,['9','10','11','12'],'implementable detector, not distribution-oracle KL'),
       ('R2.2',69,['30','53'],'specific limitation and evidence beyond existing watermarking'),
       ('R2.3',71,['54'],'fair experiments and full event evaluation'),
       ('R3.5',86,['30','34'],'difference from prior multiplicative/dynamic schemes'),
       ('R4.3',101,['31','32'],'encoder/decoder synchronization and practical conditions'),
       ('R4.6',107,['18','19','32','54'],'early attacked RMSE and watermark intensity'),
       ('R5.3',113,['13','53'],'new result versus standard derivations')]
refs=[P/'web_revision_control_20261112/29条意见_本机状态与剩余验收.md',
      P/'web_revision_control_20261112/理论衔接与检测定义_待审补充.md',
      P/'水印研究收束_参数知情边界与简单强基线_20261105.md',
      P/'adversarial_audit_20261113/投稿独立性评估.md',
      P/'original_estimator_gate_20261114/报告四_文献独立性与立项决定.md',
      ROOT/'参考文献/文献库.json']
for tag,n,_,_ in items:
 assert lines[n-1].strip(),(tag,n)
scope=json.loads((H/'frozen_scope.json').read_text(encoding='utf-8'))
assert sha(ROOT/'参考文献/文献库.json')==scope['protected_files']['参考文献\\文献库.json']
out=dict(date='2026-11-14',task='A/E reviewer mapping and literature boundary',
    real_review_items=8,original_comment_file=str(attachment),original_comment_sha256=sha(attachment),
    original_TeX=dict(file=str(original.relative_to(ROOT)),sha256=sha(original)),
    items=[dict(id=tag,comment_line=n,line_text_sha256=hashlib.sha256(lines[n-1].encode('utf-8')).hexdigest(),
                original_equations=eqs,issue=issue,status='requires performance/independence validation; not closed by rho norm bound')
           for tag,n,eqs,issue in items],
    reused_sources=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p))for p in refs],
    literature_reading='reused recorded designated-page evidence; no new PDF/full-proof claim',
    literature_library_unchanged=True,new_downloads=0,new_literature_records=0,
    exact_conclusions=['for fixed predetermined a and same M, rho does not change untouched decoded parity data or detector law',
       'current induced gain cap is not full finite-event safety',
       'rho(M) innovation weighting changes nominal estimator despite exact decode',
       'innovation amplitude clipping and secret-conditioned operator capping are different operators'],
    contribution_status='not passed; performance screening pending; no pre-approval of A',
    external_independent_peer_review=False)
(H/'reviewer_mapping.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'review_items':len(items),'old_library_unchanged':True,'new_PDFs':0},ensure_ascii=False))
