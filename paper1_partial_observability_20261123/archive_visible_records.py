"""Archive only user text and assistant public text from the research turn.

No system/developer messages, reasoning, tool payloads or credentials are read
into the output. The caller explicitly identifies the parent session log.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--v2-attachment',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;v3=here.parent/'paper1_sscd_commonrandom_20261122'
    messages=[]
    with a.session.open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line);payload=item.get('payload',{})
            if item.get('type')!='response_item' or payload.get('type')!='message':continue
            role=payload.get('role');channel=payload.get('channel')
            if role not in ('user','assistant'):continue
            if role=='assistant' and channel not in ('commentary','final'):continue
            content=payload.get('content',[])
            text='\n'.join(part.get('text','') for part in content if part.get('type') in ('input_text','output_text','text'))
            if role=='user' and text.lstrip().startswith('<environment_context>'):continue
            if text:messages.append({'timestamp':item.get('timestamp'),'role':role,'channel':channel,'text':text})
    matches=[i for i,m in enumerate(messages) if m['role']=='user'
             and '# **第一篇论文：SSCD 的新一轮理论与算法推进**' in m['text']]
    if not matches:raise ValueError('Exact visible user-proposal heading was not found')
    start=matches[-1];proposal=messages[start]['text']
    (v3/'user_proposal.md').write_text(proposal+'\n',encoding='utf-8',newline='\n')
    attachment=v3/'user_v2_review_attachment.txt';shutil.copyfile(a.v2_attachment,attachment)
    records=messages[start:]
    (here/'visible_dialogue.json').write_text(json.dumps({'scope':'Visible user messages and public assistant text from the v3 proposal onward, up to this archive operation; excludes reasoning, all tool payloads and all higher-priority instructions.',
        'message_count':len(records),'messages':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    metadata={'status':'ARCHIVED_VISIBLE_TEXT_ONLY','user_proposal_utf8_sha256':hashlib.sha256(proposal.encode('utf-8')).hexdigest(),
              'proposal_file_sha256':hashlib.sha256((v3/'user_proposal.md').read_bytes()).hexdigest(),
              'v2_attachment_sha256':hashlib.sha256(attachment.read_bytes()).hexdigest(),
              'v2_attachment_original_bytes_preserved':a.v2_attachment.read_bytes()==attachment.read_bytes(),
              'visible_message_count':len(records),'scope':'Textual user proposal canonicalized to LF plus final newline; attachment preserves original bytes; no hidden/tool text archived.'}
    (here/'conversation_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
