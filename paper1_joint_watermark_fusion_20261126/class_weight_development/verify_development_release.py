"""Verify all published task-3 archive digests and download one public sample."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

BASE="https://api.github.com/repos/center-666/paperlast"
TAG="joint-task3-development-ns83-data"
HERE=Path(__file__).resolve().parent.parent


def get_json(url):
 request=Request(url,headers={"Accept":"application/vnd.github+json","User-Agent":"paperlast-development-audit"})
 with urlopen(request,timeout=90) as response: return json.load(response)


def main():
 manifest=json.loads((HERE/"class_weight_development/INPUT_MANIFEST.json").read_text(encoding="utf-8"))
 release=get_json(BASE+"/releases/tags/"+TAG)
 assert not release["draft"] and not release["prerelease"]
 assets={row["name"]:row for row in get_json(release["assets_url"]+"?per_page=100")}
 expected={row["name"]:row for row in manifest["files"]}
 assert set(assets)==set(expected) and len(assets)==12
 for name,row in assets.items():
  source=expected[name]
  assert row["state"]=="uploaded" and row["size"]==source["bytes"]
  assert row.get("digest")=="sha256:"+source["sha256"]
 sample="config02.npz"
 request=Request(assets[sample]["browser_download_url"],headers={"User-Agent":"paperlast-development-audit"})
 with urlopen(request,timeout=180) as response:
  digest=hashlib.file_digest(response,"sha256").hexdigest()
 assert digest==expected[sample]["sha256"]
 result={"status":"PASS_public_task3_development_assets","tag":TAG,"release_id":release["id"],
         "release_url":release["html_url"],"asset_count":len(assets),
         "total_bytes":sum(row["bytes"] for row in expected.values()),
         "all_remote_size_sha256_match":True,"anonymous_download_test":{sample:digest},
         "source_RUN_sha256":manifest["source_run_sha256"]}
 target=HERE/"class_weight_development"/"DEVELOPMENT_RELEASE_VERIFY.json"
 target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
 print(json.dumps(result,indent=2))


if __name__=="__main__": main()
