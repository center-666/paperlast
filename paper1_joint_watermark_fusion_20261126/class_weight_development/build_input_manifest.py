"""Create a SHA-256 inventory for the twelve frozen development evaluation archives."""
import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent.parent
DEV=Path(r"C:\Users\zhongxin\shixidoc\paperlast-joint-audit\paper1_joint_watermark_fusion_20261126\task3_dev")
RUN=DEV/"RUN.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    run=json.loads(RUN.read_text(encoding="utf-8"))
    files=[]
    for index in range(12):
        name=f"config{index:02}.npz"
        path=DEV/name
        expected=run["archives"][name]
        actual=sha(path)
        if actual!=expected:
            raise ValueError(f"Task-3 archive checksum mismatch for {name}")
        files.append({"name":name,"bytes":path.stat().st_size,"sha256":actual})
    manifest={
        "status":"PASS_task3_development_archive_manifest",
        "source_run_sha256":sha(RUN),
        "source_run_identity":run["identity"],
        "archive_directory":"task3_dev",
        "archive_count":len(files),
        "total_bytes":sum(row["bytes"] for row in files),
        "files":files,
        "purpose":"Task-3 development configuration archives. Purpose 5 contains the previously evaluated paired development paths. Purpose 6 contains the registered independent-within-run healthy paths.",
        "release_tag":"joint-task3-development-ns83-data",
        "release_url":"https://github.com/center-666/paperlast/releases/tag/joint-task3-development-ns83-data",
        "release_id":460415994,
        "public_release_verification":"class_weight_development/DEVELOPMENT_RELEASE_VERIFY.json"
    }
    target=HERE/"class_weight_development"/"INPUT_MANIFEST.json"
    target.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:manifest[k] for k in ("status","archive_count","total_bytes","source_run_sha256","release_tag")},indent=2))


if __name__=="__main__":
    main()
