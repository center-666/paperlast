"""Build a path-and-hash inventory for the separately stored confirm archives."""
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "task4_confirm"
BACKUP_ROOT = r"D:\paperlast_task4_archive_20261127"
BASE_COMMIT = "599a2501c5a882d67497d76954c408be222ed1b3"
LOCAL_RESULTS_COMMIT = "68c6d4f5cc1abc6c4ded5374cd7b9e6e18e4a26f"
LOCK_SOURCE_COMMIT = "2f5a631d4a2343e76b13c8e70d34f47bed71ba89"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    run_path = OUT / "RUN.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    with (OUT / "ARCHIVE_SHA256SUMS.csv").open(encoding="utf-8-sig", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    run_archives = run["archives"]
    if len(inventory) != 48 or set(row["file"] for row in inventory) != set(run_archives):
        raise ValueError("Archive inventory does not match the frozen RUN.json.")
    files = []
    for row in inventory:
        name = row["file"]
        expected = run_archives[name]
        if row["sha256"].lower() != expected.lower() or row["matches_RUN"].lower() != "true":
            raise ValueError(f"Archive digest mismatch for {name}.")
        files.append({
            "path": f"task4_confirm/{name}",
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"].lower(),
        })
    protocol = HERE / "TASK4_CONFIRM_PROTOCOL.md"
    lock = HERE / "TASK4_CONFIRM_LOCK.json"
    backup_verify = OUT / "ARCHIVE_BACKUP_VERIFY.json"
    backup = json.loads(backup_verify.read_text(encoding="utf-8-sig"))
    if backup["status"] != "PASS_backup_copy_sha256" or backup["archives_verified"] != 48:
        raise ValueError("The second storage copy has no complete SHA-256 verification.")
    development_files = (
        "analyze_task3.py", "audit_task3.py",
        "task3_dev/absolute.csv", "task3_dev/paired.csv",
        "task3_dev/scenes.csv", "task3_dev/health.csv",
    )
    manifest = {
        "status": "PASS_manifest_matches_RUN_and_backup",
        "archive_set": "task4_confirm_namespace85",
        "namespace": run["identity"]["namespace"],
        "execution_commit": run["execution_commit"],
        "run_sha256": sha(run_path),
        "protocol_sha256": sha(protocol),
        "confirm_lock_sha256": sha(lock),
        "source_history": {
            "light_branch_base": BASE_COMMIT,
            "published_branch": "joint-watermark-fusion-results-20261127",
            "original_local_branch": "joint-watermark-fusion-audit-20261126",
            "original_results_commit": LOCAL_RESULTS_COMMIT,
            "protocol_and_lock_commit": LOCK_SOURCE_COMMIT,
            "relationship": "The original run and lock remain byte-identical. Their originating commits are recorded here because this light branch starts from the remote 599a250 ancestor and does not import the oversized local history.",
        },
        "verification_code": {
            "original_local_history_copy": "verify_task4_postrun_local_history.py",
            "original_local_history_sha256": sha(HERE / "verify_task4_postrun_local_history.py"),
            "published_clean_checkout_verifier": "verify_task4_postrun.py",
            "published_clean_checkout_sha256": sha(HERE / "verify_task4_postrun.py"),
            "adaptation": "Independent numerical replay is unchanged. The clean-checkout copy accepts the preserved source-history references in this manifest when the original local commits are not ancestors of the light branch.",
        },
        "task3_development_support": [
            {"path": path, "sha256": sha(HERE / path)} for path in development_files
        ],
        "archive_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "controlled_storage": {
            "verified_copy_root": BACKUP_ROOT,
            "archive_directory": f"{BACKUP_ROOT}\\task4_confirm",
            "verification_record": f"{BACKUP_ROOT}\\ARCHIVE_BACKUP_VERIFY.json",
            "sha256_inventory": "task4_confirm/ARCHIVE_SHA256SUMS.csv",
            "retrieval": "On the research workstation, copy the 48 files listed here from the verified D: copy into the checkout's task4_confirm directory. For external collaborators, the archive owner transfers the folder through an authorized channel. Recheck every digest against this manifest before running analysis.",
            "github_release_assets": {
                "status": "not_uploaded",
                "reason": "A GitHub Release upload client was unavailable in the current environment. The verified second copy and file-level retrieval manifest are retained.",
            },
        },
    }
    destination = OUT / "ARCHIVE_MANIFEST.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": manifest["status"],
        "archive_count": manifest["archive_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": sha(destination),
        "protocol_sha256": manifest["protocol_sha256"],
        "confirm_lock_sha256": manifest["confirm_lock_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
