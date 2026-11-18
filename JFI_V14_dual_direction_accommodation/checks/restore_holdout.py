"""Restore exact NumPy archives from the checked-in binary parts, if needed."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def restore(root):
    manifest = json.loads((root/"archive_parts.json").read_text(encoding="utf-8"))
    for name,spec in manifest["files"].items():
        target = root/name
        if target.is_file():
            if sha256(target.read_bytes()).hexdigest()!=spec["sha256"]:
                raise ValueError(f"The existing archive has a different SHA-256: {target}")
            print(f"Verified {target.name}")
            continue
        digest = sha256()
        count = 0
        partial = None
        try:
            with NamedTemporaryFile(dir=root,suffix=".npz.partial",delete=False) as combined:
                partial = Path(combined.name)
                for entry in spec["parts"]:
                    data = (root/entry["path"]).read_bytes()
                    if len(data)!=entry["bytes"] or sha256(data).hexdigest()!=entry["sha256"]:
                        raise ValueError(f"Archive part differs: {entry['path']}")
                    combined.write(data)
                    digest.update(data)
                    count += len(data)
            if digest.hexdigest()!=spec["sha256"] or count!=spec["bytes"]:
                raise ValueError(f"Restored archive differs: {target}")
            if target.exists():
                raise FileExistsError(target)
            partial.rename(target)
        finally:
            if partial is not None and partial.is_file():
                partial.unlink()
        print(f"Restored {target.name}")


if __name__=="__main__":
    restore(Path(__file__).resolve().parents[1]/"results/holdout")
