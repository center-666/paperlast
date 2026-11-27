"""Compare the public release to RUN, with an optional byte-level download check."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://api.github.com/repos/center-666/paperlast"
TAG = "joint-task4-namespace85-data"
HERE = Path(__file__).resolve().parent


def api(url):
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "paperlast-readback"})
    with urlopen(request, timeout=90) as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="append", default=[], metavar="FILE")
    args = parser.parse_args()
    run = json.loads((HERE/"task4_confirm/RUN.json").read_text())
    manifest = json.loads((HERE/"task4_confirm/ARCHIVE_MANIFEST.json").read_text())
    sizes = {Path(row["path"]).name: row["bytes"] for row in manifest["files"]}
    release = api(BASE+"/releases/tags/"+TAG)
    assert not release["draft"] and not release["prerelease"]
    assets = {row["name"]: row for row in api(release["assets_url"]+"?per_page=100")}
    assert set(assets) == set(run["archives"]) == set(sizes)
    for name, asset in assets.items():
        assert asset["state"] == "uploaded" and asset["size"] == sizes[name]
        assert asset["digest"] == "sha256:"+run["archives"][name]
    retrieved = {}
    for name in args.download:
        with urlopen(assets[name]["browser_download_url"], timeout=180) as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == run["archives"][name]
        retrieved[name] = actual
    print(json.dumps(dict(public_release=release["html_url"], release_id=release["id"],
                          archive_count=len(assets), bytes=sum(sizes.values()),
                          all_server_digests_match=True, anonymous_downloads=retrieved), indent=2))


if __name__ == "__main__":
    main()
