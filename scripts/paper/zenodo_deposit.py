#!/usr/bin/env python3
"""
Paper 1's datapack to Zenodo (the paper workplan's D38), through Zenodo's deposit API; never
publishes. The owner reviews the draft in the browser and presses Publish, the irreversible step.

The token is a Zenodo personal access token with the scopes deposit:write and deposit:actions, read
from `~/.zenodo-token` (or `--token-file`); it is never printed or logged. The draft's id, DOI and
file bucket are kept in DIR's `zenodo.json`, so each command resumes the same draft.

`zenodo_deposit.py reserve DIR` creates a draft with a pre-reserved DOI and prints the DOI, to be
filled into the README, the paper and the letters before the upload. `zenodo_deposit.py metadata DIR`
sets the draft's metadata from `--metadata FILE`, by default indrajala-research's
docs/datapack/zenodo-metadata.json in the sibling clone (the datapack's; the preprint's is beside it). `zenodo_deposit.py upload DIR`
uploads every file at DIR's top level (but `zenodo.json`; `datapack.py split` leaves the large
archives in parts there, the wholes in `whole/`), `--connections` at once, the largest first,
skipping those already there with the same MD5, and checks each against the checksum Zenodo reports. `zenodo_deposit.py status DIR` compares the
draft's files with DIR's.

`zenodo_deposit.py edit DIR [--dry-run]` changes a published record's metadata, no new version (its DOI
and files stay): it lists the fields of the metadata file that differ from the record's, then
unlocks the record for editing and sets them; the owner reviews the edit and presses Publish (or
Discard). With `--dry-run` it only lists them. `metadata` and `edit` refuse a file with a placeholder
left unfilled (`[RELEASE TAG]`, `[PAPER DOI]`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

API = "https://zenodo.org/api"
# the datapack record's metadata, kept with the paper in the sibling indrajala-research clone
METADATA = Path(__file__).resolve().parents[3] / "indrajala-research" / "docs" / "datapack" / "zenodo-metadata.json"


PLACEHOLDER = re.compile(r"\[[A-Z][A-Z ]*\]")  # a field still to fill, e.g. [RELEASE TAG], [PAPER DOI]


def filled(path: Path) -> dict[str, Any]:
    """The metadata file's fields, refused while any placeholder is left unfilled."""
    text = path.read_text()
    if left := sorted(set(PLACEHOLDER.findall(text))):
        raise SystemExit(f"{path}: fill {', '.join(left)} first")
    return json.loads(text)


def request(method: str, url: str, token: str, body: Any = None, data: Any = None, size: int | None = None) -> Any:
    """One API call; a JSON body, or a file's bytes streamed (`data`, `size`)."""
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif data is not None:
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(size)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=3600) as response:
        text = response.read().decode()
    return json.loads(text) if text else None


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state(out: Path) -> dict[str, Any]:
    path = out / "zenodo.json"
    if not path.exists():
        raise SystemExit(f"{path}: no draft yet; run reserve first")
    return json.loads(path.read_text())


def local_files(out: Path) -> list[Path]:
    return sorted(p for p in out.iterdir() if p.is_file() and p.name != "zenodo.json")


def remote_files(draft: dict[str, Any], token: str) -> dict[str, str]:
    """The draft's files: name -> MD5."""
    files = request("GET", f"{API}/deposit/depositions/{draft['id']}/files", token)
    return {f["filename"]: f["checksum"].removeprefix("md5:") for f in files}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".zenodo-token")
    parser.add_argument("--connections", type=int, default=12, help="files uploaded at once")
    parser.add_argument("--metadata", type=Path, default=METADATA, help="the record's metadata (metadata, edit)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reserve", "metadata", "upload", "status"):
        commands.add_parser(name).add_argument("dir", type=Path)
    edit = commands.add_parser("edit", help="a published record's metadata from --metadata; never publishes")
    edit.add_argument("dir", type=Path)
    edit.add_argument("--dry-run", action="store_true", help="only list the fields that would change")
    args = parser.parse_args(argv)
    token = args.token_file.read_text().strip()
    out: Path = args.dir

    if args.command == "reserve":
        if (out / "zenodo.json").exists():
            raise SystemExit(f"{out / 'zenodo.json'}: a draft exists already ({state(out)['doi']})")
        made = request("POST", f"{API}/deposit/depositions", token, body={"metadata": {"prereserve_doi": True}})
        draft = {
            "id": made["id"],
            "doi": made["metadata"]["prereserve_doi"]["doi"],
            "bucket": made["links"]["bucket"],
            "html": made["links"]["html"],
        }
        (out / "zenodo.json").write_text(json.dumps(draft, indent=1) + "\n")
        print(f"draft {draft['id']}: DOI {draft['doi']} (reserved, not yet registered); {draft['html']}")
        return 0
    draft = state(out)
    if args.command == "metadata":
        metadata = filled(args.metadata) | {"prereserve_doi": True}
        request("PUT", f"{API}/deposit/depositions/{draft['id']}", token, body={"metadata": metadata})
        print(f"draft {draft['id']}: metadata set; review it at {draft['html']}")
        return 0
    if args.command == "edit":
        wanted = filled(args.metadata)
        url = f"{API}/deposit/depositions/{draft['id']}"
        record = request("GET", url, token)
        current = record["metadata"]
        changed = sorted(key for key in wanted if current.get(key) != wanted[key])
        print(f"record {draft['id']} ({record['state']}): {', '.join(changed) or 'nothing'} differs")
        if args.dry_run or not changed:
            return 0
        if record["state"] == "done":  # published: unlock it for editing first
            request("POST", f"{url}/actions/edit", token)
        request("PUT", url, token, body={"metadata": current | {key: wanted[key] for key in changed}})
        print(
            f"record {draft['id']}: {', '.join(changed)} set, not published; review and press Publish at {draft['html']}"
        )
        return 0
    there = remote_files(draft, token)
    if args.command == "status":
        for path in local_files(out):
            mark = "uploaded" if there.get(path.name) == md5(path) else "missing or differs"
            print(f"{path.name}: {mark}")
        return 0

    def send(path: Path) -> str:
        local = md5(path)
        if there.get(path.name) == local:
            return f"{path.name}: there already"
        with path.open("rb") as f:
            made = request("PUT", f"{draft['bucket']}/{path.name}", token, data=f, size=path.stat().st_size)
        reported = made["checksum"].removeprefix("md5:")
        if reported != local:
            raise SystemExit(f"{path.name}: Zenodo reports MD5 {reported}, the file's is {local}")
        return f"{path.name}: uploaded, {path.stat().st_size / 1e9:.2f} GB, MD5 checked"

    with ThreadPoolExecutor(args.connections) as pool:  # Zenodo's connections are slow one by one
        for line in pool.map(send, sorted(local_files(out), key=lambda p: -p.stat().st_size)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
