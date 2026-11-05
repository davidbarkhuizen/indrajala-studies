"""scripts/paper/certnote_addendum.py (the certificate note's D15): every file a reader needs, each hash checked."""

import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "paper"))
import certnote_addendum as ca  # scripts/ isn't a package


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "checkout"
    note = base / ca.NOTE
    for name in ca.RESULTS:
        (note / name).parent.mkdir(parents=True, exist_ok=True)
        (note / name).write_text(name)
    job = {"id": "aa-1.0", "network": "data/w/n.network.json", "reproduce": "data/w/n.certificate.json"}
    (note / "recertify" / "jobs.jsonl").write_text(json.dumps(job) + "\n")
    (note / "recertify" / "tables").mkdir()
    (note / "recertify" / "tables" / "aa-1.0.npz").write_bytes(b"table")
    (base / "data" / "w").mkdir(parents=True)
    (base / "data" / "w" / "n.network.json").write_text("{}")
    (base / "data" / "w" / "n.certificate.json").write_text("{}")
    return base


def test_a_built_addendum_checks_and_holds_each_file_at_its_checkout_path(root: Path, tmp_path: Path):
    out = tmp_path / "out"
    readme = tmp_path / "readme.md"
    readme.write_text("read me")
    ca.build(root, out, readme)
    assert ca.check(out)
    with tarfile.open(out / "certnote-networks.tar.gz") as archive:
        assert sorted(archive.getnames()) == ["data/w/n.certificate.json", "data/w/n.network.json"]
    with tarfile.open(out / "certnote-tables.tar.gz") as archive:
        assert archive.getnames() == ["data/certificate-note/recertify/tables/aa-1.0.npz"]
    assert "README.md" in (out / "MANIFEST.sha256").read_text()


def test_a_missing_file_stops_the_build(root: Path, tmp_path: Path):
    (root / "data" / "w" / "n.network.json").unlink()
    with pytest.raises(FileNotFoundError):
        ca.build(root, tmp_path / "out", None)


def test_a_changed_archive_or_a_missing_entry_fails_the_check(root: Path, tmp_path: Path):
    out = tmp_path / "out"
    ca.build(root, out, None)
    manifest = (out / "MANIFEST.sha256").read_text()
    (out / "MANIFEST.sha256").write_text(manifest + f"{'0' * 64}  data/w/absent.json\n")
    assert not ca.check(out)
    (out / "MANIFEST.sha256").write_text(manifest)
    with (out / "certnote-results.tar.gz").open("ab") as f:
        f.write(b"x")
    assert not ca.check(out)


def test_a_split_addendum_checks_and_its_parts_rejoin_to_the_whole(root: Path, tmp_path: Path):
    # the parts' size shrunk so every archive splits; the README replaced, its line with it
    out = tmp_path / "out"
    readme = tmp_path / "readme.md"
    readme.write_text("read me")
    ca.build(root, out, readme)
    before = (out / "MANIFEST.sha256").read_text().splitlines()
    wholes = {p.name: p.read_bytes() for p in out.glob("*.tar.gz")}
    readme.write_text("read me, then cat the parts")
    ca.split(out, readme, part=64)

    assert not list(out.glob("*.tar.gz"))
    for name, data in wholes.items():
        parts = sorted(out.glob(f"{name}.part-*"))
        assert len(parts) > 1 and b"".join(p.read_bytes() for p in parts) == data
        assert (out / "whole" / name).read_bytes() == data
    after = (out / "MANIFEST.sha256").read_text().splitlines()
    assert [line for line in after if "README.md" not in line and ".part-" not in line] == [
        line for line in before if "README.md" not in line
    ]
    assert len([line for line in after if ".part-" in line]) == len(list(out.glob("*.part-*")))
    assert (out / "README.md").read_text() == "read me, then cat the parts"
    assert ca.check(out)

    # as a reader downloads it: the parts alone, no whole/
    for path in (out / "whole").iterdir():
        path.unlink()
    assert ca.check(out)
    min(out.glob("certnote-tables.tar.gz.part-*")).write_bytes(b"x")
    assert not ca.check(out)
