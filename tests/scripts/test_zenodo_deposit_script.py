import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "paper" / "zenodo_deposit.py"
spec = importlib.util.spec_from_file_location("zenodo_deposit", SCRIPT)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules["zenodo_deposit"] = script
spec.loader.exec_module(script)


def test_a_placeholder_left_unfilled_is_refused(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text('{"description": "<p>A preprint (doi:[PAPER DOI]).</p>", "notes": "at tag [RELEASE TAG]"}')
    with pytest.raises(SystemExit, match=r"fill \[PAPER DOI\], \[RELEASE TAG\] first"):
        script.filled(path)
    path.write_text('{"description": "<p>A preprint (doi:10.5281/zenodo.1). <i>[sic]</i></p>"}')
    assert script.filled(path)["description"].startswith("<p>A preprint")


def test_a_new_version_is_a_draft_with_its_own_doi_reserved(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str, Any]] = []
    draft = {"id": 9, "links": {"bucket": "B", "html": "H"}, "metadata": {"title": "T", "doi": "10.5281/zenodo.1"}}

    def fake(method: str, url: str, token: str, body: Any = None, **_: Any) -> Any:
        calls.append((method, url, body))
        if url.endswith("/actions/newversion"):
            return {"links": {"latest_draft": "D"}}
        if method == "GET":
            return draft
        return draft | {"metadata": body["metadata"] | {"prereserve_doi": {"doi": "10.5281/zenodo.2"}}}

    monkeypatch.setattr(script, "request", fake)
    made = script.new_version(1, "token")
    assert made == {"id": 9, "doi": "10.5281/zenodo.2", "bucket": "B", "html": "H", "version_of": 1}
    assert calls[0] == ("POST", f"{script.API}/deposit/depositions/1/actions/newversion", None)
    assert calls[2][2]["metadata"] == {"title": "T", "prereserve_doi": True}  # the old DOI dropped, published never
    assert all(method != "POST" or url.endswith("/actions/newversion") for method, url, _ in calls)
    reserved = draft | {"metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.3"}}}

    def ready(method: str, url: str, token: str, body: Any = None, **_: Any) -> Any:
        return reserved if method == "GET" else {"links": {"latest_draft": "D"}}

    monkeypatch.setattr(script, "request", ready)
    assert script.new_version(1, "token")["doi"] == "10.5281/zenodo.3"  # reserved already: no PUT


def test_a_dropped_connection_is_retried_and_a_file_still_failing_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "part-00"
    path.write_bytes(b"data")
    local = script.md5(path)
    tries: list[int] = []

    def flaky(method: str, url: str, token: str, data: Any = None, **_: Any) -> Any:
        tries.append(len(data.read()))
        if len(tries) < 3:
            raise BrokenPipeError(32, "Broken pipe")
        return {"checksum": f"md5:{local}"}

    monkeypatch.setattr(script, "request", flaky)
    assert script.send(path, "B", "token", {}, tries=3, pause=0) == "part-00: uploaded, 0.00 GB, MD5 checked"
    assert tries == [4, 4, 4]  # each try sends the whole file from its start
    assert script.send(path, "B", "token", {"part-00": local}, pause=0) == "part-00: there already"

    def down(method: str, url: str, token: str, **_: Any) -> Any:
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(script, "request", down)
    assert script.send(path, "B", "token", {}, tries=2, pause=0).startswith("part-00: FAILED after 2 tries")
