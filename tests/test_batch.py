import asyncio
import datetime
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from indrajala_studies import batch as b

# A stub run (fast, deterministic): seed s even certifies at both verdicts at step s, an odd seed never;
# an epoch a step, the budget 6 epochs ("spent", its row logging the untrained epoch too, as a run's
# does); --cutoff-epochs T cuts an attempt not certified by T epochs trained ("cut").
STUB = """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
seed = int(args[args.index("--runs") + 1].split(":")[1])
out = Path(args[args.index("--out") + 1]) / f"seed{seed}"
cutoff = int(args[args.index("--cutoff-epochs") + 1]) if "--cutoff-epochs" in args else None
sleep = float(os.environ.get("STUB_SLEEP", "0.01"))
out.mkdir(parents=True, exist_ok=True)
with (out / "log.jsonl").open("w") as log:
    for step in range(1, 7):
        time.sleep(sleep)
        won = seed % 2 == 0 and step == seed
        end = "passed" if won else "spent" if step == 6 else "cut" if cutoff is not None and step >= cutoff else "epoch"
        row = {"step": step, "end": end, "epochs": 2 if end == "spent" else 1, "state_share": 1.0,
               "seconds": float(step), "certified": won, "certificates": {"strict": {"certified": won}}}
        log.write(json.dumps(row) + "\\n"); log.flush()
        if end != "epoch":
            break
print(f"seed {seed}: step {step}, {end}")
"""


SAST = datetime.timezone(datetime.timedelta(hours=2))  # the hosts' local hours


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def worker(tmp_path: Path, name: str, slots: int, env: dict[str, str] | None = None) -> b.Worker:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    return b.Worker(tmp_path / name, slots, sys.executable, tmp_path, script=str(stub), host=name, env=env, poll=0.05)


def run_batch(tmp_path: Path, batch: b.Batch, *workers: b.Worker) -> b.State:
    directory = tmp_path / "batch"
    b.register(directory, batch)
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(asyncio.gather(*(w.serve(port, retry=0.1) for w in workers)), timeout=60)
        await asyncio.wait_for(serving, timeout=10)

    asyncio.run(go())
    return b.State.read(directory)


def test_attempts_read_a_seed_spec_with_each_backdrops_cutoff():
    made = b.attempts("s3:40-42,mtf4:7", ["continuous"], {"s3": 137})
    assert [(a.id, a.seed, a.cutoff) for a in made] == [
        ("s3-40", 40, 137),
        ("s3-41", 41, 137),
        ("s3-42", 42, 137),
        ("mtf4-7", 7, None),
    ]
    with pytest.raises(AssertionError, match="registered cells"):
        b.Batch("x", "measure", b.attempts("s3:1", ["nope"]))
    with pytest.raises(AssertionError, match="find needs k"):
        b.Batch("x", "find", made)


def test_a_batch_of_several_cells_names_each_attempt_by_its_cell_and_runs_them_in_turn(tmp_path: Path):
    made = b.attempts("s3:1-2", ["continuous", "approximation"])
    assert [(a.id, a.cell) for a in made] == [
        ("continuous.s3-1", "continuous"),
        ("continuous.s3-2", "continuous"),
        ("approximation.s3-1", "approximation"),
        ("approximation.s3-2", "approximation"),
    ]
    batch = b.Batch("t", "find", made, k=1)
    assert batch.cells == ("continuous", "approximation") and b.Batch.from_json(batch.to_json()) == batch
    state = run_batch(tmp_path, batch, worker(tmp_path, "w", 1))  # stub: seed 2 certifies, each cell's own k
    assert set(state.ended) == {a.id for a in made}  # seed 1 fails, so each cell runs both
    assert b.report(tmp_path / "batch")[1:] == [
        "  continuous s3: 2 of 2 ended, 1 certified (0.50, 0.09-0.91)",
        "  approximation s3: 2 of 2 ended, 1 certified (0.50, 0.09-0.91)",
    ]
    assert "--state-bound" in worker(tmp_path, "w", 1).command(made[0])
    assert "--state-bound" not in worker(tmp_path, "w", 1).command(made[2])


def test_a_ledger_naming_one_cell_for_its_batch_is_read_as_each_attempts():
    row: dict[str, Any] = {"type": "batch", "v": 1, "name": "t1", "mode": "measure", "cell": "approximation"}
    row["attempts"] = [{"id": "d5-sum-300", "backdrop": "d5-sum", "seed": 300, "cutoff": None}]
    assert b.Batch.from_json(row).attempts == (b.Attempt("d5-sum-300", "approximation", "d5-sum", 300),)


def test_a_measure_batch_runs_every_attempt_across_two_workers_and_logs_each_end(tmp_path: Path):
    batch = b.Batch("m", "measure", b.attempts("s3:1-6", ["continuous"]))
    state = run_batch(tmp_path, batch, worker(tmp_path, "w1", 2), worker(tmp_path, "w2", 1))
    assert state.done and not state.assigned and set(state.ended) == {f"s3-{s}" for s in range(1, 7)}
    assert {i: s["success"] for i, s in state.ended.items()} == {
        "s3-1": None,
        "s3-2": 2,
        "s3-3": None,
        "s3-4": 4,
        "s3-5": None,
        "s3-6": 6,
    }
    assert all(s["end"] == "spent" for i, s in state.ended.items() if int(i.split("-")[1]) % 2)
    hosts = [json.loads(line)["host"] for line in (tmp_path / "batch" / b.LEDGER).read_text().splitlines()[1:-1]]
    assert {"w1", "w2"} <= set(hosts)  # both workers ran attempts
    assert b.report(tmp_path / "batch")[1].startswith("  s3: 6 of 6 ended, 3 certified (0.50,")
    assert b.runs_of(list(state.ended.values())).count((None, 6.0, True)) == 3


def test_a_find_batch_stops_starting_attempts_once_k_have_succeeded(tmp_path: Path):
    batch = b.Batch("f", "find", b.attempts("s3:1-5", ["continuous"]), k=1)
    state = run_batch(tmp_path, batch, worker(tmp_path, "w", 1))
    assert set(state.ended) == {"s3-1", "s3-2"} and state.ended["s3-2"]["success"] == 2  # 3-5 never start


def test_a_cut_attempt_ends_cut_and_unsuccessful(tmp_path: Path):
    batch = b.Batch("c", "measure", b.attempts("s3:2,s3:4", ["continuous"], {"s3": 3}))
    state = run_batch(tmp_path, batch, worker(tmp_path, "w", 2))
    assert state.ended["s3-2"]["success"] == 2 and state.ended["s3-4"] | {"peak_pss_mb": None} == {
        "success": None,
        "step": 3,
        "epochs": 3,
        "end": "cut",
        "seconds": 3.0,
        "peak_pss_mb": None,
    }


def test_the_controller_rebuilds_from_its_ledger_and_requeues_a_hosts_lost_attempt(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", b.attempts("s3:1-3", ["continuous"])))
    controller = b.Controller(directory)
    controller.log({"type": "assigned", "id": "s3-1", "host": "h"})
    controller.log({"type": "ended", "id": "s3-1", "host": "h", "summary": b.summary(tmp_path / "none")})
    controller.log({"type": "assigned", "id": "s3-2", "host": "h"})
    rebuilt = b.Controller(directory).state  # a restarted controller
    assert set(rebuilt.ended) == {"s3-1"} and rebuilt.assigned == {"s3-2": "h"}
    assert rebuilt.next_attempt() == b.Attempt("s3-3", "continuous", "s3", 3)

    async def hello() -> None:  # h comes back without s3-2: lost, so it is queued again
        port = free_port()
        again = b.Controller(directory)
        serving = asyncio.create_task(again.serve(port))
        await asyncio.sleep(0.1)
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        await b.send(writer, {"type": "hello", "host": "h", "slots": 1, "running": []})
        await asyncio.sleep(0.2)
        writer.close()
        serving.cancel()
        assert again.state.next_attempt() == b.Attempt("s3-2", "continuous", "s3", 2)

    asyncio.run(hello())
    assert '"type": "lost"' in (directory / b.LEDGER).read_text()


def test_a_restarted_worker_re_adopts_its_running_attempts_from_their_pid_files(tmp_path: Path):
    first = worker(tmp_path, "w", 1, env={"STUB_SLEEP": "2"})
    running = first.launch(b.Attempt("s3-1", "continuous", "s3", 1))
    assert running.pid > 0
    again = worker(tmp_path, "w", 1)  # the daemon restarted while its attempt runs
    assert list(again.running) == ["s3-1"] and again.running["s3-1"].pid == running.pid
    assert first.command(running.attempt)[3:5] == ["--runs", "word-problem-s3:1"]
    os.killpg(running.pid, signal.SIGTERM)  # its own session
    deadline = time.monotonic() + 5
    while b.alive(running.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not b.alive(running.pid) and not worker(tmp_path, "w", 1).running  # ended: nothing to adopt


class Dropped:
    """A connection whose messages go nowhere (the worker connected to itself while the tunnel was down)."""

    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


def test_an_end_reported_into_a_dropped_connection_is_sent_again_and_acknowledged(tmp_path: Path):
    batch = b.Batch("r", "measure", b.attempts("s3:2", ["continuous"]))
    directory = tmp_path / "batch"
    b.register(directory, batch)
    b.Controller(directory).log({"type": "assigned", "id": "s3-2", "host": "w"})
    first = worker(tmp_path, "w", 1)
    running = first.launch(b.Attempt("s3-2", "continuous", "s3", 2))
    deadline = time.monotonic() + 10
    while b.alive(running.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    asyncio.run(first.look(Dropped()))  # type: ignore[arg-type]  # its `ended` lost
    assert (tmp_path / "w" / "s3-2" / "ended.json").exists() and not first.running
    again = worker(tmp_path, "w", 1)
    assert [d.name for d in again.unacked()] == ["s3-2"]
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(again.serve(port, retry=0.1), timeout=30)
        await asyncio.wait_for(serving, timeout=10)

    asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert '"type": "lost"' not in ledger and b.State.read(directory).ended["s3-2"]["success"] is not None
    assert (tmp_path / "w" / b.ACKED / "s3-2").exists() and not again.unacked()


def test_an_end_sent_again_or_from_another_host_is_acknowledged_and_not_logged(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", b.attempts("s3:1-2", ["continuous"])))
    controller = b.Controller(directory)
    controller.log({"type": "assigned", "id": "s3-1", "host": "h"})
    controller.log({"type": "ended", "id": "s3-1", "host": "h", "summary": b.summary(tmp_path / "none")})
    controller.log({"type": "assigned", "id": "s3-2", "host": "h"})
    port = free_port()

    async def go() -> list[str]:
        again = b.Controller(directory)
        serving = asyncio.create_task(again.serve(port))
        await asyncio.sleep(0.1)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await b.send(writer, {"type": "hello", "host": "g", "slots": 0, "running": [], "ended": ["s3-1", "s3-2"]})
        for attempt in ("s3-1", "s3-2"):  # one already ended, one h's
            await b.send(writer, {"type": "ended", "id": attempt, "summary": {"success": None}, "files": {}})
        acked = [json.loads(await reader.readline())["id"] for _ in range(2)]
        writer.close()
        serving.cancel()
        return acked

    assert asyncio.run(go()) == ["s3-1", "s3-2"]
    assert (directory / b.LEDGER).read_text().count('"type": "ended"') == 1
    assert b.State.read(directory).assigned == {"s3-2": "h"}


def test_a_worker_refuses_a_connection_to_itself_and_tries_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    port = free_port()
    itself = socket.socket()
    itself.bind(("127.0.0.1", port))
    itself.connect(("127.0.0.1", port))  # Linux's TCP self-connect: nothing listens on the port
    opened = asyncio.open_connection

    async def open_connection(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await opened(sock=itself)

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    with pytest.raises(ConnectionError, match="itself"):
        asyncio.run(worker(tmp_path, "w", 1).session(port))


def test_a_hosts_window_reads_its_hours_left_and_is_none_outside_them():
    thursday_night = datetime.datetime(2026, 10, 22, 20, 0, tzinfo=SAST)  # a Thursday
    assert b.window_left(b.HOSTS["stone"], thursday_night) == 11 * 3600  # weekday nights to 07:00
    assert (
        b.window_left(b.HOSTS["stone"], datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)) is None
    )  # the owner's day
    assert b.window_left(b.HOSTS["stone"], datetime.datetime(2026, 10, 24, 10, 0, tzinfo=SAST)) == 14 * 3600  # Saturday
    assert b.window_left(b.HOSTS["jebel"], thursday_night) is not None  # on grid power: around the clock
    assert b.window_left(b.HOSTS["klip"], thursday_night) == float("inf")


def test_a_worker_on_another_commit_is_refused_and_given_nothing(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("p", "measure", b.attempts("s3:1-2", ["continuous"]), commit="abc123"))
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 1).serve(port, retry=0.1), timeout=10)  # returns: refused
        serving.cancel()

    asyncio.run(go())
    state = b.State.read(directory)
    assert not state.assigned and not state.ended
    assert '"type": "refused"' in (directory / b.LEDGER).read_text()


def test_a_worker_on_another_crate_is_refused_and_given_nothing(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("p", "measure", b.attempts("s3:1-2", ["continuous"]), crate="0" * 64))
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 1).serve(port, retry=0.1), timeout=10)  # returns: refused
        serving.cancel()

    asyncio.run(go())
    state = b.State.read(directory)
    assert not state.assigned and not state.ended
    refused = [json.loads(r) for r in (directory / b.LEDGER).read_text().splitlines() if '"refused"' in r]
    assert refused and refused[0]["crate"] != "0" * 64


def test_a_windows_end_stops_a_hosts_attempts_and_queues_them_again(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("w", "measure", b.attempts("s3:1-2", ["continuous"])))
    clock = [datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)]
    host = b.Host("w", 2, str(tmp_path), sys.executable, str(tmp_path / "w"), windows=(b.Window(b.ALL_DAYS, 12, 1),))
    port = free_port()

    async def go() -> b.Controller:
        controller = b.Controller(directory, {"w": host}, now=lambda: clock[0], check=0.1)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        running = asyncio.create_task(worker(tmp_path, "w", 2, env={"STUB_SLEEP": "5"}).serve(port, retry=0.1))
        await asyncio.sleep(1.0)  # both attempts assigned and running
        clock[0] = datetime.datetime(2026, 10, 22, 13, 30, tzinfo=SAST)  # the window has closed
        await asyncio.sleep(1.5)
        running.cancel()
        serving.cancel()
        return controller

    controller = asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert ledger.count('"type": "assigned"') == 2 and ledger.count('"why": "window"') == 2
    assert not controller.state.assigned and controller.state.next_attempt() == b.Attempt("s3-1", "continuous", "s3", 1)
    assert not controller.fits("w", b.Attempt("s3-1", "continuous", "s3", 1))  # nothing given outside its hours


def test_a_remote_attempt_is_pulled_and_checked_against_its_manifest(tmp_path: Path):
    source = tmp_path / "source"
    (source / "seed1").mkdir(parents=True)
    (source / "seed1" / "log.jsonl").write_text('{"step": 1}\n')
    (source / "pid").write_text("{}")
    files = b.manifest(source)
    assert list(files) == ["seed1/log.jsonl"]  # the pid file left out
    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    assert b.verify(copy, files)
    (copy / "seed1" / "log.jsonl").write_text("changed\n")
    assert not b.verify(copy, files)


def test_the_fleets_commands_follow_each_hosts_rules():
    stone = b.worker_line(b.HOSTS["stone"], 47111)
    assert stone.startswith("source ~/labour/env.sh; ") and f"PYTHONPATH=/home/david/labour/{b.CRATE}:" in stone
    assert "--slots 11" in stone and "setsid nohup" in stone and stone.endswith("< /dev/null &")
    klip = b.worker_line(b.HOSTS["klip"], 47111)
    assert "--wrapper=scripts/runner/thermal_guard.sh" in klip and "--wrapper=--" in klip
    assert f"PYTHONPATH=/home/david/code/{b.CRATE}:/home/david/code/wt-studies:/home/david/code/wt-studies/ml " in klip
    steps = b.prepare(b.HOSTS["stone"], "abc123", Path("/repo"), Path("/tmp"))
    assert steps[1][3:5] == ["bundle", "create"] and steps[3][3:5] == ["bundle", "create"]
    assert "rev-parse abc123:ml" in steps[2][2] and "--no-recurse-submodules" in steps[6][2]
    assert "cd ~/labour/code/indrajala-studies" in steps[6][2] and "submodule update" in steps[6][2]
    (klip_step,) = b.prepare(b.HOSTS["klip"], "abc123", Path("/repo"), Path("/tmp"))
    assert klip_step[2].startswith("cd ~/code/indrajala-ml && git fetch -q origin")
    assert (
        "cd ~/code/indrajala-studies" in klip_step[2] and "config submodule.ml.url ~/code/indrajala-ml" in klip_step[2]
    )


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "protocol.file.allow=always",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def test_the_controllers_worktree_is_the_studies_commit_with_ml_at_its_pin(tmp_path: Path):
    # a toy ml clone and a studies clone pinning its first commit, beside each other as on the controller
    ml, studies = tmp_path / "indrajala-ml", tmp_path / "indrajala-studies"
    ml.mkdir()
    _git(ml, "init", "-q")
    (ml / "lib.py").write_text("VERSION = 1\n")
    _git(ml, "add", "lib.py")
    _git(ml, "commit", "-q", "-m", "one")
    pin = _git(ml, "rev-parse", "HEAD")
    (ml / "lib.py").write_text("VERSION = 2\n")
    _git(ml, "commit", "-q", "-am", "two")  # past the pin: the worktree must not follow it
    studies.mkdir()
    _git(studies, "init", "-q")
    _git(studies, "submodule", "add", "-q", str(ml), "ml")
    _git(studies / "ml", "checkout", "-q", pin)
    _git(studies, "add", "ml")
    _git(studies, "commit", "-q", "-m", "pin ml")
    commit = _git(studies, "rev-parse", "HEAD")
    host = b.Host("here", 1, str(tmp_path / "wt-batch"), sys.executable, str(tmp_path / "attempts"), remote=False)
    for command in b.prepare(host, commit, studies, tmp_path):
        subprocess.run(command, check=True, capture_output=True)
    assert _git(tmp_path / "wt-batch", "rev-parse", "HEAD") == commit
    assert _git(tmp_path / "wt-batch" / "ml", "rev-parse", "HEAD") == pin
    assert (tmp_path / "wt-batch" / "ml" / "lib.py").read_text() == "VERSION = 1\n"


def test_a_batch_with_a_remote_host_ends_once_its_attempts_are_pulled(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", b.attempts("s3:1-2", ["continuous"])))
    host = b.Host("w", 2, str(tmp_path), sys.executable, str(tmp_path / "w"))  # remote: its attempts pulled
    port = free_port()

    def pull(h: b.Host, attempt: str, destination: Path) -> bool:  # a local copy stands in for rsync over ssh
        shutil.copytree(Path(h.root) / attempt, destination, dirs_exist_ok=True)
        return True

    async def go() -> None:
        controller = b.Controller(directory, {"w": host}, pull=pull, check=0.1)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 2).serve(port, retry=0.1), timeout=30)
        await asyncio.wait_for(serving, timeout=10)  # done: every attempt ended and pulled

    asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert ledger.count('"ok": true') == 2 and ledger.rstrip().endswith("}") and '"type": "done"' in ledger
    assert (directory / "attempts" / "s3-2" / "seed2" / "log.jsonl").exists()


def test_the_controllers_own_hosts_attempts_are_copied_into_the_batch(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", b.attempts("s3:1-2", ["continuous"])))
    host = b.Host("w", 2, str(tmp_path), sys.executable, str(tmp_path / "w"), remote=False)  # rsync's local copy
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory, {"w": host}, check=0.1)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 2).serve(port, retry=0.1), timeout=30)
        await asyncio.wait_for(serving, timeout=10)

    asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert ledger.count('"type": "pulled"') == 2 and ledger.count('"ok": true') == 2 and '"type": "done"' in ledger
    assert (directory / "attempts" / "s3-1" / "seed1" / "log.jsonl").exists()


def test_a_division_backdrop_runs_by_p_seed_and_arm_its_log_at_the_runs_own_path(tmp_path: Path):
    w = worker(tmp_path, "w", 1)
    attempt = b.Attempt("d5-suffix-3", "paper", "d5-suffix", 3)
    command = w.command(attempt)
    run = command[command.index("run") + 1 :]
    assert run[:6] == ["--p", "5", "--seed", "3", "--arm", "suffix"]
    assert run[6:10] == ["--vocabulary", "sum", "suffix", "token"]
    assert "--state-bound" not in command and "--runs" not in command  # the paper's cell stops on its certificate
    assert run[run.index("--factor-width") + 1] == "32" and run[run.index("--budget-epochs") + 1] == "600"
    running = b.Running(attempt, tmp_path / "w" / attempt.id, 0)
    assert w.log_of(running) == running.directory / "log.jsonl"
    machine = b.Running(b.Attempt("s3-1", "continuous", "s3", 1), tmp_path / "w" / "s3-1", 0)
    assert w.log_of(machine) == machine.directory / "seed1" / "log.jsonl"
    bounded = w.command(machine.attempt)
    assert bounded[bounded.index("--state-bound") + 1] == "6"
    assert b.BACKDROPS["d13-family"].args(0)[:2] == ["--p", "13"] and b.BACKDROPS["d13-family"].bound == 169
    assert "--vocabulary" not in b.BACKDROPS["d11-quotient"].args(0)  # T2's: the default vocabulary


def test_a_machine_backdrop_with_an_arm_asks_its_queries():
    planted = b.Backdrop("random-planted-q16-b4-0", 16, "planted")
    assert planted.args(3) == ["--runs", "random-planted-q16-b4-0:3", "--arm", "planted"]
    assert b.BACKDROPS["rq16"].args(3) == ["--runs", "random-q16-0:3"]  # the default arm: none


def test_t1s_cell_is_the_approximation_claims():
    command = worker(Path("/tmp"), "w", 1).command(b.Attempt("d5-sum-0", "approximation", "d5-sum", 0))
    assert command[command.index("--budget-epochs") + 1] == "400"
    assert not {"--fork", "--codes-ratio", "--window-width", "--state-bound"} & set(command)


def test_a_paper_cells_run_succeeds_at_its_first_certificate_and_a_continuous_one_never(tmp_path: Path):
    paper = tmp_path / "paper.jsonl"
    rows: list[dict[str, Any]] = [
        {"step": 1, "epoch": 0, "accuracy": {}},  # an epoch's screen: no end
        {"step": 1, "end": "passed", "epochs": 30, "certified": False, "minimised": 40, "seconds": 9.0},
        {"step": 2, "end": "passed", "epochs": 25, "certified": True, "minimised": 25, "seconds": 20.0},
    ]
    paper.write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = b.summary(paper)
    assert read["success"] == 55 and read["end"] == "passed" and read["step"] == 2
    continuous = tmp_path / "continuous.jsonl"
    rows = [{"step": 1, "end": "spent", "epochs": 400, "certified": False, "long": {"24": 0.2}, "seconds": 5.0}]
    continuous.write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = b.summary(continuous)
    assert read["success"] is None and read["end"] == "spent" and read["epochs"] == 399


def test_t3s_cells_change_only_the_window_or_the_state():
    w = worker(Path("/tmp"), "w", 1)
    base = w.command(b.Attempt("d5-family-wide-0", "approximation", "d5-family-wide", 0))
    for cell, baseline in (("approximation-linear", "linear"), ("approximation-continuous", "continuous")):
        command = w.command(b.Attempt("d5-family-wide-0", cell, "d5-family-wide", 0))
        assert [a for a in command if a not in ("--baseline", baseline)] == base
    assert base[base.index("--vocabulary") + 1 : base.index("--vocabulary") + 4] == ["sum", "difference", "product"]


def test_t1_and_t3s_cells_meet_their_registered_bounds():
    # paper-workplan.md D13, D14: two factors of 16 codes (256 joint), the cap the quotient's exclusion bound,
    # the suffix arm's machine (division's state and the last column's digit sum) within the codes
    from indrajala_studies import modular_family as mf

    cell = b.CELLS["approximation"].args
    assert cell[cell.index("--factors") + 1] == "2" and "--codes-ratio" not in cell  # the default 16 x 16
    factors = [(16, 16), (16, 16)]
    joint = 16 * 16
    assert mf.exclusion_bound(25, factors) == joint + 25 - 1 == 280
    assert joint >= 25 * 5  # the suffix arm's 125 states
    assert {b.BACKDROPS[name].arm for name in ("d5-sum", "d5-suffix", "d5-token")} == {"sum", "suffix", "token"}
    assert all(b.BACKDROPS[n].vocabulary == b.T1_VOCABULARY for n in ("d5-sum", "d5-suffix", "d5-token"))


def test_t12s_arms_are_the_cover_and_the_state_on_one_vocabulary_under_the_approximation_cell():
    # paper-workplan.md D31: at p 5 the cover (sum, difference) against the state; every output a
    # function of division's 25 states, so the joint codes (256) and the cap (280) are T1's
    from indrajala_studies import modular_family as mf

    assert {b.BACKDROPS[name].arm for name in ("d5-cover", "d5-state")} == {"family", "state"}
    assert all(b.BACKDROPS[n].vocabulary == b.T12_VOCABULARY for n in ("d5-cover", "d5-state"))
    assert all(b.BACKDROPS[n].bound == 25 for n in ("d5-cover", "d5-state"))
    args = b.BACKDROPS["d5-state"].args(320)
    assert args[:6] == ["--p", "5", "--seed", "320", "--arm", "state"] and args[6:] == [
        "--vocabulary",
        *b.T12_VOCABULARY,
    ]
    assert mf.exclusion_bound(25, [(16, 16), (16, 16)]) == 280


# paper-workplan.md D15, D16: each backdrop's states q, codes a factor and cap (the exclusion bound) under the paper cell
T2_T5 = {
    "d7-family": (49, 21, 489),
    "d11-family": (121, 32, 1144),
    "d11-quotient": (121, 32, 1144),
    "d13-family": (169, 38, 1612),
    "s3": (6, 8, 69),
    "s4": (24, 15, 248),
    "mtf4": (24, 15, 248),
    "rq12": (12, 11, 132),
    "rq16": (16, 12, 159),
}


@pytest.mark.parametrize("name", sorted(T2_T5))
def test_t2_and_t5s_cells_meet_their_registered_bounds(name: str):
    from fractions import Fraction

    from indrajala_studies import modular_family as mf

    q, codes, cap = T2_T5[name]
    cell = b.CELLS["paper"].args
    assert [cell[cell.index(f) + 1] for f in ("--factors", "--codes-ratio", "--factor-width", "--window-width")] == [
        "2",
        "32/11",
        "32",
        "32",
    ]
    factors = mf.scaled_factors(q, Fraction(32, 11), 2, 32)
    assert b.BACKDROPS[name].bound == q and factors == [(32, codes), (32, codes)]
    assert codes * codes >= 8 * q  # joint codes at least 8 times the states
    assert mf.exclusion_bound(q, factors) == cap


def test_t10s_cells_meet_their_registered_bounds_and_draws():
    from fractions import Fraction

    from indrajala_studies import modular_family as mf
    from indrajala_studies import window_tasks as wt

    factors = mf.scaled_factors(16, Fraction(32, 11), 2, 32)
    assert factors == [(32, 12), (32, 12)] and mf.exclusion_bound(16, factors) == 159  # rows 1, 2: as rq16's
    t10 = {name: d for name, d in b.BACKDROPS.items() if name.startswith("t10-")}
    assert len(t10) == 47 and all(d.bound == 16 for d in t10.values())
    assert {d.arm for n, d in t10.items() if "p2" in n} == {"planted"}
    assert {d.arm for n, d in t10.items() if "p2" not in n} == {"family"}
    planted = wt.planted_blocks(16, 4)
    for m in b.T10_P0:
        assert wt.task_named(f"random-q16-{m}").machine.principal_congruences() == []  # row 17: primitive
    for m in (*b.T10_PLANTED, b.T10_PROBE):
        found = wt.task_named(f"random-planted-q16-b4-{m}").machine.principal_congruences()
        assert len(found) == 1 and wt.task_named(f"random-planted-q16-b4-{m}").machine.is_congruence(planted)
    assert b.BACKDROPS["t10-p1-0"].args(400) == ["--runs", "random-planted-q16-b4-0:400"]
    assert b.BACKDROPS["t10-p2-0"].args(400) == ["--runs", "random-planted-q16-b4-0:400", "--arm", "planted"]


def test_t10rs_cells_meet_their_registered_bounds_and_draws():
    from indrajala_studies import window_tasks as wt

    t10r = {name: d for name, d in b.BACKDROPS.items() if name.startswith("t10r-")}
    assert len(t10r) == 50 and len(b.T10R_P0) == len(b.T10R_PLANTED) == 25
    assert all(d.bound == 16 for d in t10r.values())  # rows 1, 2: T10's cell, as rq16's
    assert {d.arm for n, d in t10r.items() if "p2" in n} == {"planted"}
    assert {d.arm for n, d in t10r.items() if "p0" in n} == {"family"}
    assert not set(b.T10R_P0) & set(b.T10_P0) and not set(b.T10R_PLANTED) & {*b.T10_PLANTED, b.T10_PROBE}
    planted = wt.planted_blocks(16, 4)
    for m in b.T10R_P0:
        assert wt.task_named(f"random-q16-{m}").machine.principal_congruences() == []  # row 17: primitive
    for m in b.T10R_PLANTED:
        machine = wt.task_named(f"random-planted-q16-b4-{m}").machine
        assert len(machine.principal_congruences()) == 1 and machine.is_congruence(planted)
    for m in (25, 26, 42, 45, 47):  # skipped: other congruences
        assert len(wt.task_named(f"random-planted-q16-b4-{m}").machine.principal_congruences()) > 1
    assert b.BACKDROPS["t10r-p2-23"].args(400) == ["--runs", "random-planted-q16-b4-23:400", "--arm", "planted"]


def test_t11s_cells_meet_their_registered_bounds():
    from fractions import Fraction

    from indrajala_studies import modular_family as mf
    from indrajala_studies import window_tasks as wt

    for name, q, codes, cap in (("t11-s3", 6, 8, 69), ("t11-s4", 24, 15, 248), ("t11-mtf4", 24, 15, 248)):
        backdrop = b.BACKDROPS[name]
        assert (backdrop.bound, backdrop.arm) == (q, "cover")
        factors = mf.scaled_factors(q, Fraction(32, 11), 2, 32)
        assert factors == [(32, codes), (32, codes)] and mf.exclusion_bound(q, factors) == cap  # rows 1, 2: T5's
        machine = wt.task_named(backdrop.task).machine
        pair = [wt.query_partitions(backdrop.task)[query] for query in mf.COVER]
        assert all(machine.is_congruence(labels) for labels in pair)  # row 17
        assert len(set(zip(*pair, strict=True))) == q  # their meet the identity: together the state
    assert b.BACKDROPS["t11-s4"].args(700) == ["--runs", "word-problem-s4:700", "--arm", "cover"]


def test_an_opened_host_takes_work_outside_its_hours():
    thursday_noon = datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)  # stone's owner's day
    assert b.window_left(b.HOSTS["stone"], thursday_noon) is None
    opened = b.opened({"stone": b.HOSTS["stone"], "klip": b.HOSTS["klip"]}, ["stone"])
    assert b.window_left(opened["stone"], thursday_noon) is not None and opened["klip"] == b.HOSTS["klip"]
    assert b.HOSTS["stone"].windows  # the table itself unchanged


def test_a_tagged_batch_runs_in_its_own_worktree_and_root():
    klip = b.tagged(b.HOSTS, "t10r")["klip"]
    assert klip.code == "/home/david/code/wt-studies-t10r" and klip.root == "/home/david/code/batch-attempts-t10r"
    assert klip.wrapper[0] == b.HOSTS["klip"].wrapper[0]  # the guard itself shared
    assert klip.wrapper[1] == "/home/david/code/batch-attempts-t10r/temp"  # its temperature file its root's
    assert b.tagged(b.HOSTS, "") is b.HOSTS and b.HOSTS["klip"].root == "/home/david/code/batch-attempts"


def test_a_log_with_a_re_run_appended_is_summarised_from_its_last_run_only(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    run = [{"step": s, "end": "epoch", "epochs": 10, "certified": False, "seconds": float(s)} for s in (1, 2, 3)]
    again = [*run, {"step": 4, "end": "spent", "epochs": 11, "certified": False, "seconds": 4.0}]
    log.write_text("".join(json.dumps(r) + "\n" for r in [{"step": 1, "epoch": 0}, *run, {"step": 1}, *again]))
    read = b.summary(log)
    assert read["epochs"] == 40 and read["step"] == 4 and read["end"] == "spent" and read["seconds"] == 4.0


def test_an_attempt_run_again_gets_a_fresh_directory_its_old_one_set_aside(tmp_path: Path):
    first = worker(tmp_path, "w", 1)
    attempt = b.Attempt("s3-1", "continuous", "s3", 1)
    root = tmp_path / "w"
    for run in range(3):  # each run lost and the attempt sent to this host again
        if run:
            (root / "s3-1" / "mark").write_text(str(run))
        running = first.launch(attempt)
        while b.alive(running.pid):
            time.sleep(0.05)
    assert sorted(p.name for p in (root / "lost").iterdir()) == ["s3-1-1", "s3-1-2"]
    assert (root / "lost" / "s3-1-2" / "mark").read_text() == "2" and not (root / "s3-1" / "mark").exists()
    assert (root / "s3-1" / "seed1" / "log.jsonl").exists()
    assert not worker(tmp_path, "w", 1).running  # the set-aside directories are never adopted


def test_an_attempts_pss_follows_a_wrappers_setsid_child_and_its_workers(tmp_path: Path):
    proc = tmp_path / "proc"
    # pid: (ppid, session, PSS kB): 10 the wrapper (the attempt's session), 11 the run it setsid'd, 12 a
    # pool worker of the run's, 20 another session altogether
    for pid, (ppid, session, kb) in {10: (1, 10, 4), 11: (10, 11, 200), 12: (11, 11, 300), 20: (1, 20, 999)}.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(f"{pid} (x y) S {ppid} {session} {session} 0 -1\n")
        (proc / str(pid) / "smaps_rollup").write_text(f"Rss: 1 kB\nPss: {kb} kB\n")
    assert b.attempt_pss(10, proc) == 504 and b.attempt_pss(11, proc) == 500 and b.attempt_pss(20, proc) == 999


def test_a_sessions_pss_sums_its_processes_and_an_attempts_peak_reaches_its_summary(tmp_path: Path):
    mine = b.attempt_pss(os.getsid(0))
    assert mine is not None and mine > 0 and b.attempt_pss(1, tmp_path / "none") is None
    state = run_batch(
        tmp_path,
        b.Batch("m", "measure", b.attempts("s3:1", ["continuous"])),
        worker(tmp_path, "w", 1, {"STUB_SLEEP": "0.3"}),
    )
    assert state.ended["s3-1"]["peak_pss_mb"] > 1


def test_an_attempt_reruns_by_the_workers_own_command(tmp_path: Path):
    attempt = b.Attempt("s3-7", "approximation", "s3", 7, cutoff=120)
    made = b.attempt_command(attempt, "python", "probe.py", tmp_path / "rerun")
    assert made[:3] == ["python", "probe.py", "run"] and made[-2:] == ["--out", str(tmp_path / "rerun")]
    assert made[made.index("--cutoff-epochs") + 1] == "120"
    w = worker(tmp_path, "w", 1)
    assert w.command(attempt) == b.attempt_command(attempt, w.python, w.script, w.root / attempt.id)
