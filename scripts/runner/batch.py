"""
The batch runner (its workplan retired: git show 53d7fef:docs/workplans/batch-runner-workplan.md): a controller on one machine and a worker daemon on
each, over newline-delimited JSON on a TCP socket on 127.0.0.1 (indrajala_studies.batch).

    python scripts/runner/batch.py register DIR --name NAME --mode measure|find --cell CELL [--cell CELL ...] --attempts SPEC
                                     [--cutoff BACKDROP=EPOCHS ...] [--k K] [--commit COMMIT]
    python scripts/runner/batch.py prepare --hosts pyramidon,klip,stone --commit COMMIT [--tag TAG]
    python scripts/runner/batch.py controller DIR [--port PORT] [--hosts pyramidon,klip,stone] [--tag TAG]
    python scripts/runner/batch.py fleet --hosts pyramidon,klip,stone [--port PORT] [--slots HOST=N ...] [--tag TAG]
    python scripts/runner/batch.py worker ROOT --slots N --code WORKTREE [--python PYTHON] [--host NAME] [--port PORT]
                                   [--pythonpath PATH] [--wrapper ARG ...]
    python scripts/runner/batch.py status DIR
    python scripts/runner/batch.py report DIR

SPEC is `s3:40-47,mtf4:32-35` (BACKDROPS' short names), run under each cell in turn (the longest
first); a cutoff is a backdrop's Luby t* in epochs trained
(the run's --cutoff-epochs; epochs are the time axis). The controller serves until the batch is done; a worker runs its attempts in
ROOT/<attempt>/ from WORKTREE's scripts and reconnects after a dropped connection. Across machines
(stage 4): `prepare` puts each host's worktree at the batch's commit, `fleet` keeps a reverse ssh tunnel
to each remote host and starts its worker (HOSTS: slots, worktree, python, hours), and the controller
with `--hosts` gives work only inside each host's hours and pulls each remote attempt back, checked.
A TAG sets each host's worktree and attempts' root apart (`<path>-TAG`), so a second batch runs
beside another's workers at another commit.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import shlex
import subprocess
import sys
from pathlib import Path

from indrajala_studies import batch as b

PORT = 47111


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="a new batch's directory and ledger")
    register.add_argument("dir", type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--mode", choices=("measure", "find"), required=True)
    register.add_argument(
        "--cell", choices=sorted(b.CELLS), action="append", required=True, help="each cell, the longest first"
    )
    register.add_argument("--attempts", required=True, help="s3:40-47,mtf4:32-35")
    register.add_argument(
        "--cutoff", action="append", default=[], help="BACKDROP=EPOCHS, each backdrop's t* in epochs trained"
    )
    register.add_argument("--k", type=int, help="find mode: certificates a backdrop needs")
    register.add_argument("--commit", help="the commit every worker must run (D6); a short sha is expanded")
    register.add_argument(
        "--crate", help="the crate extension's SHA-256 every worker must import (D6); default the hosts' (b.CRATE)"
    )
    prepare = commands.add_parser("prepare", help="each host's worktree at a commit")
    prepare.add_argument("--hosts", required=True)
    prepare.add_argument("--commit", required=True)
    fleet = commands.add_parser("fleet", help="each host's tunnel and worker")
    fleet.add_argument("--hosts", required=True)
    fleet.add_argument("--port", type=int, default=PORT)
    fleet.add_argument("--slots", action="append", default=[], help="HOST=N, fewer slots than the host table's")
    controller = commands.add_parser("controller", help="serve a batch until it is done")
    controller.add_argument("dir", type=Path)
    controller.add_argument("--port", type=int, default=PORT)
    controller.add_argument("--hosts", help="the fleet's hosts: their hours and pulls (D7)")
    for command in (prepare, fleet, controller):
        command.add_argument(
            "--tag", default="", help="each host's worktree and attempts' root as <path>-TAG: a batch beside another's"
        )
    controller.add_argument(
        "--open",
        default="",
        help="hosts the owner has given outside their hours, open for this controller's life (logged in the ledger)",
    )
    worker = commands.add_parser("worker", help="run attempts on this machine")
    worker.add_argument("root", type=Path)
    worker.add_argument("--slots", type=int, required=True)
    worker.add_argument("--code", type=Path, required=True, help="the worktree whose scripts run")
    worker.add_argument("--python", default=sys.executable)
    worker.add_argument("--host")
    worker.add_argument("--port", type=int, default=PORT)
    worker.add_argument("--pythonpath", default="", help="before the worktree on PYTHONPATH (stone's crate)")
    worker.add_argument("--wrapper", action="append", default=[], help="an argument before each attempt's command")
    for name in ("status", "report"):
        commands.add_parser(name).add_argument("dir", type=Path)
    rerun = commands.add_parser("command", help="the command that reruns one attempt of a batch, from its ledger")
    rerun.add_argument("dir", type=Path)
    rerun.add_argument("attempt", help="the attempt's id, e.g. d5-sum-319")
    rerun.add_argument("--out", type=Path, required=True, help="the rerun's directory (never the original's)")
    args = parser.parse_args(argv)

    if args.command == "register":
        cutoffs = {k: int(v) for k, v in (c.split("=") for c in args.cutoff)}
        made = b.attempts(args.attempts, args.cell, cutoffs)
        commit = args.commit
        if commit is not None:  # the workers report a full sha and the controller compares exactly
            repo = Path(__file__).resolve().parents[2]
            commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        here = next(h for h in b.HOSTS.values() if not h.remote)  # the controller's machine: its batch crate
        repo = Path(__file__).resolve().parents[2]
        crate = args.crate or b.identity(repo, here.python, {"PYTHONPATH": here.pythonpath})[1]
        b.register(args.dir, b.Batch(args.name, args.mode, made, args.k, commit, crate))
        print(f"registered {args.dir}: {len(b.State.read(args.dir).batch.attempts)} attempts")
    elif args.command == "prepare":
        repo = Path(__file__).resolve().parents[2]
        hosts = b.tagged(b.HOSTS, args.tag)
        for name in args.hosts.split(","):
            for command in b.prepare(hosts[name], args.commit, repo, Path("/tmp")):
                subprocess.run(command, check=True)
            print(f"{name}: at {args.commit}")
    elif args.command == "fleet":
        hosts = b.tagged(b.HOSTS, args.tag)
        slots = {n: int(s) for n, s in (x.split("=") for x in args.slots)}
        if any(slots[n] > hosts[n].slots for n in slots):
            parser.error("--slots only lowers a host's slots")
        chosen = [dataclasses.replace(hosts[n], slots=slots.get(n, hosts[n].slots)) for n in args.hosts.split(",")]
        asyncio.run(b.fleet(chosen, args.port))
    elif args.command == "controller":
        hosts = (
            {n: h for n, h in b.tagged(b.HOSTS, args.tag).items() if n in args.hosts.split(",")} if args.hosts else None
        )
        opened = [n for n in args.open.split(",") if n]
        if opened and (hosts is None or not set(opened) <= set(hosts)):
            parser.error("--open names hosts among --hosts")
        controller = b.Controller(args.dir, b.opened(hosts, opened) if hosts else None)
        if opened:
            controller.log({"type": "opened", "hosts": opened})
        asyncio.run(controller.serve(args.port))
        print("\n".join(b.report(args.dir)))
    elif args.command == "worker":
        worker = b.Worker(
            args.root,
            args.slots,
            args.python,
            args.code,
            host=args.host,
            pythonpath=args.pythonpath,
            wrapper=args.wrapper,
        )
        asyncio.run(worker.serve(args.port))
    elif args.command == "command":
        state = b.State.read(args.dir)
        attempt = next((a for a in state.batch.attempts if a.id == args.attempt), None)
        if attempt is None:
            raise SystemExit(f"{args.attempt}: no such attempt in {args.dir}'s ledger")
        if args.out.resolve().is_relative_to(args.dir.resolve()):
            raise SystemExit("--out: inside the batch's directory, beside the originals; a rerun writes elsewhere")
        print(
            shlex.join(b.attempt_command(attempt, sys.executable, "scripts/studies/modular_family_probe.py", args.out))
        )
    elif args.command == "status":
        state = b.State.read(args.dir)
        print(f"{len(state.ended)} ended, {len(state.assigned)} running ({dict(state.assigned)}), done {state.done}")
    else:
        from indrajala_studies import modular_family as mf  # the reading's cost, not the daemons'

        state = b.State.read(args.dir)
        print("\n".join(b.report(args.dir)))
        several = len(state.batch.cells) > 1
        for (cell, backdrop), mine in b.groups(state.batch).items():
            ended = [state.ended[a.id] for a in mine if a.id in state.ended]
            if ended:
                name = f"{cell} {backdrop}" if several else backdrop
                print(f"  {name}: {mf.restart_summary(mf.restart_cost(b.runs_of(ended)), 'epochs')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
