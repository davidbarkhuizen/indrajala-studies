# Working notes: gotchas for the studies

Moved from indrajala-ml's `docs/working-notes.md` on 2026-10-31 (the research-separation workplan); its
Git, worktree, test, machine and shell notes stay there and apply here too. Paths in the notes are
this repository's: `indrajala_studies/` for ml's `indrajala_ml/studies/` research modules and
`indrajala_studies/batch.py`, the scripts at the same paths under `scripts/`.

## Studies: running them

- **Long jobs never swap** (measurement.md section 2): size workers from the largest of a probe's
  peak, a recorded `peak_mb` and a worker's RSS a few minutes in; check `free -m` and `vmstat`
  (`si`/`so`) after the first minutes. **One memory-heavy job at a time on `pyramidon`** (5.3 GB):
  a `k` 1 long test on an E8c network grows past 1.6 GB at 1,024 columns, and with a `k` 4
  `certify` (about 620 MB) beside it the machine swapped. A `certify` pool worker at `k` 4 takes
  about 440-620 MB though one in-process probe measured 45 MB; at most 3 such workers on
  `pyramidon`. **At most 2 long-test shares on `klip`** (5.3 GB): an indrajala-studies `paper_reading.py long` child on
  a 16 x 16-code network grows to 1.0-1.3 GB at 1,024 columns, and 4 shares swapped it
  (2026-10-28). Stopping a share means its `modular_family_probe.py long-test` child too: killing the
  parent leaves the child running, orphaned.
- **`/tmp` on `pyramidon` is RAM-backed (tmpfs) with a per-user quota:** scratch copies of study
  data cost memory, and a test writing many run directories filled the quota (2026-10-18). Remove
  each one's files as it goes (the resume test does).
- **Training state set on the network isn't saved with it** (temperatures, couplings, continuous
  factors, commitments): a resumed run loads the network bare, so whatever its first screen reads
  before the next epoch sets them must be kept in the checkpoint's `Place` and re-applied (the
  couplings were not until 2026-10-21, found by the adaptive cool's resume test). A resume test whose
  factors harden at once (coupling 1) can't see such a gap: give a new profile's test soft factors.
- **Epochs bind a run; a wall clock never does** (the programme audit's F12): the core matrix (D46,
  2026-10-21) chose an epoch budget because "a minutes budget would give a run's outcome its
  machine"; the probes registered that evening onward (D54-D64) still used "1,600 epochs or 690
  minutes", and on a loaded `stone` the clock cut ten runs short (D67 re-ran them). `--budget-epochs`
  is required now; set `--budget-minutes` only as a runaway guard far above any normal run, and
  register its stop as censored.
- **A run its wall-clock guard stopped resumes:** `modular_family_probe.py run` with a "deadline"
  stop keeps the last epoch's checkpoint (not "ended"), so the same command with `--resume` and a
  larger `--budget-minutes` goes on to its epoch budget as the uninterrupted run (the deadline's
  rows and the cut step's files dropped); the guard isn't part of the run's identity. Runs cut
  before this change (D67's nine) ended their checkpoints and can't: re-run them from their seeds.
- **Background Python writing to a log is block-buffered:** a killed job leaves nothing. Use
  `python -u` or `PYTHONUNBUFFERED=1` for any log you may need mid-run.
- **The harness log prints nothing within a step;** progress shows as `models/<key>/step-<j>.json`
  files, and `status` (safe over ssh while a study runs) counts them. **To see whether a run is
  going anywhere, `diagnose CONFIG --run KEY --follow`** beside it (one core, about a second a
  step): each step's accuracy per answer class and each factor's share of the target's
  coordinates, as each step is saved, until the study ends. Write the stop rule into the study's
  log before the first diagnosis, not after (C9' seed 5).
- **A stopped run doesn't resume:** its last step is saved with stop "stopped" and its result
  written, so `run` skips it and `train_run` would end at that step again. To resume one run (within
  its study's budget), move its `results/<key>.json` and its stopped step's `step-<j>.json` and
  `.network.json` aside (not delete), and the stop file; `run` then trains only it, from step
  `j - 1`'s network.
- **A registered stop check runs at its time, never behind a question to the owner:** C9''s 2 h
  rule waited on an unanswered question and both machines ran 2 h more. Apply the rule, then ask.
- **Don't `git pull` on a machine while a study runs there:** its forkserver may import changed
  modules.
- **Stopping a study** (only with the owner's say): kill its Python process's whole process group,
  `kill -TERM -- -<pgid>` (`ps -eo pid,pgid,args`); the python, its resource tracker and forkserver
  workers share it. Killing a bash wrapper's group leaves the workers training after the lock is
  freed. `status` then shows stale "running" entries until the next `run`.
- **`certify` and `long-test` don't skip work already done:** to resume, pass only the missing
  networks.
- **An existence study's certified count is "at least":** parallel runs can certify in the same
  epochs before the stop reaches them.
- **Training past a certified step can destroy the network** (E8b): existence tests use
  `certify_steps` and `stop_at_certified`.
- **A state can collapse to one code at long training lengths:** a fresh `m` 16 network starts on
  1-2 codes; lengths 1-3 or 1-6 split it, 1-10 or 1-20 don't. A 2-epoch probe (about 20 s an epoch
  at 20 columns) shows it before a long launch.
- **A hand-built training set needs the harness's final zero window** (`window_recurrent_study.
  example`): its answer needs the state the data left, so even a one-column step trains the state.
  Without it the modular-family probe's step 1 passed on the column alone and its state stayed on
  one code in every arm.
- **Every probe stops at its first certified step** (certify each step's end: under a second at
  `k` 1): all four of the modular-family runs that solved or nearly solved division collapsed when
  trained on (E8b's warning, seen again). Save the network before it is overwritten.
- **`modular_family_probe.py run` with one `--task` writes `DIR/seed<s>`**, not
  `DIR/<task>/seed<s>` as several tasks do: read and `readout` such a run as `--arms .`.
- **To probe beside a running study on `jebel`, use a worktree there** (`git worktree add --detach
  ~/code/wt-X origin/<branch>`, `.venv` and `data/window-recurrent` linked, `PYTHONPATH=$PWD`); don't
  update its files while a harness study (forkserver workers) runs from it. A single-process probe
  has its modules loaded and doesn't care.
- **When trimming a stage, check which later experiments compare against the dropped cells**
  (D34 removed E2's and E6's `k` 1 comparators).
- **A probe run at a branch's commit: tag the commit (`probe/<name>-<sha>`) and push the tag
  before the PR is squash-merged.** The squash leaves the branch's commits unreachable, and once the
  branch is deleted git's gc removes them: the registration's "at <commit>" then names nothing (13
  such commits were found and tagged on 2026-10-29). **And bring each runner's data to `pyramidon`
  when its probe ends** (`data/window-recurrent/`, a manifest): D32, D33, D38 and D56 existed only on
  `stone` and `jebel` until then, though a workplan named their place here.
- **Copying results between machines:** `sha256sum` on the source, `rsync`, `sha256sum -c` on
  the destination; record the manifest's own sha256 in the findings. **Don't rename or re-root on
  the copy:** a host's manifest lists its own paths (`klip`'s `data-d64/probe-d64/...`), so a file
  renamed on arrival (`launch.log` to `launch-klip.log`) or a different root fails `sha256sum -c`
  on the archive (the audit's F31). Keep each host's files under its own directory as listed, or
  write an archive-side manifest after the copy (`HOST-option1-dNN.archive.sha256`).
- **Manifests' base directories differ:** `X.sha256` beside a directory `X` lists paths inside it
  (`e8-q4.sha256`, run from its parent); `certificates.sha256` inside `e8-q4/` lists bare names
  (run from `certificates/`); `control/results.sha256` and `matrix/results.sha256` list
  `results/...` (run from `control/`); `e8c-q*.sha256` list `e8c-q4/...` (run from
  `data/window-recurrent`).

- **Two runs at once on one machine each need their own lock:** `run` holds `$AB_RUNS_ROOT/run.lock`,
  so a second run started with the same `AB_RUNS_ROOT` is refused ("not started: ... run.lock is held by
  study ...") before it writes data. Give each concurrent script its own `AB_RUNS_ROOT`.
- **`certify --out` is the certificate's file path, not a directory:** a second verdict written to the
  same path overwrites the first. Name them `certify/<task>-seed<s>-certified-step-<n>[.strict].certificate.json`
  (the strict verdict with `--snap-ratio 1.05`), with `certify.log` beside them.

### The batch runner (`scripts/runner/batch.py`, `indrajala_studies/batch.py`)

- **The ledger's commit must be the full sha:** workers report theirs in full and the controller
  compares exactly, so a short `--commit` refused every worker (2026-10-23, T1's first start; nothing
  ran). `register` now expands it.
- **Stop a fleet by its process group** (`kill -- -PGID`; it runs under `setsid`): killing the fleet's
  pid alone leaves its `ssh -R` tunnels holding the remote port, and the next fleet's tunnels fail with
  "remote port forwarding failed" (harmless while the old ones forward to the same port, but they
  outlive it).
- **A worker can connect to itself** (Linux's TCP self-connect): while its tunnel is down, a retry to
  127.0.0.1:PORT given PORT as its own ephemeral port connects to itself, and its messages go nowhere.
  T10 (2026-10-25/26): `stone` off the network for about 20 hours, its worker self-connected, its 11
  ends reported into the void (dropped from its memory), and the tunnel, the IPv4 port taken, bound
  `[::1]` alone; the controller held them as running. Recovered by stopping that worker, reopening the
  tunnel and replaying each `ended` from its `ended.json` (the controller's own pull and check). Now
  the worker refuses a connection to itself, the tunnel binds `127.0.0.1` (or fails and retries), and
  the controller acks each `ended`: a worker resends the unacknowledged ones (`ROOT/acked/`) on its
  next connection, listed in its hello so they aren't lost, and leaves a drain only once every end is
  acked. A worker at this commit needs a controller at it (an older one never acks: no drain ends).
- **The runner's worktree is `wt-studies`, not `wt-batch`** (2026-10-31): every host's `~/code/wt-batch`
  is a worktree of its ml clone from before the separation, and `prepare` falls back to a checkout in an
  existing directory, which can't reach a studies commit there. A tagged batch (`wt-studies-<tag>`)
  never met it, so the first test batch on `jebel` passed. The old ml worktrees can go (`git -C
  ~/code/indrajala-ml worktree remove --force ~/code/wt-batch`).
- **The crate is pinned too** (D6): `register` records the crate extension's SHA-256 (`--crate`, default
  the registering machine's) and the controller refuses a worker that imports another. Builds of one
  source differ by machine, so every host imports one build: `b.CRATE` (`crate-647f23c`), the package
  directories copied from `pyramidon`'s venv into `~/code/crate-647f23c` (`~/labour/` on `stone`) and put
  first on PYTHONPATH by the host table. A new crate: a new directory on every host, `CRATE` changed. T1 ran before the check: `pyramidon`'s crate (rust 647f23c, whose one change is
  aarch64-only NEON) differed from `klip`'s and `stone`'s (45d6a37); `same-prefix` on `d5-sum-304` gave
  31 identical rows across them (2026-10-23).
- **A lost attempt re-run on its host appended to its old log** (T1: `d5-sum-300`, `d5-sum-302`,
  `d5-suffix-302`, on `pyramidon`; the ledger summed both runs, 662, 681 and 596 epochs for 400). Now
  the worker sets the old directory aside (`ROOT/lost/<attempt>-<n>`, never adopted) and `summary`
  reads a log's last run only (a step end not after its predecessor's starts one), so re-summarising an
  older ledger's logs corrects it. Ledgers written before the fix still hold the summed summaries.
- **An attempt's memory is the worker's sample, not `/usr/bin/time`:** a run's pool workers are a
  forkserver's children, outside the attempt's rusage (`time -v` saw 51 MB of a division run). The
  worker sums PSS over the attempt's session at each look (every 2 s) and puts the peak in its summary
  (`peak_pss_mb`); a spike shorter than a look can be missed. **`klip`'s peaks before the fix are
  wrong** (3.8-6.5 MB on T10's probe against `pyramidon`'s 184-221 MB): its thermal guard `setsid`s the
  run, so the session held only the wrapper. Now the sum follows every descendant of the session's
  processes too (`attempt_pss`); use another host's peaks for `klip`'s earlier attempts.
- **Several cells in one batch:** `register --cell A --cell B ...` runs the spec under each cell in
  turn (list the longest first: the fleet takes attempts in order), the ids `<cell>.<backdrop>-<seed>`
  so the cells' runs never share a directory; `find` counts `k` per cell and backdrop. A ledger from
  before (one `cell` for the batch) is read as each attempt's, and the attempt message is unchanged,
  so a controller at a newer commit serves workers pinned at an older one.
- **A host given outside its hours:** `controller --open HOST` (the owner's grant, logged in the
  ledger as `opened`); the host table is unchanged.
- **The controller's own host's attempts weren't copied into the batch** (T2/T5, T10's probe): only
  remote hosts were pulled, so `pyramidon`'s attempts stayed in `~/code/batch-attempts/` with no
  `pulled` event, and a reading of `attempts/` missed them (T2/T5's 15, copied by hand). Now every
  host's ended attempt is pulled, the local one by rsync without ssh, and checked against its
  manifest. A controller started before the fix still needs the hand copy (excluding `ab-runs`).

## Studies: reading them

- **Test window networks at full catalogue sizes:** reduced samples passed lengths the full sizes
  fail on a near-working network. `check` uses the full default catalogue; to time other sizes,
  call `wr._evaluate(model, n, wr.test_properties(...))`.
- **`tests_past_break: false` leaves lengths untested** (the report shows `-`), and codes used then
  count only the lengths tested: at break 0 it is the initial state's code alone. To see collapse,
  run `state_codes` on real examples (`wr.example`).
- **The checks are exhaustive at short lengths** (B1 at `n` 4 is all 531,441 triples); a check at
  12-20 columns costs minutes (187 s at 12 in E8b's probe): budget it.
- **`certify` branches near ties by default** ("robustly certified"); `--snap-ratio 1` gives the
  strict verdict ("certified"). `wrong_ties` counts only wrong carries on a tie's own branch: a code
  reached only through a tie can carry wrong on its own windows, so read the strict certificate
  for "wrong as computed".
- **`certified` = no wrong output, no wrong state, no output near tie;** `correct` without
  `certified` means an output near tie only. A carry counterexample shows as a wrong sum only
  through the next window and only at alignment 0, which the harness's hash picks for few cases:
  feed it with `wr.example(task, case, n - k, k, 0)` (alignment 0), not the harness's hashed draw.
- **The certificate is fast at `k` 1** (about 2 s a 40-network pass) and slow at `k` 4 (minutes a
  network).
- **The long-test JSON:** `lengths -> n -> properties -> id -> {cases, passed, failures}`;
  `modular_family_probe.py long-summary DIR` reads a study's (`long-test --out DIR/<run>`): a line
  per run, its failures and the network's sha256, and one for all.
- **`WindowRecurrence.codes` is None for a continuous network:** `state_codes`, `first_step` and
  `window_step` assert "no codes" on one.
- **Results carry `window`** (#717) and may carry `m: null`: use `comparable`'s defaults (no
  `window` = transformer; no `task` = addition).
- **`addition_data.catalogue()` returns new Property objects:** `screen_cases` caches per Property,
  so build the catalogue once per step.
- **A modular-family step stopped by a budget logs one epoch too many:** its end row's `epochs` is
  the loop's index at the stop (`spent`, `deadline`), not trained, so a run of `--budget-epochs
  400` sums to 401. `modular_family.outcome` (the `readout`) corrects it; sum the rows by hand and
  you must too.
- **`approximations` reads exact networks:** it joins every reachable state pair a factor's code
  holds and closes the partition, each reachable (code, state) pair counted once. On an inexact
  network (a failure's peak, a collapse) a few rare wrong codes merge states, the closure falls to
  one block, and it reads "nothing" where the factor holds a place at 0.9-1.0. Read inexact networks
  with `factor-shares` (each factor's share of each named congruence, weighted as training draws);
  the 2026-10-26 re-reading of option 1 withdrew two patterns read the other way.
- **A study's end can't be read from its average rate:** later ladders are slower; quote a range.
- **Check every figure against the data before writing it into a doc.**
- **The addition study's harness:** `train_seconds` include the whole catalogue whenever a screen
  passed, so a short passing run's seconds per epoch look several times the true rate; its worker
  processes run the module's `CANDIDATES` only; `patch_study.parameter_count` is numpy-only; don't
  test its catalogue's generalisation at `n` 2; a screen passing isn't a pass.

- **A re-run that rewrites result files in place breaks every older manifest listing them:** keep
  the old files beside (as `proof-loose/`), repoint the old manifest's entries to them, and verify
  every manifest (`sha256sum -c` from its directory) before calling a data set archived.
- **A paper fetched into indrajala-literature may already sit in indrajala-research's `extra.bib`
  as "not held":** drop the `extra.bib` entry in the same change, or bibtex fails on the repeated
  key and research's CI goes red.

## Code internals

- **The window study's API** (`indrajala_studies/window_recurrent_study.py` as `wr`):
  `wr.example(task, case, n, k, a)`, `wr.alignment(task, case, n, k)`, `wr.build(task, k, s, m,
  seed)`, `wr.window_specs(task)`, `wr.WindowModel(network, k, task)` with `.evaluate(n,
  properties[, limit])` and `.predict(n)`; `wr.check_properties(task, n, cat)`,
  `wr.test_properties(task, cat)`; `settings.catalogue(task)`.
- **The certificate's JSON names:** `wrong_outputs`, `wrong_states`, kinds `outputs`/`state`,
  `state_in`, `case`, `after`, `task`. E1's files keep the older `wrong_digits`, `wrong_carries`,
  `digits`/`carry`, `carry_in`, `addition`.
- **Never `x // BASE**t` (or `d * BASE**t`) for every digit t of a long number:** each power and
  division is a big-integer operation the size of the number, so a conversion is quadratic in the
  length. At the long tests' 1,024 columns it was a quarter of a long test (the window-program
  workplan's stage 1). Use `addition_data.digits` and `from_digits` (DIGIT_CHUNK digits a
  big-integer step, the rest from a table or small ints).

## Background jobs

- **To run a study from a worktree's code into the main checkout's data** (so `main` can be pulled
  while it runs): `cd` to the main checkout, `PYTHONPATH=<worktree>`, and the script and config by
  the worktree's path. `indrajala_ml` is a namespace package, so check which copy loads (a
  one-line script in the worktree's `scripts/` printing a module's `__file__`): run as a script,
  the worktree's loads; `python -c` from the main checkout loads the main checkout's. E2 ran this
  way (`~/code/wt-e2`, launched by `docs/probe-launches/jebel/e2-launch.sh` at 40d66d9).
- **Never develop in a worktree a queued or running study imports from:** a launcher that waits
  imports the worktree as it is when the study starts (and records its commit then). Keep a separate
  worktree for development (`~/code/wt-dev`); `~/code/wt-e2b` on `pyramidon` was nearly launched on a
  development commit.
- **A `mkdir` claim over ssh isn't a perfect lock:** stage 5's shared long tests
  (`docs/probe-launches/stage5-long-tests.sh` at 40d66d9, each worker claiming a network by `ssh pyramidon mkdir ...`) had
  two of `jebel`'s workers start the same network in the same second (2026-10-16), cause not found.
  Harmless for an idempotent test (one result overwrote the other); for anything that isn't, have
  each worker take a fixed share instead.
  Worse, a queued study whose script waits on `pgrep -f <the previous script's name>` also waits
  on a waiter watching that name. On 2026-10-18 that held `jebel` idle for hours after round 2
  ended (the waiter's `while pgrep -f algebras-round2.sh` never ended, so neither did the
  diagnostic's). Wait on a log line instead (`until grep -q "ended" .../launch.log`), and have each
  launch script write one.
- **A second batch beside a running one on the same host needs its own code, root and port**
  (2026-10-25, T11 on `klip` beside T10): `prepare` would move the host's `wt-studies`, which the running
  batch's worker loads code from for every new attempt; and a worker adopts every running attempt
  in its root from their pid files (`Worker.adopt`), so a second worker in the same root takes the
  first batch's. So: a worktree of its own at the batch's commit (`~/code/wt-studies-<name>`), a root
  of its own (`~/code/batch-attempts-<name>`), its controller on another port with the tunnel by hand
  (as `batch.tunnel`; it doesn't reopen itself), and the worker started by hand (as
  `batch.worker_line`). A controller without `--hosts` assigns to any host and **pulls nothing**
  (`Controller.fits` and `ended` skip a host not in it): rsync the ended attempts in by hand.
- **A direct `modular_family_probe.py run` takes the machine's run lock:** a second one started beside
  it refuses ("not started: ... run.lock is held"); run several as one `--runs` pool, or one after
  another. Batch attempts don't contend for it.
- **A long test or study run from a worktree with `PYTHONPATH` still takes the main checkout's
  `.venv`, its compiled crate included:** check `git diff --stat <pinned> <main's> -- rust/` is empty
  before trusting the crate hash the results record.
