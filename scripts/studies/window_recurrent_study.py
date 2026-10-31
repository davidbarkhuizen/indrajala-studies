"""
The window-recurrent adder (the window-recurrent adder workplan): whether a network that reads a
window of k columns and hands a snapped state from window to window learns, through a curriculum of
window counts, to add base-3 numbers of any length, under indrajala_studies/window_recurrent_study.py's
harness and indrajala_studies/addition_data.py's catalogue.

    python scripts/studies/window_recurrent_study.py run CONFIG.json [--workers W] [--arm ARM]
    python scripts/studies/window_recurrent_study.py status CONFIG.json
    python scripts/studies/window_recurrent_study.py report CONFIG.json [--md FILE]
    python scripts/studies/window_recurrent_study.py check NETWORK.json --n N [N ...] [--task T] [--properties ID ...]
        [--sampled S] [--per-stratum P] [--doubling] [--json FILE]
    python scripts/studies/window_recurrent_study.py certify NETWORK.json [...] [--task T] [--out DIR] [--workers W]
        [--snap-ratio R]
    python scripts/studies/window_recurrent_study.py explore CONFIG.json [--n N ...] [--json FILE] [--workers W]
    python scripts/studies/window_recurrent_study.py compare ORIGINAL_DIR RERUN_DIR
    python scripts/studies/window_recurrent_study.py long-test NETWORK.json [...] [--task T] [--n N ...] [--properties ID ...]
        --out DIR [--workers W]
    python scripts/studies/window_recurrent_study.py networks CONFIG.json
    python scripts/studies/window_recurrent_study.py verdicts CONFIG.json --certificates DIR --strict DIR [--long-tests DIR]
    python scripts/studies/window_recurrent_study.py probe CONFIG.json [--steps J ...] [--epochs E] [--json FILE]
    python scripts/studies/window_recurrent_study.py diagnose CONFIG.json [--run KEY ...] [--n N ...] [--cases C]
        [--follow] [--interval S]

A config names the study's directory (results/, one JSON file per run, and models/, each run's steps
and their networks in format 2), its workers, its settings and its matrix (load_config's
docstring), its task among them (addition by default, or modular-division-5 or -7:
indrajala_studies/window_tasks.py). `run` resumes from what is on disk, a run from its last saved step, and prints a line
per finished run, then the report; it holds the machine lock, as every study. `--arm` runs one arm's
runs only (a study's arms split between machines; keys and directory unchanged). `check` loads a saved
step's network and runs its task's catalogue (without the held-out property, B3 or U2) at each
length: its properties, the samples (addition's B1-B3, A3, M1 and M5; division's U1 and U2), the
cases per stratum (M3, M4; S1, S2) and M3 or S2 at 1, 2, 4, ..., n only (the long tests of
indrajala-research/docs/research-notes/window-recurrent-report.md, D9) as given. `long-test` is those long tests over many
networks and W processes (at 128, 256 and 1,024 by default, the stratified distances doubling), each
network's results as DIR/<run>-step-<j>.json (`check --json`'s), a network already there skipped,
a line each of its cases and failures. `certify` writes each saved network's
certificate against its task's machine (indrajala_studies/window_recurrent_certificate.py) as
DIR/<run>-<step>.json and prints a line for each, over W processes; --snap-ratio 1 certifies the
network's own snaps only, branching no near tie (strict, D3 as first written). `compare` is C0's check
(the claims workplan): every run in both study directories, its result without timings and
provenance and each step's saved network, exactly; a line per run, exit 1 if any differ. `explore`
is E5: for every finished snapped run of a study, the codes its last network reaches at its check
lengths and at each N (8, 16 and 64 by default), the codes new past the check lengths, and how
many failing and working runs whose check passed reach a new code (Fisher's exact test). `networks`
prints each finished run's last step's network, the one its result's works describes (to pass to
`certify` and `check`). `verdicts` is the findings' tables: per cell, the runs working (W10),
certified (the strict certificates' directory) and robustly certified (the branched one's), each with
its Wilson interval (D4); the runs by works and class; and with the long tests' directory (each
network's `check --json` as <run>-step-<j>.json), per class the networks tested and every failure.
`probe` is a study's budget before its launch (wr.probe): its first run's network trained for
--epochs at each of --steps (default 8 and the last), each epoch's training and screen seconds, the
whole check at the last step's longest length, a certificate for a snapped state, and the process's
peak memory after each (measurement.md, section 2: size workers from it and from results' peak_mb).
`diagnose` is what each saved step's network carries (indrajala_studies/window_diagnostics.py):
its accuracy per answer class and each factor's share of each of the target's coordinates, their
zero tests and the answer, written as step-<j>.diagnosis.json beside the network and printed a line
each, a step already diagnosed skipped; --follow keeps diagnosing each new step as it is saved
until the machine's lock is free (the study ended), for deciding early whether a run is going
anywhere. It takes no lock: run it beside the study, one core.
`drift` is where saved networks' answers go wrong along a long input and how far each one's state
moves from where it sat early on (indrajala_studies/window_drift.py): over the codes property's
first --cases at --n, the wrong digits by position (--bins equal bins) and by the target's state
entering and leaving the column, and per target state each bin's state centroid's shift from
the first bin's and its spread, beside the separation of the first bin's centroids. No lock; one core.
`replay` is a step replayed from a saved network (ml's indrajala_studies/window_replay.py,
the snap wall workplan, retired: git show 53d7fef:docs/workplans/snap-wall-workplan.md): --epochs of --step's data of --run in --config, each epoch drawn by
--generator; per batch the loss before and after and the update's norm, per epoch the screen, the
certificate in both verdicts and the code margins (the least snap ratio; per snapped factor the
codes in use, their closest pair, the farthest state), these also at the start (--epochs 0: only
these). --numpy converts the network first;
--wall turns the snap wall on (numpy only); --machines reads the machine off the network after
every batch and logs each change against the last (pairs changed, states gained and lost, wrong
pairs); --guard undoes an update that makes a right machine wrong (weights, optimizer and generator
restored) and skips its batch. No lock; one core.

OPENBLAS_NUM_THREADS=1, as every study.

The files below are in the factored state's layout: on pyramidon, the factored-state workplan's
stage 5 migrated them (2026-10-14, `migrate_factored_state.py tree data/window-recurrent`, 3,452
files; each certificate certified again the same, field by field), so each manifest's sha256 is the
migrated one, its hash before in "(was ...)". Each old manifest is kept beside it as
`<name>.before-factored-state.sha256`, and every file's old and new sha256 is in
`data/window-recurrent/factored-state-migration.tsv` (7fdb7e79...0fe237e6). jebel's copies were
migrated the same way on 2026-10-15 (2,610 files; its table db181039...1f130032), each manifest
both machines hold byte for byte the same.

Findings (stage 9), on jebel (i7-9700K), indrajala-ml 6b94933, crate 8c20e510c9a9,
`scripts/studies/configs/window_recurrent_matrix.json`: the curriculum arm, `k` 1, 2, 4 x `s` 4, 16 x `m` 4, 8,
16, 5 seeds, 3 steps (1 window, then 1-2, then 1-3), at most 200 epochs a step, patience 30, the
catalogue's default sizes, each step's test pass up to its first failing length (W12), 6 workers.
90 runs, none raised. Wall time 2026-10-12 21:43 to 2026-10-13 05:14 (7.5 h; 43.4 h of runs, 24.6
of them training and 18.8 testing); a run took 2-67 minutes (median 27). The results are on jebel
and pyramidon in `data/window-recurrent/matrix/results/` (90 files, listed with their sha256 in
`data/window-recurrent/matrix/results.sha256`, itself sha256 a8130f8c...4d98fc38 (was 407d6ec3...7941cab9)), every step's
network beside them in `models/` (156 MB), and the log in `data/window-recurrent-matrix.log`
(sha256 b47626f0...68a7dd59).

- Half the runs work (W10: a break of 16 columns or more): 45 of 90. 43 of them pass every
  property at every test length up to 64 columns, after training on at most 3 windows: 3, 6 and 12
  columns at `k` 1, 2 and 4, window ratios of 21.3, 10.7 and 5.3. The other two break at 24 and 32
  (`k` 4, one B1 case in 100,000 wrong at the next length). In every working run the carry reaches the
  whole length (`L*` = n at every test length).
- It takes the recurrence: no run works after step 1, which trains one window and so never hands
  a state on; 9 work from step 2 (2 windows; at `k` 1, two columns of training add 64), 36 more
  from step 3, and none that worked at step 2 stopped working at step 3. A working run's step 3
  passed its check in a median of 2 epochs.
- Failure is mostly all or nothing: 43 of the 45 failing runs fail at 8 columns already, most
  across B1, A2, M2-M5 (their step 3 best screens: median 0.924). A few are near misses: two `k` 4
  runs break at 12 (their trained length) with 1-2 cases in 100,000 wrong at 16, and one `k` 2 run
  misses 4 of B1's 100,000 at 8 (and a few of A1's, M3's and M5's).
- Passing the check isn't working at `k` 1: 5 runs passed their step 3 check (every triple at 1-3
  columns) and still got 0.2-8.3% of B1 wrong at 8.
- By axis: `k` 1, 2, 4 work in 16, 11 and 18 of 30 runs; `s` 4 and 16 in 24 and 21 of 45 (the
  state's width barely matters); `m` 4, 8 and 16 in 10, 16 and 19 of 30 (more codes, more working
  runs). No cell works on every seed: the best, 4 of 5, are `k` 1 `s` 16 `m` 8, `k` 4 `s` 4 `m` 16
  and `k` 4 `s` 16 `m` 16; the worst, `k` 2 `s` 16 `m` 4, on none. Within a cell the seed decides,
  so what fails is the optimisation, not the capacity: every cell has a state that can hold a
  carry (3 values), and 6 working runs use exactly 3 codes.
- The codes used by working runs: 3-11 (median 5) over the lengths tested, so a working network
  uses more codes than the carry's 3 values; the probes (stage 10) say what the others hold.
- Cost: a step 3 epoch trains in a median 6.8 s; a working run's step 3 test pass (every length)
  took a median 17.6 minutes, a failing one's (length 8 alone) about a minute.

Per cell, its seeds working and each seed's break (in columns, seeds 0-4) after steps 2 and 3, and
the codes the working seeds used:

| k | s | m | parameters | seeds working | breaks after step 2 | breaks after step 3 | codes used (working) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 7,683 | 2/5 | 0/64/64/0/0 | 0/64/64/0/0 | 4,4 |
| 1 | 4 | 8 | 7,699 | 3/5 | 0/0/0/0/0 | 0/64/64/0/64 | 8,6,7 |
| 1 | 4 | 16 | 7,731 | 2/5 | 0/0/0/0/0 | 0/64/64/0/0 | 11,11 |
| 1 | 16 | 4 | 9,651 | 2/5 | 64/0/0/64/0 | 64/0/0/64/0 | 4,4 |
| 1 | 16 | 8 | 9,715 | 4/5 | 0/0/64/0/0 | 64/64/64/0/64 | 6,4,4,7 |
| 1 | 16 | 16 | 9,843 | 3/5 | 0/0/0/0/0 | 64/64/64/0/0 | 7,7,6 |
| 2 | 4 | 4 | 7,699 | 1/5 | 0/0/0/0/0 | 0/0/64/0/0 | 3 |
| 2 | 4 | 8 | 7,715 | 3/5 | 0/64/0/0/0 | 0/64/64/0/64 | 4,6,4 |
| 2 | 4 | 16 | 7,747 | 3/5 | 0/0/0/0/0 | 0/64/0/64/64 | 8,7,5 |
| 2 | 16 | 4 | 9,667 | 0/5 | 0/0/0/0/0 | 0/0/0/0/0 | - |
| 2 | 16 | 8 | 9,731 | 1/5 | 0/0/0/0/0 | 64/0/0/0/0 | 6 |
| 2 | 16 | 16 | 9,859 | 3/5 | 0/0/0/0/0 | 0/0/64/64/64 | 9,4,8 |
| 4 | 4 | 4 | 7,731 | 3/5 | 8/0/64/0/64 | 64/0/64/0/64 | 4,3,3 |
| 4 | 4 | 8 | 7,747 | 3/5 | 0/0/12/0/0 | 0/64/64/12/64 | 4,5,5 |
| 4 | 4 | 16 | 7,779 | 4/5 | 0/0/0/0/8 | 64/64/12/32/64 | 7,5,7,7 |
| 4 | 16 | 4 | 9,699 | 2/5 | 0/0/0/0/8 | 0/64/0/0/64 | 3,3 |
| 4 | 16 | 8 | 9,763 | 2/5 | 64/0/0/0/0 | 64/0/0/64/0 | 4,3 |
| 4 | 16 | 16 | 9,891 | 4/5 | 8/0/0/0/0 | 64/24/64/64/0 | 5,6,4,6 |

Findings (the control, W16 and W30), on jebel, indrajala-ml 1dd22f9, crate 8c20e510c9a9,
`scripts/studies/configs/window_recurrent_control.json`: the all-at-once arm (one step on window counts 1-3
together, at most 200 epochs) at the matrix's three best cells, seeds 0-4, the matrix's settings
otherwise. 15 runs, none raised. Wall time 2026-10-13 05:28 to about 06:29 (5.3 h of runs; a run
took 1-33 minutes, median 23). The results are on jebel and pyramidon in
`data/window-recurrent/control/results/` (15 files, listed with their sha256 in
`data/window-recurrent/control/results.sha256`, itself sha256 bbd0e9bb...a8e46949 (was 6eed795f...a6facf9b)), the networks
beside them in `models/` (9.2 MB), and the log in `data/window-recurrent-control.log` (sha256
913e996f...c52662dd).

- The curriculum isn't needed here: the control works as often, 12 of 15 runs against the
  curriculum's 12 of 15 at the same cells and seeds. At `k` 4 it works on all 10 (curriculum 8 of
  10; Fisher's exact test p 0.47); at `k` 1 `s` 16 `m` 8 on 2 of 5 (curriculum 4 of 5; p 0.52).
  At 5 seeds neither difference is decided; the claims workplan's E4 measures it at 20.
- It needs far fewer examples where it works: at `k` 4 it passes its check within 4-16 epochs,
  0.26-1.05 M examples (median 0.39 M), against the curriculum's 4.8-12.3 M over its three steps
  for the runs that work (median 6.4 M), about 16 times fewer. The curriculum spends 78-99% of
  its examples on its first two steps (one window, then one or two; 61 of the matrix's 90 runs
  stop step 1 on the plateau), and once it trains on 1-3 windows it passes in a few epochs, as the
  control does from the start.
- Which seeds work isn't shared between the arms: at `k` 4 the curriculum's two failing seeds
  (seed 2 of `s` 4, seed 4 of `s` 16) work all at once, and at `k` 1 the arms fail on different
  seeds (all at once: 0, 2, 4; curriculum: 3).
- Exact on short sums still isn't generalising: `k` 1's seed 2 passed its check (every sum of 1-3
  columns) in 3 epochs and failed at 8, as five of the matrix's `k` 1 runs did.

| cell | all at once: working, breaks (seeds 0-4) | epochs | curriculum: working, breaks after step 3 |
|---|---|---|---|
| `k` 1 `s` 16 `m` 8 | 2/5, 0/64/0/64/0 | 42, 1, 3, 3, 46 | 4/5, 64/64/64/0/64 |
| `k` 4 `s` 4 `m` 16 | 5/5, 64/64/64/48/64 | 13, 6, 7, 5, 6 | 4/5, 64/64/12/32/64 |
| `k` 4 `s` 16 `m` 16 | 5/5, 64/64/64/64/64 | 5, 4, 16, 6, 9 | 4/5, 64/24/64/64/0 |

Findings (E1 of indrajala-research/docs/research-notes/window-recurrent-report.md: the certificate and the long tests), on
pyramidon: `certify` over every network the matrix and the control tested (the matrix's step 3, the
control's step 1: 105), 4-7 workers. Two verdicts (the claims workplan's D3): **robustly
certified**, every near tie (snap ratio under 1.05) branched both ways, indrajala-ml 094a7b1
(#703), 2026-10-13 about 07:50 to 08:26, in `data/window-recurrent/certificates/` (105 files,
listed with their sha256 in `data/window-recurrent/certificates.sha256`, itself sha256
38b1efee...116f32fd (was 67a4e6b7...9550f05d)), log `data/window-recurrent-certificates.log` (sha256 f8c38080...56ae98a1); and
**certified**, the network's own snaps only (`--snap-ratio 1`), indrajala-ml 35175ea (#704; the
certificate unchanged but for the option), 08:30 to 09:22 in two parts, in
`data/window-recurrent/certificates-strict/` (`certificates-strict.sha256`, sha256
f39c74b1...573ddb88 (was 299675b8...a924fc18)), log `data/window-recurrent-certificates-strict.all.log` (the two parts
joined; sha256 e1b8e612...3dcce757). 105.9 M window evaluations each way; 3.4 h of worker time
branched, 2.9 h strict; a `k` 4 network 115-689 s branched, 74-759 s strict.

- No failing network is certified, either way: of the 48 networks that break before 16 columns
  (W10's failing), none: here the certificate passes nothing the tests fail.
- Of the 57 working networks (W10), 52 are certified and 48 robustly certified (matrix 42 and 38
  of 45, control 10 and 10 of 12). Every working `k` 1 network is certified both ways; by `k` 1, 2
  and 4, working 18, 11 and 28, certified 18, 11 and 23, robustly 18, 8 and 22.
- Every certified network's machine minimises to 3 states, the carry automaton, while reaching 3-11
  codes (median 5; exactly 3 in 9 certified networks, 8 robustly): the extra codes are equivalent
  copies of a carry value. The smallest digit lead of a certified network
  is 0.153 (a logit difference).
- 16 robust certificates rest on near ties, up to 116,944 a network, at snap ratios down to
  1.0000002: real, not duplicate codes (each codebook's closest pair is 0.02-0.3 apart). The 4
  networks certified but not robustly (matrix `k2-s16-m16-seed3` and `seed4`, `k2-s4-m8-seed2`,
  `k4-s16-m4-seed1`) are correct as they compute and wrong only when a near tie (ratio
  1.006-1.03) snaps the other way.
- **The sampled tests overstate working (C1 refuted for W10's "works", D13).** 5 working networks
  aren't certified: 3 break before 64 columns, as their tests said (matrix `k4-s16-m16-seed1` at
  24, `k4-s4-m16-seed3` at 32, control `k4-s4-m16-seed3` at 48), and **2 passed every test to 64
  columns and are wrong**: control `all-at-once-k4-s16-m16-seed2` (1559 + 6329 + 5231 = 13,119
  at 8 columns; the network says 19,680) and matrix `curriculum-k4-s4-m8-seed2` (514107 + 167355
  + 27153 = 708,615 at 12; it says 531,468), each confirmed by running the network on
  `wr.example(triple, n - k, k, 0)`, `n` the certificate's length (a carry counterexample shows only at alignment 0, which the
  harness's hash picks for few cases).
- The failing networks' counterexamples (strict): 42 a wrong carry, 11 a wrong digit, after 1-4
  windows (22, 18, 6 and 7 networks), so every failure shows within 4 windows. Their machines
  minimise to 2-9 states.

The long tests (D9), on jebel: `check --n 128 256 1024 --properties B1 M3 --doubling` on the 59
networks certified or near a miss (a break of 12 or more): the 52 certified, the 2 counterexample
networks and 5 more not certified. B1's 100,000 cases at each length and M3's 1,000 at each
distance 1, 2, 4, ... `n`: 328,000 cases a network, 19.35 M in all. indrajala-ml e1b5be3 (#704's
branch; its package as 35175ea's), 7 workers, 2026-10-13 09:38 to 14:00, in
`data/window-recurrent/long-tests/` (a `.json` and a `.log` a network, listed with their sha256 in
`data/window-recurrent/long-tests.sha256`, itself sha256 4f3542f4...9666ec77), log
`data/window-recurrent-long-tests.log` (sha256 c5a5869d...35b49106; its "exit" figures are
always 0, a shell slip: read each network's `.log` and `.json`).

- **Every certified network passes every case**: the 48 robustly certified and the 4 certified
  only strictly, 17.06 M cases, none wrong. The certificate is borne out where it can be tested.
- Every network that fails a long test is one the certificate rejects: the 5 not certified fail
  B1 at every length (1-453 of 100,000 at 1,024), but for matrix `k4-s4-m16-seed3` at 128,
  and one of them an M3 case too (matrix `k4-s16-m16-seed1` at 256).
- **The long tests miss what the certificate finds.** Of the 2 counterexample networks, control
  `all-at-once-k4-s16-m16-seed2` fails 1 case of 328,000 (B1 at 1,024) and matrix
  `curriculum-k4-s4-m8-seed2` passes all 328,000, though it is wrong at 12 columns: a sampled test,
  however long, doesn't establish working.

Per cell, its networks working (W10), certified and robustly certified, of 5:

| arm | k | s | m | working | certified | robustly certified |
|---|---|---|---|---|---|---|
| curriculum | 1 | 4 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 4 | 8 | 3 | 3 | 3 |
| curriculum | 1 | 4 | 16 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 8 | 4 | 4 | 4 |
| curriculum | 1 | 16 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 4 | 4 | 1 | 1 | 1 |
| curriculum | 2 | 4 | 8 | 3 | 3 | 2 |
| curriculum | 2 | 4 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 16 | 4 | 0 | 0 | 0 |
| curriculum | 2 | 16 | 8 | 1 | 1 | 1 |
| curriculum | 2 | 16 | 16 | 3 | 3 | 1 |
| curriculum | 4 | 4 | 4 | 3 | 3 | 3 |
| curriculum | 4 | 4 | 8 | 3 | 2 | 2 |
| curriculum | 4 | 4 | 16 | 4 | 3 | 3 |
| curriculum | 4 | 16 | 4 | 2 | 2 | 1 |
| curriculum | 4 | 16 | 8 | 2 | 2 | 2 |
| curriculum | 4 | 16 | 16 | 4 | 3 | 3 |
| all at once | 1 | 16 | 8 | 2 | 2 | 2 |
| all at once | 4 | 4 | 16 | 5 | 4 | 4 |
| all at once | 4 | 16 | 16 | 5 | 4 | 4 |

Findings (E7 of indrajala-research/docs/research-notes/window-recurrent-report.md: modular division, C7), trained on jebel
(i7-9700K), indrajala-ml 67fc12f, crate 8c20e510c9a9, 6 workers, 2026-10-13 14:58 to 20:42
(`window_recurrent_e7_p5.json` then `_p7.json`: 2 `p` x 2 arms x `k` 1 and 4 x `m` 32 and 64 x 5
seeds, `s` 16, the matrix's settings). Results and models in `data/window-recurrent/e7-p5/` and
`e7-p7/` (listed with their sha256 in `data/window-recurrent/e7-p5.sha256`, itself sha256
9d8fd606...70bd96e7 (was 6ca9ee8c...a992317f), and `e7-p7.sha256`, da982e22...25128a4a (was 10d0b5bd...0f5a2b7a)), copied to pyramidon and checked;
logs `data/window-recurrent-e7-p5.log` (196cd344...0776527a) and `-p7.log` (2c06d807...fbbc1349).
Certified on pyramidon at indrajala-ml 9c50fe6, 3 workers, both verdicts (D13), in each
directory's `certificates/` and `certificates-strict/` (`certificates.sha256`: p 5 cdf49604...57b9b420 (was 6d0b3e11...46d4cb7d),
p 7 0fa7a105...153386c6 (was 4cae99c0...5426ada6); `certificates-strict.sha256`: p 5 d5d75c8c...a2b56e03 (was 43c9d4af...88c26033), p 7
17588161...168d5f57 (was dc612ff5...a5d4eff5)); about 140 s a pass of 40 networks. No long tests: none certified (D22).

- **C7 is refuted as registered**: no run works and no network certifies, either way, at either
  `p`: 0 of 40 at `p` 5 and 0 of 40 at `p` 7 (Wilson 95%: 0.00-0.09 each), every cell 0 of 5.
- **E7's design was flawed** (found during its p 7 runs; the claims workplan's E7, D23): the
  automata minimise to exactly 25 and 49 states, so `p` 7 at `m` 32 (20 runs) can't be the
  automaton; they ran as registered, a negative control, and fail as they must. The other cells
  had 1.3 and 2.6 times the states' codes, the low end of what worked on addition.
- **At `k` 4 no run fits its training lengths**: best screens 0.11-0.34 at every `m` and `p`;
  the certificates find a wrong next state in the first window of every network.
- **At `k` 1 every run is a near miss at length 3**, the first length that uses every transition
  (every state is reachable within 2 columns): the curriculum passed lengths 1 and 2 every time
  and plateaued (never the budget: D26) on length 3, whose U1 is all 729 pairs; at `p` 5 the
  networks got 10-342 of them wrong, median 172 at `m` 32 and 85 at `m` 64 (doubling the codes
  about halved the errors; best, all-at-once `m` 64 seed 0, 10 wrong, a best screen of 0.996); at
  `p` 7, 67-500, median about 136 at `m` 64. Every certificate's counterexample is a wrong next
  state after 2 windows (one `p` 7 network's after 1).
- The `k` 1 `m` 64 networks reach codes for most of the automaton's states (the certificate's
  `phi` covers 23-25 of `p` 5's 25 and 31-46 of `p` 7's 49) from 37-60 codes, but their
  transitions between them don't close: the codes are there, the dynamics aren't. Every run
  used almost all its codes (a median of 31 of 32 and 62-63 of 64 per cell), as addition's
  working runs did (14 of 16).
- E7' (the claims workplan) is the test E7 meant to be: `k` 1 at 5 and 10 times the states'
  codes, 10 seeds a cell.

Findings (E7' of indrajala-research/docs/research-notes/window-recurrent-report.md: modular division with enough codes,
C7'), trained on jebel, indrajala-ml 9c50fe6, crate 8c20e510c9a9, 6 workers, 2026-10-13 20:45 to
23:56 (`window_recurrent_e7_prime_p5.json` then `_p7.json`: 2 `p` x 2 arms x 2 `m` x 10 seeds,
`k` 1, `s` 16, E7's settings; `m` 128 and 256 at `p` 5, 256 and 512 at `p` 7, about 5 and 10
times the states). Results and models in `data/window-recurrent/e7-prime-p5/` and `-p7/`
(`e7-prime-p5.sha256`, sha256 273f7b77...c788cf6f (was 67f56ab0...05dc44d8); `e7-prime-p7.sha256`, 4bad23ad...82834fe8 (was 69c4efe2...4fd8c79e)),
copied to pyramidon and checked; logs `data/window-recurrent-e7-prime-p5.log` (eab06819...c7a1b8ac)
and `-p7.log` (a258d10a...8da00eea). Certified on pyramidon at indrajala-ml cf6ac86, both verdicts
(`certificates.sha256`: p 5 1e889358...ac000655 (was de4a138c...10c88e98), p 7 948929fe...63f18945 (was 1453332d...d9b5cc27);
`certificates-strict.sha256`: p 5 9e8d34c5...9e1bd422 (was eb3530ac...3a9703aa), p 7 e38b3bf2...59c3f8ab (was cd76b51e...aca8370f)). No long tests:
none certified.

- **C7' is refuted as registered**: no run works and no network certifies, either way: 0 of 40
  at `p` 5 and 0 of 40 at `p` 7 (Wilson 95%: 0.00-0.09 each), every cell 0 of 10.
- **The codes are no longer the limit.** The networks now hold codes for (nearly) every state
  of the automaton: the certificate's `phi` covers all 25 of `p` 5's states in every all-at-once
  `m` 256 network, and 47-49 of `p` 7's 49 at all-at-once `m` 256, from 30-154 codes reached. Yet
  79 of the 80 counterexamples are a wrong next state (76 after 2 windows, 2 after 1, 1 after 3),
  and one a wrong output: the states are represented, the transitions between them aren't exact.
- More codes cut the errors at length 3, the first to use every transition (U1, all 729 pairs),
  but none reach zero. Median and best pairs wrong, `k` 1, against E7's `m` 64:

| `p` | arm | E7 `m` 64 | E7' 5 times | E7' 10 times |
|---|---|---|---|---|
| 5 | all-at-once | 88 (best 10) | 25 (8) | 28 (4) |
| 5 | curriculum | 79 (39) | 93 (43) | 38 (23) |
| 7 | all-at-once | 195 (113) | 60 (24) | 51.5 (28) |
| 7 | curriculum | 113 (67) | 106 (41) | 70 (27) |

- The best network, `p` 5 all-at-once `m` 256, is 4 pairs of 729 from exact at length 3; from 5
  to 10 times the codes the all-at-once medians stop improving (25 to 28, 60 to 51.5).
- **6 of E7''s 160 steps stopped on the budget while still improving** (best screen within the
  last 21 epochs of 200: five at `p` 7, four of them curriculum step 3 at `m` 256-512, best
  screens 0.94-0.99). D26 kept E7's budget (no `k` 1 budget stop in E7); E7' shows a few runs it
  cut short, none near exact (best screens under 0.99).

Findings (E8 of indrajala-research/docs/research-notes/window-recurrent-report.md: the modular-sum ladder's first rungs,
C8), trained on pyramidon (Ryzen 7 3700U), indrajala-ml b886001, crate 8c20e510c9a9, 3 workers,
2026-10-14 05:50 to 06:19 (`window_recurrent_e8_q4.json` then `_q5.json`: `q` 4 and 5 x 2 arms x
10 seeds, `k` 1, `s` 16, `m` 32, E7''s settings). Results and models in
`data/window-recurrent/e8-q4/` and `-q5/` (`e8-q4.sha256`, sha256 9d5ea06c...cdebfff2 (was 011ee2de...596754fd);
`e8-q5.sha256`, 97f2466f...9500e30f (was 8ec6a2ae...ebb5ca80)), logs `data/window-recurrent-e8-q4.log` (51504a22...646b3d13)
and `-q5.log` (36923d55...d9b94fce). Certified on pyramidon, both verdicts
(`certificates.sha256`: q 4 a5909738...c98d5c11 (was 184bc355...1590b51e), q 5 c246a596...5becea37 (was 0fc8b159...d3f8c535);
`certificates-strict.sha256`: q 4 3871662f...402f020d (was 852c0b1b...90a1eafb), q 5 81e5b156...48aa8e9c (was 9ced897f...bd3031ff)); the certified
network long-tested (U1 and S1 at 128, 256 and 1,024; `e8-q4/long-tests.sha256`,
a4ca4818...ee347172).

- **C8 holds at `q` 4, weakly; not at `q` 5.** One network certifies (both verdicts), q 4
  all-at-once seed 3: its 8 codes minimise to the 4-state automaton, and it passes all 312,000
  long-test cases. 1 of 20 at `q` 4 (Wilson 0.01-0.24), 0 of 20 at `q` 5 (0.00-0.16).
- **The failures are memorising, not fitting** (the opposite of E7's): 37 of the 40 runs passed
  every check at lengths 1-3, most in 2-20 epochs, and failed at the first test length (8). Their
  codes minimise to machines mostly larger than the task's: at `q` 4, 17 of the 19 failing
  networks' machines have 6-15 states (one 2, one 4), at `q` 5 all 20 have 7-22, against 4 and 5;
  the counterexamples come mostly after 3-4 windows, just past the training lengths.
- **Why the design allowed it** (D23's exclusion bound, added after E8): a network of `m` codes
  exactly right on every input up to length `L` is the `q`-state machine only if
  `L >= m + q - 1` (two machines of `n` and `n'` states that agree on every input up to `n + n' - 1`
  are equivalent). E8 trained to 3 columns with 32 codes, where 35-36 were needed; so did every
  earlier experiment (addition's `k` 1 needed 6-18, its `k` 4 trained to 12). E8b is E8 trained
  past the bound.

Findings (E3 and E4 of indrajala-research/docs/research-notes/window-recurrent-report.md, its stage 7: the code count and the
curriculum against all at once, C3-C5; C0's reruns), trained on jebel (i7-9700K), indrajala-ml
cf6ac86, crate 8c20e510c9a9, 6 workers, finished 2026-10-14 12:41 (`window_recurrent_e3_e4_trimmed.json`,
D34: `k` 4 `s` 4, the curriculum at `m` 3, 4, 6, 8 and 16 and all at once at `m` 16, seeds 0-19,
120 runs). Results and models in `data/window-recurrent/e3-e4/` (`e3-e4.sha256`, sha256
1dc4d875...d9a54341 (was 448dd136...652dda8a)), log `data/window-recurrent-e3-e4.log` (52f0b1df...133d5f0a). Every run's
last network certified on pyramidon at 9dce80e, both verdicts (`e3-e4/certificates.sha256`,
7005c870...5745770c (was 37c1c694...42a38caa); `certificates-strict.sha256`, eb10292c...ef837eff (was a68ed4f2...3b74f7a3); logs `certify-branched.log`
60f328ea...b0d93bc8, `certify-strict.log` f8ffc9b7...7dc7a7f3). The registered tests are D4's on
the working rate (W10, exact to 16 columns and past); the certified rate beside them is a
measurement. The p values here are raw; Holm's correction over C2-C4 and C6, registered (D4),
came with E5 (its findings): C3 0.042, still holding; C4 0.088, so C4's refutation doesn't stand.

| arm | `m` | working | certified (strict) | robustly certified | working, not certified |
| --- | --- | --- | --- | --- | --- |
| curriculum | 3 | 3/20 (0.05-0.36) | 3/20 (0.05-0.36) | 2/20 | 0 |
| curriculum | 4 | 11/20 (0.34-0.74) | 10/20 (0.30-0.70) | 10/20 | 1 |
| curriculum | 6 | 8/20 (0.22-0.61) | 6/20 (0.15-0.52) | 6/20 | 2 |
| curriculum | 8 | 11/20 (0.34-0.74) | 7/20 (0.18-0.57) | 6/20 | 4 |
| curriculum | 16 | 13/20 (0.43-0.82) | 8/20 (0.22-0.61) | 8/20 | 5 |
| all at once | 16 | 19/20 (0.76-0.99) | 12/20 (0.39-0.78) | 11/20 | 7 |

- **C0 holds: 15 of 15 reruns reproduced exactly** (the matrix's `k` 4 `s` 4 runs, 6b94933,
  against their reruns here: results up to timings, and every step's weights, optimizer state and
  generator, `compare`).
- **C3, as registered, holds: the working rate rises with `m`** (Cochran-Armitage over `m` 3-16,
  scores `m`: z 2.45, p 0.014). **But the certified rate doesn't** (z 0.79, p 0.43): from `m` 4 on
  it stays near 0.3-0.5, and the extra codes buy working networks that aren't the automaton
  (working but not certified: 0, 1, 2, 4, 5 of 20 from `m` 3 to 16). Under C1's restatement
  (certified is what works at every length), extra codes don't make training find the automaton
  more often; `m` 3, exactly the automaton's size, is the one low cell (3 of 20).
- **C4 holds as registered, after Holm's correction:** the arms' working rates don't differ at
  0.05 (all at once 19 of 20 against the curriculum's 13 of 20 at `m` 16, Fisher's exact p 0.044
  raw, 0.088 adjusted over C2-C4 and C6; read raw, before E5, this was reported as refuted, all at
  once working more often). All at once leans higher, on far fewer examples (working runs'
  medians 393,216 against 7,798,784, Mann-Whitney p 2e-6, every all-at-once run below every
  curriculum one). Certified, the arms don't differ (12 against 8 of 20, p 0.34): all at once's
  extra working runs are mostly ones the certificate fails (7 of its 19).
- **Every certified network, in every cell, minimises to the 3-state carry automaton**, from 3
  reachable codes (at `m` 3) to 10 (all at once). Of the 19 working networks not certified, 14 give
  a wrong digit and 5 a wrong carry on some reachable code and window that the sampled tests to 64
  columns missed; none fails only on a label's margin.
- **C5 (a measurement):** the rates above with their Wilson intervals; at 20 seeds a cell they
  span about 0.4 each.
- **The steps:** the curriculum's step 1 (one window) plateaus in all 100 runs; every working run
  passes step 3 (lengths 1-3), 19 of the 46 at step 2 already. The failures differ with `m`: at `m`
  3-6 they never fit (all three steps plateau, 36 of the 38 failures there), at `m` 8-16 they mostly
  fit (12 of 16 pass step 3) and break at 8-12 columns: memorising, as the exclusion bound (D23)
  predicts for training to 3 windows (12 columns) with 8-16 codes.

Findings (E8c of indrajala-research/docs/research-notes/window-recurrent-report.md: the modular sum as an existence test,
C8, D35), trained on jebel (i7-9700K), indrajala-ml 713ff5f, crate abcba3d7a30e, 6 workers,
2026-10-14 14:19 to 14:31 (`window_recurrent_e8c_q4.json` then `_q5.json`: E8b's curriculum cell,
`k` 1, `s` 16, `m` 16, 20 steps, the cap the bound `m + q - 1`; each step's network certified, a
run ended at its first certified step, the study at its first certified run; 6 seeds a rung).
Results and models in `data/window-recurrent/e8c-q4/` and `-q5/` (`e8c-q4.sha256`, sha256
e285e4cb...72b1d83c; `e8c-q5.sha256`, 14f9395b...1b1a139c; 85 files each, run from
`data/window-recurrent`), log `data/window-recurrent-e8c.log` (4b583cb0...cde878; `q` 4's run
ended in the report's crash on stopped runs, fixed in #747, after every result was written).
Strict certificates in `e8c-q4-strict/` and `e8c-q5-strict/` (`e8c-q4-strict.sha256`,
2503c79c...12901741; `e8c-q5-strict.sha256`, 09ef995d...23b425a6). Long tests (U1 and S1 at 128,
256 and 1,024, one worker, on pyramidon: `q` 4 from 3b6cb12, `q` 5 from 867d356, the same
package code) in `e8c-q4/long-tests/` and `e8c-q5/long-tests/`, log
`e8c-long-tests.log` (`e8c-long-tests.sha256`, a5405265...42e949af, 10 files).

- **C8 holds at both rungs, as an existence claim: at least 3 of 6 networks certify at `q` 4 and
  6 of 6 at `q` 5.** `q` 4: seeds 1, 3 and 5 at step 4; `q` 5: seeds 0, 2, 4 and 5 at step 4,
  seeds 1 and 3 at step 5; "at least", since parallel runs stop when the first certifies (the
  three `q` 4 runs stopped were at steps 4-6). Each run took 4.5-7 minutes. Both verdicts: the
  harness's per-step certificate (near ties branched) and the strict one (`--snap-ratio 1`).
- **Each minimises to exactly the `q`-state automaton**, from 7-9 codes reached at `q` 4 and 7-13
  at `q` 5, with 0 wrong outputs and states and no near tie; snap ratios 1.02-7.91 and logit leads
  1.99-7.92 (the smallest each network's).
- **The long tests agree:** every one of the 9 passes every case, 312,000 each at `q` 4 and
  315,000 each at `q` 5 (35-61 minutes a network).
- **What changed from E8 and E8b** (no network at `q` 5 in E8, 0 of 20): training past the
  exclusion bound with the curriculum (E8b's cell), and certifying every step, so a run stops at
  the machine it found instead of training on past it (E8b: training past a certified step can
  destroy the network). Learning a counter used about twice its states in codes.

Findings (E6 of indrajala-research/docs/research-notes/window-recurrent-report.md, re-registered by D37: the window without
its transformer, as an existence test), trained on pyramidon (Ryzen 7 3700U), indrajala-ml 9114f21,
crate abcba3d7a30e, 3 workers (memory: E8c's long tests ran beside it), 2026-10-14 17:06 to 17:26
(`window_recurrent_e6_linear.json`: the linear window, the embedding and output layer only, at `k`
1 `s` 16 `m` 16 on addition, the curriculum capped at 18 columns, the bound `m + q - 1`, each step
certified, the study stopped at its first certified run, 6 seeds). Results, models and per-step
certificates in `data/window-recurrent/e6-linear/` (56 files, `e6-linear.sha256`, sha256
716761f3...62f2e5da, run from `data/window-recurrent`), log `data/window-recurrent/e6-linear.log`
(9a6c848b...cba8648d; its run ended in the report's crash on runs stopped before their first epoch,
fixed with these findings, after every result was written). Strict certificates of the two
certified networks in `e6-linear/certificates-strict/` (listed in the same manifest).

- **The window's transformer isn't needed at `k` 1: at least 2 of 6 runs certify.** Seed 1 at step
  3 (trained to 3 columns) and seed 2 at step 4 (4 columns), both verdicts (the harness's per-step
  certificate branches near ties; the strict one, `--snap-ratio 1`, agrees), each minimised to the
  3-state carry automaton from 5 and 8 codes reached, 0 wrong outputs and states, no near ties, snap
  ratios at least 2.99 and 1.37, logit leads at least 5.01 and 2.29. Both pass every test length to
  64 columns. "At least": seeds 1 and 2 certified in parallel before the stop reached them.
- **A network of 2,963 parameters** (the transformer window's `k` 1 networks: 9,651-9,843 at `s`
  16) certifies, so at one column a window the GRU does the whole step; the FFN window (D37's
  fallback) wasn't run.
- The others: seed 0 fitted steps 1-3 and plateaued from step 4 until stopped at step 7; seeds 3-5
  were stopped before their first epoch (3 workers). No rate is claimed (D37).

Findings (E2 of indrajala-research/docs/research-notes/window-recurrent-report.md, trimmed by D36: the snapping ablation, C2),
trained on jebel (i7-9700K), indrajala-ml ce4015d, crate abcba3d7a30e, 6 workers, 2026-10-15 16:54
to 21:29 (`window_recurrent_e2.json`: E3's curriculum `m` 16 cell at `k` 4 `s` 4 run by run with a
continuous state, `m` null, seeds 0-19; 7,715 parameters, 38-109 minutes a run, 379 MB peak).
Results and models in `data/window-recurrent/e2/` (`e2.sha256`, sha256 a02fe6a5...0696d5d9d, 140
files, run from `data/window-recurrent`), log `e2.out` (951b9516...0b7b507a). The snapped arm is
E3's (above). Its 9 runs whose break is 64 long-tested (B1 at 256 and 1,024, 100,000 cases each)
on jebel at ce4015d, 2 workers, into `e3-e4/long-tests/`, log `e2-long-tests.log`
(`e2-long-tests.sha256`, 9f91bb11...80957bd4, 10 files, run from `data/window-recurrent`).

| arm | working at 64 (W10) | break 64 | working at 1,024 |
| --- | --- | --- | --- |
| continuous (E2) | 3/20 (0.05-0.36) | 0/20 | 0/20 (0.00-0.16) |
| snapped (E3, `m` 16) | 13/20 (0.43-0.82) | 9/20 | 8/20 (0.22-0.61) |

- **C2 holds: the continuous state works far less often, at 1,024 columns and at 64.** At 1,024,
  0 of 20 against 8 of 20 (Fisher's exact, two-sided, p 0.0033); at 64, 3 of 20 against 13
  of 20 (p 0.0031). No continuous run got past 24 columns (breaks 16, 16 and 24 among the three
  working), so none was long-tested: the registered measure needs a break of 64. Raw p values;
  Holm-adjusted over C2-C4 and C6 (E5's findings), 0.013 at 1,024: C2 still holds.
- **The snap matters for extrapolating, not for fitting.** The continuous arm fits its training
  lengths about as often (step 3, lengths 4-12, passed by 15 of 20 against E3's 19 of 20), but 12 of
  those 15 break at 12 columns or below, exactly the longest length trained; the snapped arm's
  fitted runs go on to 64 in 9 of 19. The continuous state fits what it is shown and drifts past it
  (Robinson 2020; Brandoit et al. 2026).
- **The long tests agree with the certificates exactly.** The snapped arm's 8 networks working at
  1,024 are its 8 certified ones, each passing all 200,000 cases. The ninth, seed 10, works by the
  sampled tests to 64 columns but isn't certified (25 wrong states, no wrong output, in the branched
  verdict); it passes B1 at 256 and fails 1 case of 100,000 at 1,024. Sampled tests overstate
  "works"; the certificate is what extrapolates.

Findings (E2b of indrajala-research/docs/research-notes/window-recurrent-report.md, D38: the snapping ablation on the plain
recurrence, E6's linear window at `k` 1 `s` 16, the curriculum to 18 columns; **exploratory by D39**:
the snapped arm's runs ended at their first certified step, against the registered "no early stop",
so C2b's registered test is E2b'), crate abcba3d7a30e, code at eae605d (`~/code/wt-e2b`; the results'
"commit" is the launching checkout's), 20 seeds an arm. Snapped (`window_recurrent_e2b_snapped.json`,
2,963 parameters): `pyramidon`, 3 workers, 2026-10-15 18:11 to 22:32, 14-124 minutes a run, 589 MB
peak, every step certified in both verdicts (`e2b-snapped.sha256`, 7d791033...4c43856b, run from
`data/window-recurrent`; `certificates-strict/`, `certify-strict.log`), log `e2b-snapped.out`
(2f0e07ac...c5234237). Continuous (`_continuous.json`, 2,707 parameters, no codebook): `jebel`, 6
workers, 2026-10-15 21:30 to 2026-10-16 06:34, 103-220 minutes a run (median 151), 1,086 MB peak
(`e2b-continuous.sha256`, 4e5bac78...4c97f8fb, 740 files), log `e2b-continuous.out`
(43bc38c6...c3171a89). Long tests (B1 at 256 and 1,024, 100,000 cases each) on the 22 runs whose
break is 64, `pyramidon` at 050757b, 2 workers, 2026-10-16 07:22 to 11:10, about 20 minutes a run, into
`e2b-long-tests/`, log `e2b-long-tests.log` (`e2b-long-tests.sha256`, a8d38c01...2136dd84, 23
files, run from `data/window-recurrent`).

| arm | working at 64 (W10) | break 64 | certified (last step) | working at 1,024 |
| --- | --- | --- | --- | --- |
| snapped | 20/20 (0.84-1.00) | 20/20 | 20/20 | 20/20 (0.84-1.00) |
| continuous | 20/20 (0.84-1.00) | 2/20 | - | 0/20 (0.00-0.16) |

- **At 64 columns the arms don't differ; past it they do.** Every run of both arms works at 64 by
  W10 (a break of 16 or more), but the continuous breaks are 16 (4 runs), 24 (8), 32 (2), 48 (4) and
  64 (2): 18 of 20 fail before 64. At 1,024, the snapped arm works 20 of 20 (every one of
  200,000 cases each) and the continuous arm 0 of 20: both its break-64 runs fail B1, seed 2 on 1
  case of 100,000 at 256 and 16,756 at 1,024, seed 9 on 4 at each (Fisher's exact, one-sided, p
  7.3e-12; exploratory, D39).
- **The snapped GRU learns the carry automaton from a few columns.** Its runs certified at steps 3
  (6 runs), 4 (7), 5 (2), 6, 7, 8, 10 and 15, step `j` training lengths 1 to `j`: 13 of 20 from at
  most 4 columns, every network minimised to the 3-state carry automaton from 4-14 codes reached.
  Below the exclusion bound (18) training can't exclude memorising; the certificate proves these
  networks anyway, so the bound is sufficient, not necessary.
- **The continuous GRU extrapolates a little past what it is shown, then drifts.** It trained all 18
  steps (to 18 columns); its break at the last step was at most 24 columns in 12 of 20 and reached
  64 in 2, at steps 16 and 17 (seeds 2 and 9). The same pattern as E2's, without the transformer.
- **Its state moves along the input; its errors follow** (exploratory, after the data: `drift`, 3,000
  of B1's cases at 1,024, `e2b-drift-1024.json`, a3d57ae6...2cf3971). Every long-test failure is one
  or two wrong digits, not a run of them, and none follows a long run of one carry. Seed 2's
  wrong digits by 128-column bin are 0, 0, 0, 7, 32, 166, 247, 474 (512 of 3,000 cases), and its
  states' centroid for carry 1 moves from the first bin's by 0.32 in the second bin, then steadily
  to 0.76 in the last (0.28-0.84 over the three carries), against a separation of 2.36 between
  carries: the state is not stationary though the columns are drawn independently. Seed 9's moves
  0.23 by the second bin (the first columns' transient) and then stays within 0.01 to the end: it
  is stationary, 0 of 3,000 cases wrong, 4 of 100,000 in the long tests. A snapped network (seed 0,
  step 10) keeps one code per carry from the second bin on (spread 0, no shift).

Findings (E2b' of indrajala-research/docs/research-notes/window-recurrent-report.md, D39: E2b's snapped arm with no run
stopped early, **C2b's registered test**), `scripts/studies/configs/window_recurrent_e2b_prime.json` (E2b's snapped
config but for `end_at_certified_step` off), crate abcba3d7a30e, code at cea696f (`~/code/wt-e2bp`;
the results' "commit" is the launching checkout's, ce4015d), 20 seeds, 2,963 parameters: `jebel`,
6 workers, 2026-10-16 09:12 to 21:38, 131-206 minutes a run (median 196), 1,103 MB peak, every
step's network certified (`e2b-prime.sha256`, d9cd390a...bfe46b7, 1,100 files, run from
`data/window-recurrent`), log `e2b-prime.out` (038e59a0...afe44). The last step's networks
recertified in both verdicts on `pyramidon` (`e2b-prime-certificates/`, `-strict/`,
`e2b-prime-certify.log`, `-strict.log`). Long tests (B1 at 256 and 1,024, 100,000 cases each) on
all 20 (every break is 64), `pyramidon` at c4a91e7 (`~/code/wt-dev2`), 2 workers, 2026-10-17 04:01 to 07:49, about
22 minutes a run, into `e2b-prime-long-tests/`, log `e2b-prime-long-tests.log`
(`e2b-prime-long-tests.sha256`, 487751b4...1511cada, 21 files, run from `data/window-recurrent`).

| arm | working at 64 (W10) | break 64 | certified (last step) | working at 1,024 |
| --- | --- | --- | --- | --- |
| snapped, no early stop (E2b') | 20/20 (0.84-1.00) | 20/20 | 20/20 | 20/20 |
| continuous (E2b, as run) | 20/20 (0.84-1.00) | 2/20 | - | 0/20 (0.00-0.16) |

- **C2b holds** (registered: refuted unless the continuous arm's rate at 1,024 is below the
  snapped arm's by Fisher's exact test, one-sided, p < 0.05): Fisher's exact, one-sided, p 7.3e-12.
- **Training past the first proof cost nothing here:** every run trained all 18 steps (to 18
  columns, the exclusion bound), and every last-step network is certified in both verdicts,
  minimised to the 3-state carry automaton from 3-14 codes reached, snap ratio >= 1.059 and logit
  lead >= 3.9 (no near tie in either verdict).
- **D39's comparison (exploratory): the early stop didn't change the rate.** E2b's snapped arm,
  stopped at its first proof, worked 20 of 20 at 1,024; E2b', trained on, also 20 of 20 certified
  at the last step. The runs are E2b's until then: all 102 step networks the two studies share (to
  each run's first certified step) are byte-identical, and the first certified steps are E2b's (3 in
  6 runs, 4 in 7, 5 in 2, then 6, 7, 8, 10 and 15).
- **One proof lost and regained, none lost for good:** 19 of 20 runs are certified at every step
  from their first; seed 6 at steps 5-7, not at 8-15, and again at 16-18 (its loss replayed in
  P1's snap-wall work: an ordinary step across a code boundary, then Adam's amplification). So
  E8b's collapse past a solution, general at division, is rare at this cell (the stable-replay
  probe: 20 of 20 replays of 4 stable seeds kept the proof).

Findings (E5 of indrajala-research/docs/research-notes/window-recurrent-report.md: unexplored states, C6), from the saved
networks, no training, on pyramidon, 2026-10-17: `explore scripts/studies/configs/window_recurrent_matrix.json
scripts/studies/configs/window_recurrent_e3_e4_trimmed.json --workers 4`, about 2 minutes. Every finished snapped
run's last network, the matrix's and E3/E4's pooled: 195 networks (E3's 15 reruns of the matrix are
byte-identical, C0, and count once), 121 of them passing their last check. Codes over the codes
property's first cases (B1) at the check lengths and at 8, 16 and 64 columns; "new" a code never
used at a check length. Data `data/window-recurrent/e5/` (`e5.json`, every run's codes; `e5.log`),
manifest `e5.sha256` c442c79e...fcd81059cf (2 files, run from data/window-recurrent).

| columns | failing runs with a new code | working runs with a new code | Fisher's exact p |
| --- | --- | --- | --- |
| 8 | 2/21 | 1/100 | 0.078 |
| 16 | 2/21 | 3/100 | 0.21 |
| 64 | 4/21 | 5/100 | 0.048 |
| past training (registered: 8, 16 at `k` 4) | 2/21 | 3/100 | 0.21 |

- **C6 is not supported:** 2 of 21 failing runs reach a code training never produced, against 3 of
  100 working runs (two-sided p 0.21; Holm-adjusted 0.21). 19 of the 21 fail using only codes they
  used at their training lengths: they fail by wrong transitions among explored codes, not by
  reaching unexplored ones. The two exceptions are one cell, the matrix's `k` 1 `s` 4 `m` 16 (seeds
  0 and 4: codes 10, 11, 15 and 10). So Buitrago Ruiz & Gu's intervention (training from other
  sequences' final states) isn't indicated as this method's fix.
- **A new code isn't a failure by itself:** 3 working runs reach one by 16 columns, 5 by 64.
- **At 64 columns** (exploratory, not the registered length): 4 of 21 against 5 of 100, p 0.048
  raw.
- **Holm's correction over C2-C4 and C6** (D4, registered; `inference.holm`): C2 0.0033 to 0.013
  (holds), C3 0.014 to 0.042 (holds), C4 0.044 to 0.088 (**the arms don't differ at 0.05: C4's
  refutation, reported raw at stage 7, doesn't stand**), C6 0.21 (not supported).

Findings (stage 9 and 9b of the factored-state workplan: modular division `p` 5 with a factored
state, C9, C9' and C9'', each an existence claim), crate abcba3d7a30e, the curriculum at `k` 1, `s`
16 a factor, the state read before the first `StateInput` (D10), every step certified, a study
stopping at its first certified run. No network certified in any of the three: each is **"not found
within the budget"**, not refuted (D7).

- **C9** (D12's fallback, `window_recurrent_factored_p5_fallback.json`: two factors of 6 codes, cap
  60), jebel, indrajala-ml 3b6cb12, 6 workers, 2026-10-14 15:46 to 20:27, stopped by its stop file
  for C9' (D13) at steps 19-21 of 60 (910-1,124 epochs a run, 429-445 MB peak). Data
  `data/window-recurrent/factored-p5-fallback/` (`factored-p5-fallback.sha256`, 4c1ec2e7...15387d62,
  360 files, run from `data/window-recurrent`), log `data/window-recurrent-c9.log`
  (c56e0dc8...400ae428; it ends in the report's crash on a step stopped before its first epoch,
  fixed in #758, after every result was written).
- **C9'** (D13, `window_recurrent_factored_p5_m16.json`: two factors of 16 codes, a 4 h budget),
  jebel, indrajala-ml 6abd961, 6 workers, 20:34 to 21:41, stopped by the owner for the diagnosis at
  steps 9-10, then seed 5 alone (21:47 to 22:32, stopped by the owner at step 12, its steps 8-12
  flat). Every run plateaued from step 2: "0" 81-86% right and "undefined" 74-85%, the nonzero
  quotients 31-44% (25% a guess among four), each factor's code at most 0.41 of `A` or `B` but up to
  0.67 of "is it zero" (D14). Data `factored-p5-m16/` (`factored-p5-m16.sha256`,
  695545eb...65dfb97e, 188 files; the probes `signal-split.txt` a5475eee...58e9c0ae and
  `signal-split-seed5.txt` 143ec450...d67fe843; seed 5's first stop in `stopped-2141/`), log
  `data/window-recurrent-c9prime.log` (fb25e506...f4497369).
- **C9''** (D14-D17, `window_recurrent_factored_p5_stagger.json` and `_stagger_pyramidon.json`:
  three factors of 16 codes, learning staggered, the background at 0.1), indrajala-ml 740436d,
  2026-10-14 22:36 to 2026-10-15 02:36, the whole 4 h budget: seeds 0-5 on jebel (6 workers, steps
  10-13) and 6-8 on pyramidon (3 workers, steps 7-8), 362-392 MB peak. **The registered stop rule
  at 2 h (D17) was not applied**: the check waited behind a question to the owner. At 2 h no run
  met it (the largest factor share 0.33 of 0.6, the nonzero quotients at most 0.36 of 0.55), so it
  would have stopped both machines; the last 2 h changed none of that. Data, both machines' runs
  together, `factored-p5-stagger/` (`factored-p5-stagger.sha256`, 2eaa68e0...e42fde95, 344 files,
  each step's `diagnose` beside it), logs `data/window-recurrent-c9pp.log` (jebel,
  b253e781...6f034ea8), `-c9pp-pyramidon.log` (defced1e...e8f51ccc), and the diagnoses'
  `-c9pp-diagnose.log` (0303eeac...04e48420) and `-c9pp-diagnose-pyramidon.log`
  (4692c721...baf91c09).

- **Staggering didn't reach the second factor: in 7 of 9 runs factors 1 and 2 never left one
  code.** Their turns came (at each plateau, from step 2) and their cells trained (seed 6, steps 1 to 2:
  the GRU's weights moved 1.6-6.7 times their norm, as factor 0's), but their codebooks barely
  moved (0.19) and every window snapped to the same code. **The cell's output itself collapsed
  during its turn** (seed 6, a probe on the saved networks): factor 1's `h` before snapping varied
  a little at initialisation (a feature's spread 0.018 across inputs, mean |h| 0.08) and after its
  turn was one point for every input (spread under 0.001, |h| 0.005), its update gate's bias
  pushed from 1.0 to 2.5 (the cell copying its state), its one code on that point; factor 2 the
  same. A state of one code passes nothing on, so the answer's gradient through it averages out
  and the commitment loss, pulling `h` onto its code, is what's left: the reading, not yet tested,
  is that the turn taught the factor to be a constant. Re-seeding the unused codes onto `h` can't
  split it (the probe: 15 codes moved, still 1 used). Factor 0 alone reached 7-12 codes and 0.17-0.33 of `A`,
  0.11-0.26 of `B`; "0" 79-91% right, "undefined" 69-86%, the nonzero quotients 11-44% a class.
- **Two runs split a second factor, both on pyramidon, at step 2 (a step that ran its whole 200
  epochs):** seed 7's factor 1 (3 codes, then 13-16; at the end 0.12 of `A`, 0.24 of `B`) and seed
  8's factor 2 (5, then 11-14; 0.06 and 0.03). Neither carried a residue: the nonzero quotients
  stayed at 0.29-0.38.
- **What it says:** C9' failed by the factors settling on the zero tests together; C9'' by the
  factors after the first collapsing to a constant in their turn. A factor has to vary before its
  turn can teach it, so a next design must start it (its cell re-drawn and its codes re-seeded at
  its turn, its commitment off until it uses several codes; then a turn kept only if the answer
  improved), or train jointly until each factor splits, then stagger, or change the rung (division at `p` 3, where 23% of the answer needs the
  pair, against 50% at `p` 5).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.layers.numpy.recurrent_array_layer import WALL_RATIO
from indrajala_ml.model.persistence.load_network import load_network

from indrajala_studies import addition_data as ad
from indrajala_studies import window_diagnostics as wd
from indrajala_studies import window_drift as wdr
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_replay as wrp
from indrajala_studies import window_tasks as wt


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report", "networks"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
        if name == "run":
            command.add_argument(
                "--workers", type=int, help="the config's worker count, overridden (a study moved between machines)"
            )
            command.add_argument(
                "--arm", choices=wr.ARMS, help="this arm's runs only (a study's arms split between machines)"
            )
    check = commands.add_parser("check")
    check.add_argument("network", type=Path)
    check.add_argument("--n", type=int, nargs="+", required=True, help="the lengths to check")
    check.add_argument("--task", default=wt.ADDITION.name, help="the network's task (window_tasks.task_named)")
    check.add_argument("--properties", nargs="+", help="these properties only (default: all but the held-out one)")
    check.add_argument("--sampled", type=int, default=ad.SAMPLED, help="the sampled properties' sample")
    check.add_argument("--per-stratum", type=int, default=ad.PER_STRATUM, help="the stratified properties' per stratum")
    check.add_argument("--doubling", action="store_true", help="M3 (S2) at 1, 2, 4, ..., n only")
    check.add_argument("--json", type=Path, help="write each length's results and seconds to this file")
    long = commands.add_parser("long-test")
    long.add_argument("networks", type=Path, nargs="+")
    long.add_argument("--task", default=wt.ADDITION.name, help="the networks' task (window_tasks.task_named)")
    long.add_argument("--n", type=int, nargs="+", default=list(wr.LONG_LENGTHS), help="the lengths")
    long.add_argument("--properties", nargs="+", help="these properties only (default: all but the held-out one)")
    long.add_argument("--out", type=Path, required=True, help="write each network's results here")
    long.add_argument("--workers", type=int, default=1)
    certify = commands.add_parser("certify")
    certify.add_argument("networks", type=Path, nargs="+")
    certify.add_argument("--task", default=wt.ADDITION.name, help="the networks' task (window_tasks.task_named)")
    certify.add_argument("--out", type=Path, help="write each certificate here, as <run>-<step>.json")
    certify.add_argument("--workers", type=int, default=1)
    certify.add_argument(
        "--snap-ratio", type=float, default=wc.SNAP_RATIO, help="near ties branch under this ratio (1: strict)"
    )
    compare = commands.add_parser("compare")
    compare.add_argument("original", type=Path, help="a study's directory (results/ and models/)")
    compare.add_argument("rerun", type=Path, help="another study's directory, rerunning some of its runs")
    explore = commands.add_parser("explore")
    explore.add_argument(
        "config", type=Path, nargs="+", help="one or more studies, pooled; a network in two (a rerun) counts once"
    )
    explore.add_argument(
        "--n", type=int, nargs="+", default=list(wr.EXPLORED_LENGTHS), help="the lengths past training"
    )
    explore.add_argument("--json", type=Path, help="write every run's codes to this file")
    explore.add_argument("--workers", type=int, default=1)
    probe = commands.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("--steps", type=int, nargs="+", help="curriculum steps to time (default: 8 and the last)")
    probe.add_argument("--epochs", type=int, default=2, help="training epochs at each step")
    probe.add_argument("--json", type=Path, help="write the probe's figures to this file")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("config", type=Path)
    diagnose.add_argument("--run", nargs="+", help="these runs' keys only (default: every run)")
    diagnose.add_argument("--n", type=int, nargs="+", default=list(wd.LENGTHS), help="the lengths drawn")
    diagnose.add_argument("--cases", type=int, default=wd.CASES, help="cases drawn at each length")
    diagnose.add_argument("--follow", action="store_true", help="diagnose each new step until the lock is free")
    diagnose.add_argument("--interval", type=float, default=30.0, help="seconds between looks when following")
    drift = commands.add_parser("drift")
    drift.add_argument("networks", type=Path, nargs="+")
    drift.add_argument("--task", default=wt.ADDITION.name, help="the networks' task (window_tasks.task_named)")
    drift.add_argument("--n", type=int, required=True, help="the length")
    drift.add_argument("--cases", type=int, default=wdr.CASES, help="the codes property's first cases")
    drift.add_argument("--bins", type=int, default=wdr.BINS, help="equal bins of the columns")
    drift.add_argument("--json", type=Path, help="write each network's profile to this file")
    replay = commands.add_parser("replay")
    replay.add_argument("network", type=Path, help="a saved step's network")
    replay.add_argument("--config", type=Path, required=True, help="the study whose run and settings the replay takes")
    replay.add_argument("--run", required=True, help="the run's key")
    replay.add_argument("--step", type=int, required=True, help="the step whose data is replayed")
    replay.add_argument("--epochs", type=int, default=1)
    replay.add_argument(
        "--generator", required=True, help="each epoch's generator: random.Random(f'{generator}-{epoch}')"
    )
    replay.add_argument("--numpy", action="store_true", help="convert the network to numpy first")
    replay.add_argument("--wall", type=float, default=0.0, help="the snap wall's weight (numpy only; 0: off)")
    replay.add_argument("--wall-ratio", type=float, default=WALL_RATIO, help="the ratio the snap wall keeps")
    replay.add_argument("--json", type=Path, help="write the replay's batches and epochs to this file")
    replay.add_argument(
        "--machines", action="store_true", help="read the machine off the network after every batch, against the last"
    )
    replay.add_argument(
        "--guard", action="store_true", help="undo an update that makes a right machine wrong, and skip its batch"
    )
    verdicts = commands.add_parser("verdicts")
    verdicts.add_argument("config", type=Path)
    verdicts.add_argument("--certificates", type=Path, required=True, help="certify's branched certificates")
    verdicts.add_argument("--strict", type=Path, required=True, help="certify --snap-ratio 1's certificates")
    verdicts.add_argument("--long-tests", type=Path, help="each network's check --json, as <run>-step-<j>.json")
    args = parser.parse_args(argv)

    if args.command == "check":
        long = wr.long_test(
            args.network,
            wt.task_named(args.task),
            args.n,
            args.properties,
            args.sampled,
            args.per_stratum,
            args.doubling,
            log=lambda line: print(line, flush=True),
        )
        if args.json is not None:
            args.json.write_text(json.dumps(long, indent=1))
        return 0
    if args.command == "long-test":
        args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            test = functools.partial(
                wr.long_test_file, out=args.out, task_name=args.task, lengths=args.n, properties=args.properties
            )
            for line in pool.map(test, args.networks):
                print(line, flush=True)
        return 0
    if args.command == "certify":
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for path, result in zip(
                args.networks,
                pool.map(
                    functools.partial(wc.certify_file, task_name=args.task, snap_ratio=args.snap_ratio), args.networks
                ),
                strict=True,
            ):
                name = wr.network_name(path)
                print(f"{name}: {wc.summary(result)}", flush=True)
                if args.out is not None:
                    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
        return 0
    if args.command == "drift":
        profiles: dict[str, Any] = {}
        for path in args.networks:
            name = wr.network_name(path)
            profiles[name] = wdr.profile(
                load_network(str(path)), wt.task_named(args.task), args.n, args.cases, args.bins
            )
            print("\n".join(wdr.lines(name, profiles[name])), flush=True)
        if args.json is not None:
            args.json.write_text(json.dumps(profiles, indent=1))
        return 0
    if args.command == "replay":
        study = wr.load_config(args.config)
        spec: wr.RunSpec = next(spec for spec in study.runs if spec.key == args.run)
        network = wrp.load(args.network, numpy=args.numpy)
        if args.wall > 0.0:
            wrp.set_wall(network, args.wall, args.wall_ratio)
        replayed = wrp.replay(
            network,
            wt.task_named(spec.task),
            list(spec.step_counts(study.settings)[int(args.step) - 1]),
            study.settings,
            args.epochs,
            args.generator,
            log=lambda line: print(line, flush=True),
            machines=args.machines,
            guard=args.guard,
        )
        replayed |= {
            "network": str(args.network),
            "numpy": args.numpy,
            "wall": args.wall,
            "wall_ratio": args.wall_ratio,
            "guard": args.guard,
        }
        if args.json is not None:
            args.json.write_text(json.dumps(replayed, indent=1))
        return 0
    if args.command == "compare":
        compared = wr.compare_runs(args.original, args.rerun)
        for key, found in compared.items():
            print(f"{key}: " + ("reproduced" if not found else f"{len(found)} differ, e.g. {', '.join(found[:5])}"))
        differ = sum(bool(found) for found in compared.values())
        print(f"{len(compared)} compared, {differ} differ")
        return 1 if differ else 0
    if args.command == "explore":
        context = multiprocessing.get_context("forkserver")
        explored: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for config in args.config:
                pooled = wr.load_config(config)
                for run in pool.map(functools.partial(wr.explore_run, pooled, lengths=args.n), pooled.runs):
                    if run is not None and all(run["network"] != other["network"] for other in explored):
                        explored.append(run)
        for run in explored:
            new = ", ".join(f"{n}: {run['new'][str(n)]}" for n in args.n)
            print(f"{run['key']}: works {run['works']}, check passes {run['passes']}, new codes {new}", flush=True)
        print(wr.explore_summary(explored, args.n))
        if args.json is not None:
            args.json.write_text(json.dumps(explored, indent=1))
        return 0
    study = wr.load_config(args.config)
    if args.command == "probe":
        steps = args.steps or sorted({min(8, study.settings.steps), study.settings.steps})
        figures = wr.probe(study.runs[0], study.settings, steps, args.epochs)
        for row in figures["epochs"]:
            print(
                f"step {row['step']} epoch {row['epoch']}: train {row['train_seconds']:.1f} s, "
                f"screen {row['screen_seconds']:.1f} s ({row['screen']:.3f}), peak {row['peak_mb']:.0f} MB",
                flush=True,
            )
        check = figures["check"]
        print(f"whole check at {check['n']}: {check['seconds']:.1f} s, peak {check['peak_mb']:.0f} MB")
        if "certificate" in figures:
            certificate = figures["certificate"]
            print(
                f"certificate: {certificate['seconds']:.1f} s, {certificate['states']} states, "
                f"peak {certificate['peak_mb']:.0f} MB"
            )
        if args.json is not None:
            args.json.write_text(json.dumps(figures, indent=1))
        return 0
    if args.command == "diagnose":
        runs = [spec for spec in study.runs if args.run is None or spec.key in args.run]
        while True:
            ending = args.follow and machine_lock.holder() is None  # looked at before the pass: none missed
            for spec in runs:
                if (study.models / spec.key).is_dir():
                    task = wt.task_named(spec.task)
                    for text in wd.diagnose_new(study.models / spec.key, task, load_network, args.n, args.cases):
                        print(text, flush=True)
            if not args.follow or ending:
                return 0
            time.sleep(args.interval)
    if args.command == "networks":
        for path in wr.last_networks(study).values():
            print(path)
        return 0
    if args.command == "verdicts":
        print(wr.verdicts(study, args.certificates, args.strict, args.long_tests))
        return 0
    if args.command == "run":
        if args.workers is not None:
            study = dataclasses.replace(study, workers=args.workers)
        if args.arm is not None:
            study = dataclasses.replace(study, runs=[spec for spec in study.runs if spec.arm == args.arm])
        with machine_lock.study():
            wr.run(study, log=lambda line: print(line, flush=True))
        print(wr.report(study))
    elif args.command == "status":
        print(wr.status(study))
    else:
        text = wr.report(study)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
