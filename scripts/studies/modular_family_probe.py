"""
The modular family's probe (the modular-family workplan, stage 2): one network answering the
running quotient, sum and difference of two numbers mod p at every window (query tokens), against
the quotient alone; a factored state of C9''s cell, trained by the curriculum within a wall-clock
budget, diagnosed as it trains (indrajala_studies/modular_family.py).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/modular_family_probe.py run \\
        --arm family --seed 0 --budget-minutes 60 --out data/window-recurrent/modular-family-probe/family-seed0
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/modular_family_probe.py run \\
        --arm family --reads output --first-seed 2 --seeds 6 --budget-minutes 60 --out DIR
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/modular_family_probe.py run \\
        --arm family --reads output --factor 32 128 --first-seed 0 --seeds 4 --budget-minutes 60 --out DIR
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/studies/modular_family_probe.py run \\
        --arm family --reads output --first-seed 100 --seeds 14 --workers 3 --budget-epochs 400 \\
        --budget-minutes 180 --out DIR
    .venv/bin/python scripts/studies/modular_family_probe.py long-test NETWORK.json [...] --task modular-division-5 --out DIR
    .venv/bin/python scripts/studies/modular_family_probe.py trace DIR/seed<s>/log.jsonl [...]
    .venv/bin/python scripts/studies/modular_family_probe.py readout DIR --pairs family:quotient sum:quotient sum:product

`--fork` turns on snapshot and fork on collapse (the collapse-fork workplan, retired:
`git show 89cf3a3:docs/collapse-fork-workplan.md`): a step's epoch
whose screen falls below 0.6 of the step's best goes back to that best network, halves the step's
rate and takes new data, at most 4 times a step; each fork a line in the log, each step's end its
count.
`--guard on` turns on the certificate guard (the guard workplan, retired:
`git show ed0e378:docs/guard-training-workplan.md`;
indrajala_studies/certificate_guard.py): for an epoch whose screen before it got every quotient
right, the quotient's machine is read after every batch and an update that makes a right machine
wrong is undone; each epoch's row logs `armed` and `rollbacks`, each step's end its `rollbacks`.
Several `--p` run in one pool, each seed's moduli in turn, into DIR/p<p>/seed<s>; `--codes-ratio
R` gives each of `--factors` factors `ceil(R p)` codes (the division ladder, R 32/11:
`git show f9101d8:docs/division-ladder-workplan.md`).
`--task NAME ...` runs a general algebra's machines in place of division (the general-algebras
workplan, D6: `word-problem-s<n>`, `move-to-front-<n>`, `random-<n>-<seed>`; window_tasks), each
its own task, one token a window, several in one pool into DIR/<task>/seed<s>; `--codes-ratio R`
gives each factor `ceil(R sqrt(q))` codes for `q` states (the workplan's D3, R 32/11). `certify`,
`long-test` and `guard-epoch` take any such `--task` too (slot 0, the machine's output).
`--guard both` runs each seed without and with it, in one pool, into DIR/unguarded/seed<s> and
DIR/guarded/seed<s> (the readout's arms). `run` holds the machine lock (machine_lock.study), as
the harness's studies do, so an A/B isn't started beside it. `run ... --resume`, the same command
again, takes each run on from its checkpoint (DIR/seed<s>/checkpoint.json, left at each step's start
and each epoch's end): a stopped run goes on as if it hadn't stopped (every network and log row but
the seconds), an ended one is skipped, one not begun starts (the code-consolidation workplan's stage
3, so `stone`'s runs pause at 07:00 and go on at 18:00).
The code-consolidation workplan's arms (stage 3): every run's curriculum starts at `s0`, the
shortest length a stateless network can't pass (D3; 1 for every round-1 machine but random-3-0's 2),
a step named by its longest length; `--long-tail F` (B, D4) trains every other example at lengths
j+1..F j, capped at the exclusion bound; `--consolidate` (A, D1) groups the network's own codes at
each diagnosis by `d`-equivalence (D2's `d`) and pulls each duplicate to its representative, each
diagnosis row logging `consolidation` (codes reached, groups, duplicates pulled); `--restore-best`
(stage 2c's P2) ends a step stopped by a plateau on its best screen's network, saved at each new
best, before the step's certificate and the next step, a row logging it; `--commitment-from J`
(the commitment-control workplan's diagnostic) trains the steps before J with every factor's
commitment 0; `--window-width D` widens the window's residual stream from W11's 16, its FFN twice
it (the window branches workplan's W1); `--branches` gives the window a shared trunk, then a branch
a factor and one for the answers, each factor's pooling reading its own (the plan's B1-B3; with
`--reads output`), and `readout-tune` then trains the answers' whole branch; `--answer-branch`
gives it a shared trunk, then the state branch (W11's last block, unchanged) that every factor reads
and the answers' own single-head branch (the one-code-one-state workplan's R1); `--branch-heads H`
gives `--branches`' blocks H heads (the post-paper workplan's branches-32, D8); `--recombine` gives
the window a two-block shared trunk, then a two-head branch a factor, each factor's pooling reading its own,
the branches' ends summed into one more block before the output (the post-paper workplan's
recombining window, D7; with `--reads output`); `--thermostat` turns
each factor's commitment off while it uses fewer than 3 effective codes (2 ** the entropy of its
codes' use on the step's screen) and ramps it back over 20 epochs once it uses half its codebook
(the commitment-control workplan's stage 2; levels set by `--thermostat-floor`, `-restore`,
`-ramp`; each epoch's row logs its effective codes and commitments); `--anneal` trains each snapped
factor's codes soft (the softmax mix at a temperature, from `--anneal-start` 1.0 halved every
`--anneal-halving` 10 epochs), its commitment 0, until it hardens (99% of the step's screen's windows
weighting one code above 0.99), then hard with its commitment ramped back over `--anneal-ramp` 20
epochs (the commitment-control workplan's stage 3; each epoch's row logs them); with `--warm-up`, every
snapped factor runs continuous first (its commitment 0) until a step's best screen reaches 0.95 with its
distinct states at least 4 times its codes, or the 8th step; its codes then set by k-means of the states
seen, and its temperature profiled from T_hot (`--melt-hot` 2 times the whole's critical temperature):
heated doubling over `--melt-heat` 6 epochs, held `--melt-hold` 10, cooled halving every `--melt-halving`
15 with no floor, its soft codes moved by thermal noise of variance `--melt-kappa` 0.01 T, soft for
`--anneal-tail` epochs past hard (the state formation workplan's option 1, D14-D19; never with the fork,
`--restore-best` or `--keep-machine`); `--keep-machine` keeps each
step's network whose machine has the fewest pairs per code (read at each diagnosis, ties to more
right pairs) and restores it at a plateau, and `--counterexamples [SHARE]` draws SHARE (0.25) of each
epoch's examples from the shortest words to the codes wrong for their states (at most 256, read at
each diagnosis), each followed by `d` uniform tokens (the one-code-one-state workplan's R3 and R4;
the diagnosis rows log `machine`, `kept` and `counterexamples`). `run` prints each
target's feasibility line first (q, codes, m, the cap, s0, d) and refuses a cell whose joint codes
can't hold its states.
`guard-epoch NETWORK --p P --step J` trains one epoch of step J from a saved network, unguarded
and with the guard armed, on the same data, and prints each one's seconds, whether its quotient's
machine is right before and after, and the guard's rollbacks (the guard workplan's budget probe).
`separability NETWORK --task T --out FILE` reads the features a saved network's output layer reads
for every reachable code and window, and fits a linear readout to all of them (a softmax fit, then a
perceptron): whether its answers are linearly separable, and for each pair left wrong how near the
nearest feature with another answer is (the window branches workplan's check).
`readout-tune NETWORK --task T --step J --out DIR` trains a saved network's output layer alone
(every other layer and the recurrence frozen, so its state machine is the saved one) on step J's
lengths, an epoch at a time, until its certificate passes or `--epochs` are spent, a row an epoch
(DIR/log.jsonl; DIR/certified.network.json if it certifies).
`--first-seed F --seeds N` runs seeds F..F+N-1, a process each, `--workers` at once (default
all), into DIR/seed<s>/ (and its output and errors in DIR/seed<s>.log); `--runs TASK:SEED ...` runs
those alone, in one pool (chosen seeds replayed: DIR/<task>/seed<s> for several tasks); `--budget-epochs` (required)
stops a run when that many epochs are trained in all, the binding budget (epochs, not wall-clock, are
the run's time axis: the owner, 2026-10-23); `--budget-minutes` is only a runaway guard (none by
default), its stop "deadline" read as censored.
`--trigger-signal`, `--trigger-rule`, `--trigger-threshold` and `--trigger-floor` set the fork's
trigger (indrajala_studies/collapse.py; the default is the fork's first: pooled, below 0.6 of
the peak, no floor). `drops LOG... [--trigger-...]` scores a trigger against runs' logs, offline
(the p11-trigger workplan, D6; retired: `git show 28894cf:docs/p11-trigger-workplan.md`): each run's large drops (the worst answer kind from 0.9 down
by more than 0.3), how many the trigger catches, of those never recovered how many, and its false
alarms; at each epoch where the log has every answer kind's share (from the p11-trigger build on),
else at each diagnosis (a proxy).
`by-length NETWORK... --task T` and `breaks NETWORK... --task T` read saved networks' machines against
the task's (indrajala_studies/machine_reading.py; the code-consolidation workplan's stage 2):
exact accuracy over every word of each length, and the share ending on a code right for its state;
and the transitions that break, from each state's first code and from its extra codes, over every
code and from right ones (a code right when `--d` tokens of outputs agree, default the task's
distinguishing length plus one), by depth. A line a network, its JSON into `--out`.
`approximations NETWORK... --task T` reads what each factor's code determines of the task's machine
exactly (machine_reading.approximations): the finest approximation (a congruence's quotient) it
determines, the named congruences that refines (an ordering machine's, or division's lines `A + kB`:
window_tasks.named_labels), whether the factor's code is a machine of its own, and whether the
factors together cover the target. The readings take `--arm` and `--vocabulary` as the run had them.
`factor-shares NETWORK... --task T` reads each factor's share of each of a machine's named congruences
and of its state, I(code; x) / H(x) on the windows training draws (window_diagnostics.congruence_shares):
weighted as the network meets them, so valid on inexact networks (a failure's peak, a collapse), where
`approximations`, built for certified networks, merges states on any shared code. Its line lists each
factor's congruences at `--show` or more; its JSON (`--out`) every share.
`answers-proof LOG... --task T` is the paper's T6 (paper-workplan.md): at each `--slack` bound, the
answers-only stop from a run's log, then Chow's test set on the stop's saved step network under the
same bound ("out of reach" past `--limit` words), a line a bound and its JSON into `--out`.
`trace` prints runs' logs a line per diagnosis (each factor's codes in use, the state's share, each
answer's accuracy), marking a collapse: the codes in use below half the diagnosis before.
`long-test` runs the long tests (the division workplan, D4: U1 and S1 at 128, 256 and 1,024
columns, the harness's sizes) on saved family networks, one slot (the task's), into
DIR/<network>.long-test.json (skipped when there), a line each of its cases and failures.

Each run writes `log.jsonl` (a line per diagnosis: each operation's accuracy, the quotient by
class, each factor's codes used and its share of A, B, the sum S, the difference D and the zero
tests; at each step's end, the quotient's accuracy at 24 and 48 columns, its certificate, and
`machine`: the codes, the pairs (code, state) and the right pairs its machine reaches beside the
target's, so pairs per code, 1.0 in every certified network (the one-code-one-state workplan's R2;
`trace` prints it)) and `network.json` (the network at its last step's end).

Findings (stage 2 of the modular-family workplan), on jebel (i7-9700K), one process a run,
OPENBLAS_NUM_THREADS=1, 2026-10-15; p 5, two factors of 16 codes, s 16 each, k 1 column (D3), the
probe's settings (D4), a 60-minute budget a run. Data in data/window-recurrent/modular-family-probe/
(a directory a run: log.jsonl, network.json; manifest `modular-family-probe.sha256`,
4d465007...a9847c25, 41 files, run from data/window-recurrent), the D10 check's in
data/window-recurrent/probe-d10/ (`probe-d10.sha256`, 4236a059...557ae9c3, 81 files), the certified networks and their certificates in
modular-family-probe/certify/ (in the first manifest); both machines.

- **Registered arms (the poolings before the state, D10, as C9's cell), indrajala-ml 9b4be51,
  06:08-07:09: none works.** family seeds 0 and 1 and quotient seeds 0 and 1 all end their hour
  with the nonzero quotients at 0.33-0.37 (a guess 0.25), the state's share of the pair 0.22-0.27,
  24-30% right at 24 columns; every factor's share of A, B, S or D at most 0.18, the shares frozen
  to three decimals from step to step. The family made no difference.
- **The cause is D10 (the D10 check: the harness, E8c's own cell and task, `modular-sum-5`, seeds 0
  and 1, 30 minutes each, the same commit):** one factor whose pooling reads before the state passes
  steps 1-2, then plateaus at every step from 3 (31-91 epochs each, break 0): 0 of 2 certified; the
  same factor reading the window's output certifies as E8c did, step for step (seed 0 at step 4,
  seed 1 at step 5): 2 of 2. Reading before the state, a factor's cell gets only the embedded column
  and must compute the whole transition itself; reading the output, the window's three blocks do
  most of it (a reading, not tested apart). Every factored division run (C9, C9', C9'') read before
  the state.
- **Reading the output (indrajala-ml 600b19b, 06:53-07:54), the family certified 2 of 2; the
  quotient alone 0 of 2.** family seed 1 was right on every answer at 3-12 columns and at 24 and 48
  from step 7, seed 0 from step 6. Their networks saved at steps 11 (seed 1, sha256 718b382d...4bd595be)
  and 12 (seed 0, 77b558a0...838bc3ef) **certify against division's machine, both verdicts** (certify,
  2da6d0b, pyramidon): seed 1 branched 33 states reached, minimised to 25, 0 wrong, 2 near ties,
  logit lead >= 2.03; strict 32 reached, minimised to 25, snap ratio >= 1.018, lead >= 3.81; seed 0
  branched 53 reached, minimised to 25, 6 near ties, lead >= 2.21; strict 52, minimised to 25, snap
  ratio >= 1.025, lead >= 2.21. Their sums certify against the modular sum too (minimised to 5). The
  first certified networks for division. seed 1's step-6 network, 98.7% right at 48 columns on
  samples, fails the certificate (1,025 wrong outputs): sampled tests overstate "works". The
  quotient alone reached nonzero quotients of at most 0.64 and at most 36% at 24 columns; seed 1
  collapsed at step 9 (codes 2 and 5).
- **The state isn't factored by residue.** At seed 1's step 6 each factor uses 14 codes and holds
  0.37-0.71 of every direction xA + yB (no factor a function of one), the pair of codes 63 states
  for the 25; the certificate's readout agrees (neither factor's code a function of the target's
  state). So the registered success's factor clause (a factor at 0.9 of A, B, S or D) isn't met;
  its quotient clause is: an outcome the plan didn't name, division solved with its state spread
  over both codebooks.
- **Training past a solution destroyed it, in every run that got far:** family seed 1 collapsed at
  step 12 (codes 4 and 4), seed 0 at step 14, both quotient runs by the hour's end (all four ended
  at nonzero quotients 0.13-0.31, 20-30% at 24 columns). The certified networks had been saved first.
  The probe now certifies each step's quotient and stops at the first certified (38f787c), as the
  harness's `stop_at_certified` does (E8b).
- Two counts at 2 seeds an arm are a screen: the family's part (2 of 2 against 0 of 2) is
  suggestive, not shown. Before launch the probe's examples lacked the harness's final zero window
  and the state never split in either arm (the step at one column passed on the column alone); found
  in local checks, fixed in 9b4be51 before any registered run.

Findings (stages 2 and 3 of the division workplan), 2026-10-15; p 5, reading the output, k 1
column, the probe's settings, 60 minutes a run, stop at certified; the runs on jebel (i7-9700K), a
process each, OPENBLAS_NUM_THREADS=1. Data in data/window-recurrent/division-probe-<probe>/ (a
directory a seed: log.jsonl, network.json, certified-step-<s>.network.json; seed<s>.log,
probe.out, launch.txt; certificates in certify/, long tests in long-tests/), one manifest each,
run from data/window-recurrent, both machines: `division-probe-2a.sha256` 159740f5...529d032d
(29 files), `division-probe-2b.sha256` 785ecb08...4586bbd2 (14), `division-probe-3a.sha256`
63fcbd13...6f70fff1 (35), `division-probe-3b.sha256` 89a1e28e...32efd627 (14),
`division-probe-3c.sha256` 1e2dc84f...e42c0be5 (19). Certificates (both verdicts: snap ratio 1.05
and 1.0) on pyramidon.

- **Probe 2a (family, two factors of 16, seeds 2-7, c5378bc, 08:29-09:29): 3 of 6 certified;
  with the first probe's two, 5 of 8** (Wilson 0.31-0.86, a screen). seed 5 at step 7 (6.3
  minutes, sha256 141c25db...40610515), seed 7 at step 8 (19.7, 6e31fcec...e308dc9a), seed 4 at step
  10 (24.1, 4529c5dc...b7ab6d66), each against division's 25-state machine in both verdicts, 0 wrong
  outputs, 0 wrong states: branched 65, 57 and 88 codes reached, minimised to 25, 19, 13 and 35 near
  ties, logit lead >= 1.41, 0.598 and 1.07 (seed 4's snap ratio >= 1.001); strict 63, 57 and 88
  reached. Seeds 2, 3 and 6 ran out the hour (steps 17, 17, 21), each **collapsed**: the state
  minimised to 7, 6 and 11, 23-25% at 24 columns. seed 3 was right on every sampled answer at steps
  6-9 (100% at 24 columns, 99.99% at 48, minimised to 30 at step 9) without certifying, and
  collapsed at step 10 (minimised to 2 by step 12): sampled tests overstate "works" again, and
  training past a near-solution destroys it. Every success came within 25 minutes and every failure
  is a collapse, so a longer budget wouldn't raise the rate; the collapse would (the
  commitment-control workplan's thermostat).
- **Probe 2b (the long tests, D4: U1 100,000 and S1 25,000 cases at each of 128, 256 and 1,024
  columns): all five certified networks pass every one of their 375,000 cases**, read by
  `long_test_failures` (the first probe's two on pyramidon, 4,276 and 4,284 s; 2a's three on jebel,
  not pyramidon as D8 says, since jebel was free, 4,412-4,423 s, three at once). The certificate
  and the long tests agree.
- **Probe 3b (the quotient alone, two factors of 16, seeds 2-5, c5378bc, 09:30-10:30): 0 of 4; with
  the first probe's two, 0 of 6.** Each ran out the hour (steps 20-34) at nonzero quotients
  0.19-0.28, 21-27% at 24 columns; their best nonzero quotients 0.61-0.77, at 24 columns at most
  46%; seed 3 collapsed to one code a factor. The family against the quotient alone: 5 of 8 against
  0 of 6, Fisher's exact (two-sided) p 0.031, a screen across two probes (one per arm, not
  interleaved seeds), so the family looks like a real ingredient, the question for stage 5.
- **Probe 3a (family, one factor s 32 m 128, D5, seeds 0-3, 92b98da from a worktree, 10:30-10:48):
  4 of 4 certified**, so the second codebook isn't needed at p 5 (D9's question). seed 0 at step 5
  (8.2 minutes, 92677677...4385246b), seed 1 at step 6 (10.7, d50aad5a...1fb1452b), seed 3 at step 6
  (14.5, 25ca3d32...443529da), seed 2 at step 8 (18.0, ed3f9f3a...233867b8); both verdicts, 0 wrong
  outputs, 0 wrong states, 61-77 codes reached, minimised to 25, logit leads >= 2.27, 0.62, 2.41 and
  0.103 (seed 3's thin, but both verdicts hold). Long tests: every one of each network's 375,000
  cases passes (jebel, 92b98da, four at once, 4,336-4,349 s). Against two factors of 16 (5 of 8):
  p 0.49, no difference shown; every 3a run finished within 18 minutes. The working
  configuration stays factored (D9 (a)): the product state's sizes multiply in one codebook as `p`
  grows, so stage 4 keeps D7's two factors.
- **Probe 3c (the quotient and the sum, `--arm sum`, 3a's factor, seeds 0-1, ae2a09f from a
  worktree, 11:05-11:18): 2 of 2 certified, so one dense counter is enough.** seed 1 at step 10
  (9.4 minutes, 913daa16...ec78b606), seed 0 at step 9 (12.7, 4e9e5dd4...56c3a367); both verdicts, 0
  wrong outputs, 0 wrong states, 41 and 51 codes reached, minimised to 25, snap ratio >= 1.006,
  logit lead >= 4.7 and 4.14 (the widest margins yet). seed 1 minimised to 25 at steps 8 and 9 and
  certified only at 10. Long tests: every one of each network's 375,000 cases passes (jebel,
  ae2a09f, 3,534 and 3,539 s).
- **Eleven certified networks for division at p 5, all certified in both verdicts and passing
  every long-test case** (archived in indrajala-models, window-recurrent/2026-10-15-division).
- The feasibility table's cap of 152 for the one-factor cells isn't enforced: `run` passes no cap,
  so the curriculum's default 280 applied; no run passed step 34. (Since 8364a83 the cap defaults
  to the cell's exclusion bound, `modular_family.exclusion_bound`.)

Findings (stage 4 of the division workplan, probe 4), 2026-10-15; p 7, the family, two factors
of 16 (256 joint states for 49; the exclusion bound 304, the cap), reading the output, k 1 column,
the probe's settings, 60 minutes a run, stop at certified, seeds 0-3; jebel (i7-9700K), the main
checkout at 8364a83, a process each, OPENBLAS_NUM_THREADS=1, 12:41-13:42. Data in
data/window-recurrent/division-probe-4/ (as stages 2-3's; the certificates on pyramidon, the long
test on jebel), manifest `division-probe-4.sha256` 5de66ce4...2e131958 (20 files, run from
data/window-recurrent), both machines.

- **1 of 4 certified: division at p 7, 49 states, for the first time.** seed 2 at step 7 (11.2
  minutes, 139 epochs, sha256 972d4699...fd929c4d), against division's 49-state machine in both
  verdicts, 0 wrong outputs, 0 wrong states: branched 88 joint codes reached, minimised to 49, 34
  near ties, logit lead >= 2.05; strict 87 reached, snap ratio >= 1.004, lead >= 2.33. It was right
  on every sampled answer at 24 columns from step 5 (minimised to 72, then 50 at step 6) and
  certified at step 7. Every p 7 attempt before (E7, E7': the quotient alone, one codebook) failed:
  0 of 80.
- **Its long test: every one of its 447,000 cases passes** (U1 and S1 at 128, 256 and 1,024
  columns, `--task modular-division-7`; jebel, 8364a83, 5,054 s; `long_test_failures`: none).
- **Seeds 0, 1 and 3 ran out the hour** (steps 19, 19, 18; 428-441 epochs), each **collapsed**:
  peaks of 31%, 33% and 30% at 24 columns (steps 10, 7, 13; minimised to 81-117), ending minimised to
  17, 5 and 20 at 19-21% at 24 columns. None was rising at the hour's end, so the registered
  conditional rule calls no 120-minute rerun; the next probe for p 7 would be D7 (b), two factors
  of 32, not yet decided against stage 5's needs.

Findings (the collapse, traced with `trace`; 2026-10-15, from the logs of every run above, no new
runs):

- **A collapse is abrupt.** Within one diagnosis interval (5 epochs) the codes in use fall to half
  or less and every answer to about chance: 15 such events in 11 runs (the codes in use halving from
  at least 8), 11 mid-step and 4 at a step's first diagnosis. p 7 seed 1, step 10, epochs 15 to 20:
  codes [15, 14] to [1, 1], the difference 0.91 to 0.20; 2a seed 3: right on every answer at step
  9, at chance at step 10's first diagnosis (codes 29 to 20 there, 3 by epoch 15). Runs partly
  recover and collapse again (p 7 seed 0, steps 11-13), and a collapse isn't always the end: 3c
  seed 0 collapsed at step 5 (codes 60 to 24) and certified at step 9.
- **No slow drift precedes it:** the codes in use hold at 15-16 a factor up to the diagnosis before.
  That fits an optimiser instability (a loss spike; Thilak et al. 2022's late instabilities of
  Adam) better than the commitment loss's gradual pull, the deferred commitment-control plan's
  premise. Training has no gradient clipping and a constant rate (Adam, 0.003), and the
  back-propagated sequences lengthen with every curriculum step.
- **At p 7 the codebooks are saturated:** both factors at 16 of 16 codes from step 2 to the
  collapse, one counter learned (the difference or the sum at 0.95-1.00), the other near 0.5. That
  argues for 32 codes a factor (division workplan, D7 (b)) apart from any fix.
- Not measured: the loss and the gradient's norm per batch around a collapse (the diagnosis is
  every 5 epochs); the next probe should log them.

Findings (stage 5 of the division workplan, the registered claim, D11-D14), 2026-10-15; p 5,
two factors of 16, reading the output, the probe's settings, 400 epochs trained in all (180
minutes only a guard; no run reached it), stop at certified, seeds 100-119 in each arm; ce4015d,
OPENBLAS_NUM_THREADS=1; jebel seeds 100-113 (each arm 3 at once), pyramidon 114-119 (each arm 2 at
once), 14:00-18:10. Every certified network recertified on pyramidon (f22a9f0, the same code) in
both verdicts. Data in data/window-recurrent/division-claim/, manifest `division-claim.sha256`
3b17c4ca...518ed4b49 (168 files, run from data/window-recurrent; the long tests apart), both
machines. The readout is `readout data/window-recurrent/division-claim`.

- **Works: the family helps.** The family 13 of 20 certified (Wilson 95%: 0.43-0.82), the quotient
  alone 1 of 20 (0.01-0.24); Fisher's exact test, one-sided, p 6.9e-5 (registered: works at p <
  0.05). The screens' difference (5 of 8 against 0 of 6) holds at 20 fresh seeds.
- **The family's successes:** steps 6-12 (median 8), 99-257 epochs (median 177), 10-30 minutes on
  jebel and 18-55 on pyramidon (the minutes are the machine's and its load's; the epochs are the
  measure, D13). The quotient alone's one: seed 100, step 10, 202 epochs, 15.0 minutes.
- **Every recertification agrees:** 14 of 14 certified in both verdicts against division's
  25-state machine, 0 wrong outputs, 0 wrong states, each minimised to 25; 35-74 joint codes
  reached; snap ratio >= 1.0002 (220 near ties in the branched verdict, none strict), logit lead
  >= 0.10.
- **Every failure spent its 400 epochs:** the family's 7 at steps 14-20, the quotient's 19 at
  steps 13-25 (no run ended on the guard).
- **The family's failures mostly lost what they had; the quotient's mostly never had it**
  (`readout`'s course of each run). The worst answer at its best: 4 of the family's 7 failures
  reached >= 0.95 (seeds 101, 111, 113, 119 at 0.96-1.00, steps 6-8; 115 and 116 at 0.87 and 0.90),
  against 2 of the quotient's 19 (their median 0.79). That is the sampled answers, not the states:
  runs were right on every sampled answer, 48 columns included, with 160-177 states minimised (seed
  104, steps 6-8) and certified a step or more later, the minimised count falling to 25 in one
  step; seed 101, right on every sampled answer at step 6 with 67, never certified.
- **Large drops are common and mostly recovered** (from the diagnoses, every 5 epochs: the worst
  answer from >= 0.9 down by more than 0.3 within one interval): 16 in 10 of the family's runs (6
  of the 13 that certified, 4 of the 7 failures) and 8 in 4 of the quotient's; 7 at a step's first
  diagnosis, 17 mid-step. 17 of the 24 came back to >= 0.9 within 1-6 diagnoses. Whether the codes
  collapsed didn't decide it: seed 100 fell to [7, 6] codes and certified, the quotient's 105 and
  115 kept all 16 a factor and never came back. So a failure is a drop not recovered from, and the
  family's successes survived theirs.
- **The curriculum moves on while broken:** a step ends on a plateau after 10 epochs (`patience`)
  without beating its best screen score, and a drop leaves that best at the peak before it, so a
  run has 10 epochs to beat its own peak or takes on the next length broken. 13 of the family's 16
  drops ended their step on a plateau; of all the steps ending on one, 119 of the family's 133 and
  261 of the quotient's 262 ended with the worst answer below 0.9.
- **seed 107 never came near:** the sum 0.98-1.00 at 9 of steps 8-17 (0.58 at 12), the difference
  near chance (0.45-0.56) over steps 7-17 with 9-16 codes a factor in use: one counter learned, the
  other stuck, as at p 7 (stage 4).
- **For the fork (the collapse-fork workplan):** its snapshot is the step's start (the step
  before's network) and its best since, so it reaches drops at a boundary and mid-step alike; but it
  fires only below 0.6 of the step's peak screen score (the share of every answer right, not the
  worst answer). Of the 24 drops, from the logged screen scores (a diagnosis's; the fork sees every
  epoch, so these are the fewest it would fire on): 6 below 0.6 of the step's logged peak (100,
  111, 119, the quotient's 113 and 118 twice); 4 at a step's start below 0.6 of any start above
  0.97 (102, 106, 113, the quotient's 118); 11 not, 8 above 0.6 of the logged peak (110 twice, 112,
  113, 117, the quotient's 105 twice and 115) and 3 at a step's start above 0.6 of any start up to
  1.0 (101 at 0.76, 119 at 0.90, the quotient's 118 at 0.61); 3 unknown (no screen logged). So
  seed 101's loss, the clearest, is one it would miss. A restore doesn't reset the plateau count. Read the fork probe with this in
  mind; a drop threshold on the worst answer, or a step that ends only when its peak is recovered,
  are the alternatives the logs suggest.
- **Measured from here on** (after stage 5, the training unchanged: a 3-step run's network and
  diagnoses identical before and after): every epoch's screen score and the step's start (the step
  before's network on the new lengths), each step's network (`step-<j>.network.json`), and in
  `readout` each run's large drops (`large_drops`) with how many recovered. Stage 5's logs have
  the screen score at diagnoses only, and only the last network.
- **The long tests (D4) pass: every one of the 14 networks passes every case**, 375,000 each (U1
  100,000 and S1 25,000 at each of 128, 256 and 1,024 columns), 5,250,000 in all, no failure listed
  (`long_test_failures`: none). 11 on pyramidon (f22a9f0, one at a time, 4,232-6,143 s each), 3 on
  jebel (ce4015d, the same code, 2,321-3,430 s), 2026-10-15 19:00 to 2026-10-16 09:12, each worker
  claiming its next network by a mkdir on pyramidon (`~/code/stage5-long-tests.sh`; two of jebel's
  workers claimed the quotient's seed 100 in the same second, so it was tested twice, both passing).
  Results in `division-claim/long-tests/<arm>-<seed>/`, the per-network summary
  `division-claim/long-tests-summary.txt`, logs `long-tests-pyramidon.txt` and
  `long-tests-jebel.txt`; manifest `division-claim-long-tests.sha256` ae437c1d...c499329 (14 files,
  run from data/window-recurrent).

Findings (the fork probe, stage 3 of the collapse-fork workplan, D6, registered before its data;
the workplan retired: `git show 89cf3a3:docs/collapse-fork-workplan.md`), 2026-10-16; p 7, the family, two factors of 32 codes (`s` 16), reading the output, seeds
0-3, 400 epochs a run (`--budget-epochs`), stop at certified; two arms on the same seeds, with
`--fork` (drop 0.4, cool 0.5, at most 4 forks a step) and without; jebel, ~/code/wt-fork at a55ef4c,
8 runs at once, 06:35-07:38. Every certified network recertified on pyramidon in both verdicts
(`certify/`, `certify.log`). Data in data/window-recurrent/fork-probe/, manifest
`fork-probe.sha256` 7cdcacc0...75324e162 (47 files, run from data/window-recurrent). The readout
is `readout data/window-recurrent/fork-probe --arms fork nofork`.

| seed | fork | without |
| --- | --- | --- |
| 0 | certified, step 9, 149 epochs, 1 fork | certified, step 12, 208 epochs |
| 1 | spent at step 15, 18 forks, the worst answer at best 0.85, 358 states minimised | spent at step 14, collapsed, at best 0.70 |
| 2 | certified, step 11, 382 epochs, 6 forks | certified, step 12, 260 epochs |
| 3 | certified, step 15, 328 epochs, 23 forks | spent at step 17: a drop at step 12 (1.00 to 0.19) never recovered |

- **Works, as registered:** the fork arm certifies 3 of 4 (Wilson 95%: 0.30-0.95) against 2 of 4
  (0.15-0.85), and no fork run ends collapsed (seed 1's last step minimised to 358 states, not
  below half of 49). A screen (D6): one-sided Fisher p 0.5; the runs are paired, not a rate.
- **Five certified 49-state networks** (stage 4 had one in 4 runs at 16 codes a factor): each
  robustly certified against division's machine mod 7, 0 wrong outputs and states, minimised to
  49 from 82-242 codes reached (79-225 strict); snap ratio >= 1.000, logit lead >= 0.028 (fork
  seed 0's branched verdict; 2.63 strict). **Every one passes all 447,000 long-test cases**
  (U1 and S1 at 128, 256 and 1,024; `pyramidon`, the main checkout at 07d4cbd, one at a time
  2-2.5 h each, `~/code/fork-long-tests.sh`): `fork-probe/long-tests/`, summary
  `fork-probe/long-tests-summary.txt`, log `fork-probe/long-tests.log`; manifest
  `fork-probe-long-tests.sha256` 4fc08d26...7e84ab (5 files, run from data/window-recurrent).
  Archived in indrajala-models (window-recurrent/2026-10-15-division, `family-f2x32-p7-*`).
- **The arms are the same run until the first fork** (the same seed, the fork off is unchanged):
  seed 2's two arms are identical to step 6, epoch 53. So each seed is a paired comparison. Seed 3
  is the fork's case: the run without lost everything at step 12 and spent its budget; with the
  fork, 4 forks at step 12 (worst answer 0.81 after the first three, 1.00 after the fourth) and
  certified at step 15. Seed 0 forked once, at step 9's first epoch, and certified there, 3 steps
  and 59 epochs before the run without (which fell 1.00 to 0.25 at step 9 and recovered). Seed 2
  certified in both, the fork a step sooner but 122 epochs later.
- **Many forks are at a step's start:** 28 of the 48 at a step's epochs 1-3, the new length
  dropping the screen below 0.6 of the step's peak (its start included); the fork restores the
  start at half the rate. Seed 1 forked 3 times in each of steps 10-15 (at the start of 11-15) and
  stayed near its best (the worst answer mostly 0.82-0.85) instead of collapsing as without; its last machine
  minimises to 358 states, too many rather than too few.
- **Drops the fork didn't catch:** 4 large drops (the worst answer from >= 0.9 down > 0.3 between
  diagnoses) in the fork arm, all recovered: 2 forked in the next epochs (seed 2 at step 6, seed 3
  at step 8), 2 never forked (seed 2, steps 6 and 7: the screen, every answer's share, stayed
  above 0.6 of its peak while the worst answer fell). Without the fork, 8 drops in 3 runs (`readout`), 7
  recovered; the lost one is seed 3's.
- **Next, as registered:** the `p` 11 ladder, the fork on (two factors of 32, then 64; 4-6 seeds).
  The alternatives the logs suggest (a trigger on the worst answer; a step that ends only when
  its peak is recovered) stay in next-steps.md.

Findings (the approximation claim, A1-A3 and D1-D6 registered before its data; the workplan
retired: `git show c201622:docs/approximation-claim-workplan.md`), 2026-10-16/17; p 5, two factors of 16, reading the output, one vocabulary for every
arm (`--vocabulary sum difference product`), stage 5's settings otherwise (400 epochs, 180 minutes
only a guard, stop at certified, no fork), seeds 200-219 in each of four arms: `quotient` alone,
`sum` (the quotient and the sum, one approximation), `family` (the cover) and `product` (the
quotient and `A B mod p`, an extra output that is not an approximation). Code 7618dfe,
OPENBLAS_NUM_THREADS=1: pyramidon seeds 214-219 (`~/code/wt-approx`, each arm 3 at once in turn),
2026-10-16 15:13 to 00:46; jebel seeds 200-213 (`~/code/wt-p11`, two arms at once, 3 workers
each), 23:38 to 06:21. Every certified network recertified in both verdicts (`certify/`,
`post-jebel.txt`; on jebel at 7618dfe: pyramidon had no memory to spare beside its long tests, and
the certificate is deterministic). Data in data/window-recurrent/approximation-claim/, manifest
`approximation-claim.sha256` 9517dc2b...cf6f2772 (1,501 files). The readout is `readout data/window-recurrent/approximation-claim --pairs
family:quotient sum:quotient sum:product` (`readout.txt`).

| arm | certified (Wilson 95%) | successes: steps, epochs (median) | failures near a solution |
| --- | --- | --- | --- |
| family | 15/20 (0.53-0.89) | 6-12, 78-235 (136) | 4 of 5 |
| sum | 10/20 (0.30-0.70) | 7-11, 64-160 (116) | 5 of 10 |
| quotient | 1/20 (0.01-0.24) | 12, 279 | 4 of 19 |
| product | 0/20 (0.00-0.16) | - | 0 of 20 |

- **A1, A2 and A3 hold.** Fisher's exact, one-sided, Holm over the three (D5): A1 family more often
  than quotient p 5.0e-6 (adjusted 1.5e-5); A3 sum more often than product p 2.2e-4 (4.4e-4); A2
  sum more often than quotient p 0.0017 (0.0017). By the plan's reading: one approximation helps,
  and because it is an approximation, not because it is one more output.
- **Stage 5 replicates with the wider vocabulary:** the family 15 of 20 and the quotient 1 of 20
  here against 13 and 1 there (A1), so the product's token, one more embedding row, changes neither.
- **The cover does more than one approximation:** 15 of 20 against 10 (not a registered test).
- **Every recertification agrees:** 26 of 26 certified in both verdicts against division's 25-state
  machine, 0 wrong outputs, 0 wrong states, each minimised to 25; 36-137 joint codes reached (36-131
  strict); snap ratio >= 1.000 (653 near ties branched, none strict), logit lead >= 0.32.
- **Every failure spent its 400 epochs** (steps 14-22; no run ended on the guard).
- **The product neither helps nor is learned** (exploratory, after the data: each run's step end
  with the best worst answer). The quotient is the worst answer in all 20 product runs, at a
  median 0.59 right against the quotient alone's 0.66, and the product itself 0.71; no product run
  came near a solution (the quotient alone's 4 did), 11 of 20 collapsed. The sum and the cover's
  runs are right on their extra outputs at a median 1.00. Read digit by digit, the sum's next value
  is a function of the sum and the column, so it can be learned on its own; the product's needs
  both residues, as the quotient's does: the extra output that doesn't help is one as hard as the
  target. The family's worst answer at its best is the difference in 18 of 20.
- **Large drops** (`readout`): the family 34 in 14 runs (29 recovered), the sum 14 in 7 (10), the
  quotient 11 in 5 (9), the product none.
- **Every certified network passes every long-test case:** 26 of 26, all 375,000 cases each
  (division's U1 and S1 at 128, 256 and 1,024 columns; 9,750,000 in all), on jebel at 7618dfe,
  6 at a time, 2026-10-17 06:37 to 11:32, 46-68 minutes a network (`long-tests/`,
  `post-jebel.txt`; copied to pyramidon, manifest `approximation-claim-long-tests.sha256`
  a1c41b20...4ebd4c05, 53 files). Archived in indrajala-models
  (window-recurrent/2026-10-15-division, `*-approx-*`).

Findings (the `p` 11 ladder: the p11-trigger workplan, stage 3, registered before its data:
works if at least one network certifies against division's 121-state machine in both verdicts),
2026-10-16; p 11, the family, two factors of 32 codes (`s` 16), reading the output, the fork on
with today's trigger (pooled screen below 0.6 of the step's peak), 600 epochs (900 minutes only a
guard), stop at certified, seeds 0-5. Code 7618dfe, OPENBLAS_NUM_THREADS=1, jebel, 6 at once,
21:38 to 23:38 (`~/code/jebel-chain.sh`, after E2b'). Data in data/window-recurrent/p11-ladder/,
manifest `p11-ladder.sha256` ef3281cd...f460b7dc (155 files, run from data/window-recurrent). The
readout is `readout data/window-recurrent/p11-ladder --arms family`.

| seed | end | step | epochs | minutes | forks | worst answer at best | large drops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | spent | 17 | 600 | 97.5 | 12 | 0.75 | 0 |
| 1 | certified | 10 | 293 | 34.5 | 1 | 1.00 | 0 |
| 2 | certified | 16 | 402 | 52.1 | 16 | 1.00 | 2 |
| 3 | spent | 27 | 600 | 119.9 | 0 | 0.41 | 0 |
| 4 | certified | 31 | 527 | 91.7 | 72 | 1.00 | 1 |
| 5 | certified | 16 | 543 | 80.1 | 11 | 1.00 | 1 |

- **Works: 4 of 6 certified** (Wilson 95%: 0.30-0.90) at the ladder's first rung, two factors of
  32, so the second rung (64 a factor) doesn't run. The largest machine learned so far: 121 states
  against `p` 7's 49.
- **Every one robustly certified:** both verdicts against division's machine mod 11, 0 wrong
  outputs, 0 wrong states, each minimised to 121; 210-428 joint codes reached strict (snap ratio
  1.0), 212-448 at 1.05 (821 near ties branched, none strict); logit lead >= 0.26 (seed 4 at 1.05;
  0.37 strict). `certify/`, `certify.log`, on pyramidon.
- **Every one passes every long-test case:** all 663,000 each (division's U1 and S1 at 128, 256 and
  1,024 columns), on pyramidon at c4a91e7 (`~/code/p11-long-tests.sh`), one at a time, 2026-10-17
  04:02 to 14:11, 140-174 minutes a network (`long-tests/`, `long-tests.log`), manifest
  `p11-ladder-long-tests.sha256` d0ab5c4e...6d0133250c49 (5 files). Archived in indrajala-models
  (window-recurrent/2026-10-15-division, `family-f2x32-p11-*`).
- **No run collapsed, and every large drop recovered:** 4 large drops in 3 runs (`readout`), all
  certified runs. The two failures never came near a solution: seed 0's worst answer at best 0.75,
  seed 3's 0.41 (at step 9, then 18 more steps without the fork firing once). Neither failure is the
  collapse the trigger targets.
- **The fork fires most where the run succeeds slowest:** 72 forks in seed 4 (certified at step 31,
  the latest), 1 in seed 1 (the earliest, step 10). Exploratory: whether those forks rescued seed 4
  or held it back isn't separable without the alternate trigger's paired replays (stage 4).
- **Next:** stage 4, the alternate trigger (D7), deferred to the beyond-arithmetic targets, where
  a trigger can be tuned across machine classes (the plan's status, 2026-10-17); the certificate
  guard (`replay --guard`, the snap-wall workplan's stage 2d) joins the fork rather than
  superseding it.

Findings (the general algebras' build probes, stage 2 of indrajala-research's docs/workplans/general-algebras-workplan.md; probes,
no claim), 2026-10-17, pyramidon, the workplan's cell (two factors, `--codes-ratio 32/11`, reading
the output, the fork), OPENBLAS_NUM_THREADS=1, in a scratch directory (not kept):

- **Budget:** 20 epochs each of S5, move-to-front 5, random-5-0 and division `p` 11 in one pool, 4
  at once: a machine's epoch about 2.6 s at steps 1-3 against division's 5.5 s (its window three
  tokens, a machine's one); under 80 MB a process.
- **The final identity token is needed:** without it (words of n tokens alone), 6-state cells,
  seeds 0-1, 150 epochs: 1 of 6 certified (move-to-front 3), the other 5 keeping one code from
  step 1, where no answer needs the state (state share 0). With it (MachineEncoding.draw, as
  division's final zero column), the same seeds: move-to-front 3 certified 2 of 2 (step 8, 3-7
  minutes); S3 0 of 2, 2-4 codes in use, its answers about half right at 150 epochs; random-3-0 0
  of 2, still one code. Round 1 (E1, 600 epochs, 6 seeds) runs as registered.

Findings (round 1 of the general algebras, stage 3 of indrajala-research's docs/workplans/general-algebras-workplan.md, E1
registered before its data), 2026-10-17/18; `n` 3 and 4, three classes, the workplan's cell (two
factors, `--codes-ratio 32/11`, reading the output, the fork with today's trigger, the guard off),
600 epochs, stop at certified, seeds 0-5. Code 5a89bfe, OPENBLAS_NUM_THREADS=1; seeds 0-4 on
pyramidon (21:46 to 09:22), seed 5 on jebel (06:22 to 07:42; runs are bit-identical across the
machines). Data in data/window-recurrent/general-algebras/round1/ (jebel's seed 5 merged in;
`readout.txt` is the per-run readout, `certify/` the recertifications), manifest `round1.sha256`
71f54433...e13a3a (1,239 files, run from general-algebras/); jebel's seed 5 as copied,
`round1-seed5-jebel.sha256` c9f635ff...c9ed0f (239 files, run from inside
round1-seed5-jebel/).

| cell | certified (Wilson 95%) | E1 | slowest success's epochs | joint codes reached | smallest lead |
| --- | --- | --- | --- | --- | --- |
| move-to-front-3 | 6 of 6 (0.61-1.00) | works | 87 | 12-21 | 1.59 |
| word-problem-s3 | 1 of 6 (0.03-0.56) | works, a decline, undecided | 13 | 10 | 3.56 |
| random-3-0 | 0 of 6 (0.00-0.39) | fails | - | - | - |
| move-to-front-4 | 4 of 6 (0.30-0.90) | works | 398 | 56-72 | 2.25 |
| word-problem-s4 | 4 of 6 (0.30-0.90) | works | 576 | 38-65 | 2.59 |
| random-4-0 | 0 of 6 (0.00-0.39) | fails | - | - | - |

- **Every certified network passes both verdicts** (snap ratio 1.0 and 1.05): 0 wrong outputs,
  0 wrong states, each minimised to its machine's states (6 or 24); joint codes reached are the
  same in both verdicts but S4 seed 1 (49 strict, 54 at 1.05). No factor reads as one of the
  machine's coordinates; the names these certificates carry ("position-1 + ... + position-n") are
  the trivial one-block partition that stage 2 of the consolidation workplan stopped naming.
- **E1 is refuted for the random class at `n` 3 and 4** (it fails where the other two work), but
  random-3-0's failure is a flaw of the cell, not a reading of the class: every one-token word
  answers label 2, so step 1 passes on one code and all six runs stay there, spent at step 56 with
  the worst answer 0.633 from step 1. Recorded, not fixed here; the fix (D3, `s0`) is in the
  consolidation workplan's arms, a new registered experiment.
- **Random-4-0's six failures are one mechanism** (D9, `git show df1dcac:docs/code-consolidation-workplan.md`, "Why"):
  an inexact core of first codes and extra codes reached by longer words, whose transitions break
  far more often; best steps 8-15, worst answer at best 0.86-0.96, four of six collapsing after it.
  S4's near miss (seed 0, 0.999 at step 13) has the same signature around an exact 24-code core;
  its other failure (seed 5, 0.958) wasn't diagnosed.
- **S3's five failures are a state that never forms** (worst answer at best 0.44-0.61 at steps
  2-5; at most 6 codes on 2-6 of 6 states), as is move-to-front-4 seed 3 (0.785 at step 3);
  move-to-front-4 seed 5 came near (0.999 at step 14).
- **The budget crowds at `n` 4:** S4's slowest success took 576 of the 600 epochs, move-to-front-4's
  398; at `n` 3 every success took 87 epochs or fewer.
- **Next:** round 2 (`n` 5) is deferred until the random class passes round 1 (D5, the owner); the
  consolidation workplan tests both remedies (A, consolidation; B, longer words) with D3 in every
  arm, after its B-origins probes (stage 2b).

Findings (B's origins, stage 2b of `git show df1dcac:docs/code-consolidation-workplan.md`; probes, their answers
written before the data), 2026-10-18; round 1's cell (as above), 600 epochs, seeds 0-2, on four
random machines with random-4's tokens and classes. Code 1496113, OPENBLAS_NUM_THREADS=1, jebel, 6
at once, 11:07 to 12:36 (a re-launch after a power cut; the first launch's three certified runs
repeated by bits). Data in data/window-recurrent/consolidation/b-probes/ (`readout.txt`,
`certify/`, `certify.log`, `reading/` with `breaks` at `d` 1 and 2 and `by-length` on each failure's
best step network), manifest `b-probes.sha256` ca688177...c8bd2b74 (291 files, run from
consolidation/; `certify/` and `reading/` added on pyramidon after it).

| machine | states | certified (Wilson 95%) | slowest success's epochs | failures' worst answer at best |
| --- | --- | --- | --- | --- |
| random-q8-1 | 8 | 3 of 3 (0.44-1.00) | 61 | - |
| random-q12-0 | 12 | 2 of 3 (0.21-0.94) | 169 | 0.984 |
| random-q16-0 | 16 | 0 of 3 (0.00-0.56) | - | 0.968-0.995 |
| random-perm-4-0 | 24 | 0 of 3 (0.00-0.56) | - | 0.788-0.908 |

- **Every certified network passes both verdicts:** 0 wrong outputs, 0 wrong states, each
  minimised to its machine's states; 17-19 joint codes reached at 8 states, 32-49 at 12; logit
  lead >= 3.58, no near ties.
- **Merging isn't failure B's cause:** random-perm-4-0 (every token a bijection) fails as random-4-0
  does: every state reached, at `d` 1 first codes breaking 6-33% and extra codes 37-57%, breaks from
  depth 2-3; at `d` 2 only 5-19 of 24 first codes right, breaking 27-60% (S4's near miss: 1 of 120);
  exact accuracy at most 0.15 by length 16.
- **No sharp edge in size:** failure B (no exact core at `d` 2) appears in one run at 12 (seed 1:
  1 of 12 first codes right at `d` 2) and one at 16 (seed 0: 9 of 16, breaking 31%), and in every
  run at 24. random-q16-0's seeds 1 and 2 instead fail as S4's near miss: an exact core at `d` 2
  (first codes breaking 1-3%), extra codes breaking 6-7%, exact accuracy 0.99 and 0.98 at length 8,
  0.89 and 0.85 at 16, 0.43 and 0.33 at 64; every random-q16-0 run collapsed after its best step.
- **Next:** remedy B (`--long-tail 4`) against the control on random-q16-0, seeds 0-5, registered in
  the plan's stage 2b before its data.

Findings (B and then A on random-q16-0, stage 2b of `git show df1dcac:docs/code-consolidation-workplan.md`; probes,
their answers written before the data), 2026-10-18; round 1's cell, 600 epochs, seeds 0-5, three
arms on one commit (D5): the control, B (`--long-tail 4`) and A (`--consolidate`, `d` 2). Code
0b4a6b9 (~/code/wt-bq16, its own venv), OPENBLAS_NUM_THREADS=1, jebel, 6 at once: the control 12:39
to 13:37, B to 16:31, A 16:37 to 17:37. Data in data/window-recurrent/consolidation/b-q16/
(`control/`, `b/`, `a/`, each with its `.out` and `.readout.txt`; `reading/` with `breaks` at `d` 1
and 2 and `by-length` on each run's best step network, A's in `reading/a/`), manifests
`b-q16-control-b.sha256` 4d2136a6...  (460 files) and `b-q16-a.sha256` 0e449fc6... (204 files),
both run from consolidation/; the readouts and `reading/` added on pyramidon after them.

| arm | certified (Wilson 95%) | worst answer at best | came near (>= 0.95) | minutes |
| --- | --- | --- | --- | --- |
| control | 0 of 6 (0.00-0.39) | 0.895-0.995 | 4 | 49-58 |
| B | 0 of 6 (0.00-0.39) | 0.937-0.999 | 4 | 130-174 |
| A | 0 of 6 (0.00-0.39) | 0.976-0.999 | 6 | 48-60 |

- **Random machines set aside** (the plan's rule at A 0 of 6, the owner's direction): at round 1's
  cell, with either remedy, the method certifies random machines to about 12 states. At 16 the
  runs found near solutions and didn't keep them: every run of all three arms collapsed after its
  best step (steps 4-14). That isn't a capacity limit (9 times the states in joint codes). The
  general algebras go on without the random class (general-algebras-workplan.md, D5).
- **The control repeats 2b by bits:** seeds 0-2, every log row but the seconds.
- **B lengthens the exact prefix where a core forms, and only there:** seeds 2 and 4 reached an exact
  core at `d` 2 (first codes breaking 1-3%) with extra codes breaking 2-3% (the control's best 6-7%),
  exact accuracy 0.98 at length 16 and 0.916 at 64 (the control's best 0.89 and 0.43). Seeds 1 and 5
  never formed a state (1 right code at `d` 2, failure A); seeds 0 and 3 had no exact core.
- **A brings every run near, but doesn't move C2:** codes reached 21-64 (median 52; the control's
  25-63, median 50); extra codes break more often from right codes at `d` 2 (0.30-0.56 in five
  seeds, the control's median 0.17). Seed 0 is the exception and the nearest run of the three arms:
  at step 14, 21 codes on 16 states, its first codes breaking 0-1%, exact accuracy 0.996 at 16 and
  0.975 at 64. It collapsed after that step.
- **Why A pulls little:** at the best steps the network's own groups number 22-63, against 16
  states, and duplicates pulled are a median 6-13% of the codes over a run. An extra code's
  outputs are wrong, so within `d` tokens they differ from its state's first code, and A's grouping
  (the network's own `d`-equivalence) can't merge them.
- **Next:** stage 2c's retention probes, registered before their data: P1 (A with the certificate
  guard on S4 and move-to-front 4, on `stone`) and P2 (A with the step's best restored on
  random-q16-0, on `jebel`).

Findings (the division ladder, retired 2026-10-19: `git show f9101d8:docs/division-ladder-workplan.md`; L1, L2 (the budget) and
L3 (the width) each registered before its data), 2026-10-17 to 19; `p` 11's cell (two factors of width 16, `ceil(32 p / 11)` codes
each, reading the output, the fork, 600 epochs) at higher primes, seeds 0-5 a rung. Code 440a680,
OPENBLAS_NUM_THREADS=1. L1 round 1 (`p` 17 and 23) on jebel, 8 at once, 2026-10-17 19:53 to 22:29;
round 2 (`p` 13) on jebel 2026-10-18 04:32 to 06:03; L2 (`p` 13 at 1,800 epochs) on stone, 6 at once,
06:39 to 22:40. Data in data/window-recurrent/division-ladder/ (`round1/p17`, `round1/p23`,
`round2`, `l2`), manifests `round1.sha256` 2b6a9a3a... (288 files), `round2.sha256` ab734b02...
(160) and `l2.sha256` c453dbc5... (453, run from division-ladder/; L2's `readout.txt` added on
pyramidon after it).

| `p` | states | epochs | certified (Wilson 95%) | worst answer at best |
| --- | --- | --- | --- | --- |
| 11 (the p11 ladder, above) | 121 | 600 | 4 of 6 (0.30-0.90) | - |
| 13 | 169 | 600 | 0 of 6 (0.00-0.39) | 0.394-0.916 |
| 13 (L2) | 169 | 1,800 | 0 of 6 (0.00-0.39) | 0.394-0.936 |
| 13 (L3, factors 32 wide) | 169 | 600 | 0 of 6 (0.00-0.39) | 0.382-0.994 |
| 17 | 289 | 600 | 0 of 6 (0.00-0.39) | 0.609-0.774 |
| 23 | 529 | 600 | 0 of 6 (0.00-0.39) | 0.360-0.386 |

- **L1: the failure point lies between `p` 11 and 13** (121 and 169 states), D4's search ending at
  adjacent primes. No run at 13, 17 or 23 came near (worst answer >= 0.95). Every one spent its
  600 epochs, at steps 15-29.
- **L2: not the budget.** At three times the epochs, 0 of 6 again. Each seed's first 600 epochs
  repeat round 2's by bits (across machines: jebel, then stone). The best worst answer is round 2's
  in five seeds, reached in the first 600 (seed 1: 0.916 to 0.936). Afterwards five runs collapsed,
  their last machines minimising to 1-96 states.
- **Noted, not explained:** `p` 17's runs came nearer (0.61-0.77) than most at 13 (0.39-0.42 in four
  seeds), so the worst answer at best isn't monotone in `p`.
- **L3: not the width either** (registered answer at 0 of 6). `p` 13 with each factor 32 wide
  instead of 16, the same 38 codes, else L1's round 2 (440a680; stone, 6 at once, 2026-10-19 01:22
  to 03:20; data in `l3/`, manifest `l3.sha256` 1e6545d6... (137 files, run from division-ladder/;
  `readout.txt` added on pyramidon after it)). Every run spent its 600 epochs (83-118 minutes,
  steps 14-25). So the edge between 11 and 13 stands, explained by neither the budget nor the width.
  With the general algebras' round 2 (0 of 12 at 120 states in `p` 11's cell), the limit lies in
  training at this cell, not in a state count or a code space.
- **Descriptively, the width moved the runs** (not a claim, one change on six seeds): at their
  best step four of six formed a state (share 0.996-1.0, against two of six at width 16), using
  fewer codes on one factor (17-18 of 38 in three seeds). Seed 5 came nearest of any run at 13: worst
  answer 0.994, sampled accuracy 0.99 at 48, its machine minimising to 172 against 169.

Findings (P2, A with the step's best restored, stage 2c of `git show df1dcac:docs/code-consolidation-workplan.md`; a
probe, its answers written before the data), 2026-10-18; random-q16-0, 2b's cell with `--consolidate
--restore-best`, seeds 0-5, 600 epochs. Code b5f9a05 (~/code/wt-p2, its own venv, crate 40a2761),
OPENBLAS_NUM_THREADS=1, jebel, 6 at once, 17:44 to 18:46. Its control is 2b's A arm: A's first 15
epochs repeat by bits on b5f9a05. Data in data/window-recurrent/consolidation/b-q16/p2/ (and
`p2.out`, `p2.readout.txt`), manifest `b-q16-p2.sha256` b24d3d97... (209 files, run from
consolidation/; the readout added on pyramidon after it).

- **0 of 6 (Wilson 95%: 0.00-0.39), so the limit recorded in 2b stands:** near solutions aren't
  retained at random-q16-0. Every run spent its 600 epochs (51-62 minutes) and came near (worst
  answer at best 0.976-1.000).
- **The restore kept each step's best:** 21-27 restores a run. A plateaued step had fallen a
  median 4-11 points of screen below its best (at most 26-35), and its end regained them. The best
  came later than A's (steps 10-25 against 7-14), with fewer collapses (0 in two runs, 3 and 5 in
  the worst; A had 1-4 in every run).
- **But the step's best was never a solution:** the best steps' networks minimise to 37-62 states
  against 16, so none certified. Seed 4's best step passed its screens perfectly and its machine
  still minimised to 57. Within each later step the near solution broke again before its plateau.
  Keeping a near solution doesn't make it a machine. The extra codes are wrong, as A's reading
  found.

Findings (P1, A with the certificate guard, stage 2c of `git show df1dcac:docs/code-consolidation-workplan.md`; a
probe, its answers written before the data), 2026-10-19; round 1's cell with `--consolidate --guard
both` (each seed unguarded and guarded), word-problem-s4 and move-to-front-4, seeds 0-5, 600
epochs. Code bab7690 (crate 7df828d, its own venv), OPENBLAS_NUM_THREADS=1, stone (i7-1255U), 6
at once, 2026-10-18 22:41 to 2026-10-19 01:06. Data in data/window-recurrent/consolidation/p1/
(and `certify/`, added on pyramidon after it), manifest `p1.sha256` 2a753178... (662 files, run
from consolidation/).

| cell | unguarded | guarded | round 1's control (another commit) |
| --- | --- | --- | --- |
| word-problem-s4 | 6 of 6 | 6 of 6 | 4 of 6 |
| move-to-front-4 | 3 of 6 (seeds 1, 2, 5) | 1 of 6 (seed 5) | 4 of 6 |

- **The guard certified fewer seeds, not more,** and isn't kept with A. Every certified network
  was recertified in both verdicts (28-75 codes reached, minimised to 24, snap ratio >= 1.00-1.25).
- **It froze near solutions.** Move-to-front 4's seeds 1 and 2 repeat the unguarded runs by bits
  until the guard arms (steps 11 and 13, a perfect screen). Their networks were right at snap 1.0
  but not at 1.05: 6-8 near ties that, branched, give 1-4 wrong states (minimised to 32 and 28).
  Armed for 18 and 10 epochs, the guard undid 96% and 100% of the updates, so the margins never
  widened, and the stall rule ended the runs (steps 28 and 22). Unguarded, the same seeds
  certified at steps 14 and 13. On S4 it undid 61-77% in seeds 4 and 5, which certified 3 and 7
  steps later than unguarded; in the other four S4 seeds it undid 0-2 updates.
- **So a perfect screen isn't a certificate,** and a guard armed by one keeps whatever it
  finds. Arming it only on a step that passes both verdicts is the change it needs (added to
  next-steps.md's guard entry; not built).
- **C2:** every certified network's reached codes are all right within `d` (2 for S4, 3 for
  move-to-front 4), breaking 0. The stalled networks' codes are too, so their fault is the
  margins. Move-to-front 4's other failures, seeds 0, 3 and 4 in both arms (never armed, so
  identical), are failure A: 3-5 codes, state share 0.09-0.29, spent at steps 41-46.
- **Descriptively, A against round 1's control:** A certified S4's seeds 0 and 5 (the control's
  failures) and move-to-front 4's seed 5. It lost move-to-front 4's seeds 0 and 4 to failure A,
  where the control certified them.

Findings (the commitment-control workplan's diagnostic, registered in #851 before its data; a
probe, its answers written before the data), 2026-10-19; round 1's failure-A runs (word-problem-s3
seeds 0, 1, 3, 4 and 5, move-to-front-4 seed 3), round 1's cell with one change, `--commitment-from
6` (every factor's commitment 0 for steps 1-5). Code 6e5d746 (~/code/wt-cc), OPENBLAS_NUM_THREADS=1,
jebel, 6 at once, 04:14 to 05:13, when the owner stopped it (S3 seeds 0, 1, 3 and 4 at steps 41-45,
516-558 of 600 epochs, with checkpoints; the reading was complete). Its control is round 1's runs:
without the flag, S3 seed 0's first 12 log rows repeat round 1's by bits (checked on 6389d26, crate
afa7204, after the data: `commitment-control/control-bits-check/`; not on the build's commit). Data
in data/window-recurrent/commitment-control/diagnostic/ (and `diagnostic-certify/`, added on
pyramidon after it), manifest `diagnostic.sha256` d5262970... (237 files, run from
commitment-control/).

| run | round 1, state's share at steps 1-5 | the commitment off, steps 1-5 | from step 6, the commitment on |
| --- | --- | --- | --- |
| S3 seed 0 | 0.04-0.27 | 0.06, 0.28, 0.41, 0.95, **0.97** | 0.56 at step 6, 0.43 by step 10 |
| S3 seed 1 | 0.06-0.21 | 0.06-0.31 | one code a factor from step 6 |
| S3 seed 3 | 0.00-0.22 | 0.06, 0.21, 0.28, **0.95**, 0.15 | 0.13, one code a factor by step 10 |
| S3 seed 4 | 0.06-0.32 | 0.06-0.30 | 0.24-0.27 |
| S3 seed 5 | 0.06, then 0.00 | 0.06-0.32, **1.0** at step 5 | **certified at step 6** |
| move-to-front 4 seed 3 | 0.18-0.38 | 0.04-0.76, **0.87** at step 5 | 0.69 at step 6; 0.94 at step 29, spent (minimised to 62 against 24) |

- **3 of 6 formed a state by step 5's end** (share at least 0.5: S3 seeds 0 and 5, move-to-front 4
  seed 3), against none of the same runs in round 1. By the registered answer, **the commitment's
  pull is failure A's cause, and the thermostat (D1 (a)) is worth building.** S3 seed 5, at a share
  of 0.00 by step 3 in round 1, certified at step 6: recertified in both verdicts (11 codes
  reached, minimised to 6, snap ratio >= 1.412, logit lead >= 4.2).
- **The commitment's return at full weight eroded the states it found:** S3 seed 0 fell from 0.97 to
  0.56 in one step and never recovered; move-to-front 4 seed 3 fell from 0.87 to 0.69 and regained
  0.94 only by step 29. So the thermostat's ramp back (D2) and its hysteresis are needed, not a
  phase schedule.
- **Without the commitment a state can form and still be lost:** S3 seed 3 reached 0.95 at step 4
  and fell to 0.15 at step 5 with the commitment still off. Forming isn't keeping: the thermostat
  should restore the commitment once a state has formed (its restore level), not leave it off for a
  fixed number of steps.
- Two runs (S3 seeds 1 and 4) formed nothing either way: the commitment isn't failure A's only
  cause.

Findings (the thermostat's probe, the commitment-control workplan's stage 2, D5'; a probe, its
expectations written before its data), 2026-10-19; the commitment diagnostic's six failure-A runs
(word-problem-s3 seeds 0, 1, 3, 4, 5; move-to-front-4 seed 3; round 1's cell) with `--thermostat`
(D2': off below 3 effective codes, back at half the codebook over 20 epochs). Code 7617c52,
OPENBLAS_NUM_THREADS=1, pyramidon, 6 at once, 12:56 to 16:07. Its controls: round 1 (all six
failure A) and the diagnostic (the commitment off for steps 1-5: 3 of 6 formed a state, S3 seed 5
certified). Data in data/window-recurrent/commitment-control/thermostat-probe/, manifest
`thermostat-probe.sha256` a7b190e7... (246 files, run from commitment-control/).

| run | the diagnostic (off for steps 1-5) | the thermostat: best state share |
| --- | --- | --- |
| S3 seed 0 | formed (0.97), eroded | 0.44 |
| S3 seed 1 | none | 0.24 |
| S3 seed 3 | formed (0.95), lost | 0.32 |
| S3 seed 4 | none | **0.74**, then collapsed to one code |
| S3 seed 5 | **certified at step 6** | 0.23, one code by step 9 |
| move-to-front-4 seed 3 | formed (0.87) | 0.57, held |

- **0 of 6 certified; 2 formed a state, one of them lost** (every run spent its 600 epochs). The
  expectation met is "states form but the restore is wrong": not the thermostat's success, so C3
  doesn't launch as registered.
- **Code use is the wrong signal for a formed state.** In S3, a factor's effective codes reach
  half its codebook (4 of 8) within steps 2-3, while the state's share is still about 0.2, so the
  commitment comes back before any state has formed. S3 seed 5's was fully back by step 3; under
  the diagnostic's schedule it stayed off through step 5, the state formed at step 5 (share 1.0),
  and the run certified at step 6. A network uses many codes before its codes mean anything.
- **Switching it off again doesn't undo a collapse:** S3 seeds 3, 4 and 5 fell to one or two codes
  with the commitment returned, and stayed there once the thermostat had switched it back off.
- **What the diagnostic's schedule got right:** off until the network's answers came right. Next
  (the workplan's D2''): restore the commitment when the step's screen passes, an answers-only
  signal (the fork's trigger is one too).

Findings (round 2 of the general algebras, stage 4 of indrajala-research's docs/workplans/general-algebras-workplan.md, E1 as
amended before its data in #850), 2026-10-18; `n` 5, word-problem-s5 and move-to-front-5, round 1's
cell unchanged, 600 epochs, seeds 0-5. Code ff2b2f2 (~/code/wt-round2, crate 40a2761),
OPENBLAS_NUM_THREADS=1, jebel, 6 at once, 18:47 to 21:35. Data in
data/window-recurrent/general-algebras/round2/ (`readout.txt` added on pyramidon after it),
manifest `round2.sha256` c2e1b2a5... (480 files, run from general-algebras/).

| machine | states | certified (Wilson 95%) | came near (worst answer >= 0.95) | best minimised |
| --- | --- | --- | --- | --- |
| word-problem-s5 | 120 | 0 of 6 (0.00-0.39) | 5 | 175-842 |
| move-to-front-5 | 120 | 0 of 6 (0.00-0.39) | 2 | 314, 346; 3 (four runs) |
| division `p` 11 (the p11 ladder) | 121 | 4 of 6 (0.30-0.90) | - | - |

- **Both extremes fail at 120 states (E1's third answer), not one class.** Every run spent its 600
  epochs (43-92 minutes, steps 22-55). Division certifies at 121 in the same cell, so the state
  count alone isn't the limit.
- **Extra codes, as S4's near miss in round 1, at scale.** All six S5 runs formed a state (share
  0.88-1.0). At `d` 2 their 120 states' first codes break 0-4%, an exact core. But they reach
  385-748 codes, whose transitions break 83-100%, and the machines minimise to 175-842. Seed 5 came
  nearest: its first codes break 0 of 720, with exact accuracy 1.0 at 24 and 0.999 at 48, minimised
  to 175. Seeds 4 and 5 reached a worst answer of 1.000 at their best step and still didn't certify.
- **Move-to-front-5 has no exact core:** seeds 0 and 2 formed a state (0.93, 0.97), but at `d` 4
  only 69 and 40 of 337-339 codes are right, the first codes breaking 49% and 39%.
- **Failure A in move-to-front-5's other four seeds** (1, 3, 4, 5): 2-3 codes a factor, share
  0.21-0.24, minimised to 3; seeds 1 and 3 collapsed.

Findings (the readout probe, after L3; the owner, 2026-10-19: "build and run it now"; a probe, its
answers written before the data: certified, and width 32 then the readout is division's candidate
cell at 13; else the errors aren't an under-trained readout), 2026-10-19; L3 seed 5's best
network (step 15: 0 wrong states, 22 wrong outputs, minimised to 172 against 169), its output layer
trained alone (`readout-tune`, every other layer and the recurrence frozen, tested) on step 15's
lengths, 30 epochs. Code b7285fb, pyramidon. Data in
data/window-recurrent/division-ladder/readout-tune/l3-seed5/, manifest `readout-tune.sha256`
e3776728... (4 files, the final network's certificate among them, run from division-ladder/).

- **Not certified: the errors aren't an under-trained readout.** After one epoch the machine
  minimised to exactly 169: the 3 extra states were only the readout's errors. Wrong states stayed 0.
  But wrong outputs stayed at 15-19 for 29 more epochs (15 at the end), with no trend.
- **The 15 are confident and on the states' own codes:** 15 of 1,521 (state, query) pairs, over 12
  of the 169 states. 14 are on a state's first code (one on an extra code), with logit leads of 0.15-4.3.
  So the window's features for those pairs give the wrong answer, and a linear readout over them
  can't separate it. The poolings read the output layer's input, so every earlier layer also feeds
  the state: no further layer can be trained without risking the machine.
- **So at width 32 and `p` 13 the state machine was learned exactly, and the answers it reads out
  were 1% wrong.** Next candidates (not registered): the whole network trained with the certificate
  guard armed only on a right state machine (P1's lesson), or the answers given their own head
  apart from what the state reads.

Findings (W1, the window branches workplan's width claim, registered in #863 before its data),
2026-10-19; L3's cell (division `p` 13, two factors 32 wide with 38 codes each, reading the output,
the fork, 600 epochs, stop at certified) with the window's residual stream 32 wide instead of 16
(`--window-width 32`: `FFN` 64, 2 heads), seeds 0-5. Code 85a9e44 (crate 40a2761),
OPENBLAS_NUM_THREADS=1, jebel, 6 at once, 05:28 to 07:20. Its control is L3: the flag's default
repeated L3's first epochs by bits on 85a9e44 (4 epochs of seed 0). Data in
data/window-recurrent/window-branches/w1/, manifest `w1.sha256` 35064b2a... (102 files, run from
window-branches/; `readout.txt`, `certify/` and `long-tests/` added on pyramidon after it).

| cell | window | certified (Wilson 95%) | certified at | worst answer at best (failures) |
| --- | --- | --- | --- | --- |
| L3 | 16 wide | 0 of 6 (0.00-0.39) | - | 0.382-0.994 |
| W1 | 32 wide | 5 of 6 (0.44-0.97) | steps 8-11, 209-255 epochs | 0.808 (seed 3) |

- **W1 works, and helps over L3** (5 of 6 against 0 of 6, Fisher one-sided p 0.008): by the
  registered answer, a crowded 16-wide window limited division at 13. Every certified network was
  recertified on pyramidon (8941eb7) in both verdicts (`certify/`): 253-412 codes reached,
  minimised to 169, 0 wrong outputs and 0 wrong states; strict snap ratio >= 1.000-1.001, logit
  lead >= 0.64-1.63. The runs certified fast: 29-37 minutes, where L3's spent their 600 epochs.
- **Its long test passes:** seed 2's certified network (step 9) passed every one of its 807,000
  cases (U1 and S1 at 128, 256 and 1,024 columns; pyramidon, 8941eb7, 10,925 s; `long-tests/`).
- **The certified machines are compact:** at the certified step each factor uses 22-32 of its 38
  codes, the state's share 1.0. Every certified run's worst answer reached 1.000 at step 6 or 7,
  one to five steps before its certificate.
- **Seed 3 never formed a state:** from step 5 it used every code (36-38 a factor) with the state's
  share 0.72-0.84, its machine minimising to 360-733; 600 epochs spent at step 19, no collapse. Not
  failure A (few codes): the opposite, codes saturated.
- **What W1 leaves to the branches (D3, D8):** a wider stream suffices at 13, so the branches'
  claim on division (B1, 16 wide) reads against W2, the window at 22 matched to B1 in parameters:
  separation against size. A branches-at-32 cell could read only "no harm" against W1.
Findings (the window branches workplan's stage 6: B1, W2, B2 and B3, registered in #863 and #866
before their data; and a side probe), 2026-10-19; each cell its control's, with `--branches` (a
two-block trunk, then one single-head block a factor and one for the answers, `D` 16) or
`--window-width 22` (W2), seeds 0-5, 600 epochs, the fork, stop at certified. OPENBLAS_NUM_THREADS=1.
B1 on stone from 6d808a5 (10:00 to 13:56) and B3's division `p` 11 on stone from 6d808a5 (13:56 to
17:43); W2, B2 and B3's algebras on jebel from 393b3b1 (B2 08:04 to 11:24, W2 11:25 to 13:18, B3
13:19 to 16:18), crate afa7204. Each control repeated by bits without the flag on its run's commit
(`b2-bits-check/`, `b3-bits-check/`, `w2-bits-check/`; B1 by `controls/`; B3's `p` 11 only by B1's
check of L3's cell on the same commit, the p11 ladder itself not replayed). Every certified network
recertified on pyramidon in both verdicts (`certify/`). Data in
data/window-recurrent/window-branches/{b1,w2,b2,b3,b3-p11}/, manifests `b1.sha256` 977fec17...
(161 files), `w2.sha256` 4e433e2c... (135), `b2.sha256` a2b39a33... (472), `b3.sha256`
556122c0... (457), `b3-p11.sha256` 66bcd433... (176: stone's b3/ copied whole; its `certify/`,
`readout.txt` and `breaks/` added on pyramidon after, `b3-p11-readings.sha256` 36817b18..., 9);
each failure's best step read by `breaks` afterwards (`breaks/` in each, and in
division-ladder/l3/breaks/ for L3's five failures), manifest `stage6-breaks.sha256`
7d35a970... (23 files, run from window-recurrent/).

| claim | cell | control | result (Wilson 95%) | failures: pairs per code at best |
| --- | --- | --- | --- | --- |
| B1, branches at 16 | division `p` 13 | L3, 0 of 6 | 1 of 6 (0.03-0.56): seed 5, step 14 | 26-76, never near (worst answer 0.40-0.94) |
| W2, width 22 (size) | division `p` 13 | L3, 0 of 6 | 3 of 6 (0.19-0.81): seeds 0-2, steps 10-14 | 34-120 |
| B2, branches at 16 | word-problem-s5 | round 2, 0 of 6 | 1 of 6 (0.03-0.56): seed 4, step 17 | 36-120 (one at 1.65) |
| B2, branches at 16 | move-to-front-5 | round 2, 0 of 6 | 1 of 6 (0.03-0.56): seed 5, step 23 | no state in 3 (4-34 codes) |
| B3, branches at 16 | word-problem-s4 | round 1, 4 of 6 | **0 of 6 (0.00-0.39): harm**, p 0.030 | 2.0-23; 4 came near, 4 collapsed |
| B3, branches at 16 | move-to-front-4 | round 1, 4 of 6 | 3 of 6 (0.19-0.81): no harm | no state (2-16 codes) |
| B3, branches at 16 | division `p` 11 | the p11 ladder, 4 of 6 | 1 of 6 (0.03-0.56): seed 0, step 18; no harm | 9.3-88, never near (worst answer 0.39-0.67) |

- **B1 works, gain undecided; W2 works too** (3 of 6 against L3's 0, p 0.091): by the registered
  reading ("both work"), **size at B1's count suffices, and the separation's own part is
  undecided**. B1 did worse than W2 (1 against 3 of 6) with 5% fewer parameters. B1's certificate:
  408 and 397 codes reached, minimised to 169, 0 wrong outputs and states, snap ratio >= 1.000,
  logit lead >= 0.552. W2's three: 286-338 codes, logit lead >= 0.256-1.69.
- **Division at width 16 was mostly bottleneck 2, not 3.** L3's five other failures, read now at
  their best steps, stood at 20-135 pairs per code (1-7 right pairs); only seed 5, the one the
  readout probe read, had an exact state. B1's failures (26-76) and W2's (34-120) are the same kind.
  So the branches didn't move division from 3 to 2 (the one-code-one-state workplan's reading of
  B1 mid-run, corrected there): the narrow stream starved the transitions with or without them, and
  width (W2 3 of 6, W1 5 of 6) is what relieved it.
- **B2: works on both `n` 5 cells, gain undecided** (1 of 6 each against 0). Certificates: S5 seed
  4, 331 codes minimised to 120, logit lead >= 3.91; move-to-front-5 seed 5, 219-220 codes, logit
  lead >= 5.92. Its failures by bottleneck in the one-code-one-state workplan's "Why": S5's in 2,
  move-to-front-5's mostly in 1.
- **B3: harm on S4** (0 of 6 against round 1's 4 of 6, Fisher one-sided p 0.030). Four of its six
  came near (worst answer >= 0.95) and four collapsed; seed 3 reached 1.000 at step 8 with an exact
  core (2.0 pairs per code, its first codes breaking 0 of 120) and was lost. Move-to-front-4 3 of 6
  against 4: no harm; its failures formed no state. Division `p` 11 1 of 6 against the ladder's 4 of 6:
  no harm by the registered rule (p 0.12), but a fall; its five failures are bottleneck 2 (9.3-88
  pairs per code, 1-3 right pairs), as B1's at 13: the narrow branch starving the transitions.
  Seed 0's certificate: 169 codes reached, minimised to 121, 0 wrong outputs and states, snap
  ratio >= 1.002, logit lead >= 1.39.
- **The side probe (branches at 32 on S5 seeds 0 and 1; a probe, from 393b3b1 on jebel, 10:12 to
  12:09; `probe-s5-b32/`, manifest `probe-s5-b32.sha256` 638c64ed..., 68 files):** 0 of 2, pairs
  per code at the best steps 120 and 77 against B2's same seeds' 120 and 66: by its expectation,
  **width adds nothing on S5's confusion** (bottleneck 2).
- **Adoption (the workplan's rule): not adopted.** B1 and B2 work on cells whose controls failed,
  but B3 shows harm on S4, so the branches stay an option (`--branches`). What they changed:
  nothing width doesn't do on division; one seed each on the `n` 5 cells; harm on S4. The base
  window is width 32 (the one-code-one-state workplan's D4', C0 running).
Findings (the night of 2026-10-19 to 20: the thermostat's second probe, annealing, the commitment never
on, C0, C2 and R1; each registered or its expectation written in its script before its data; the
owner then abandoned the plans these served, 2026-10-20, so the stopped runs are read as they stand).
Data under data/window-recurrent/: commitment-control/thermostat-probe-2/ (manifest
`thermostat-probe-2.sha256` 88a1382d..., 240 files), anneal-probe/ (`anneal-probe.sha256` d36128ae...,
386; arm 2's in `stone/`), never-probe/ (`never-probe.sha256` ccb53e4c..., 244), never-r34/
(6cf4aa9a..., 90), never-harm/ (81a518fc..., 74), never-scorecard/ (11253ceb..., 98);
one-code-one-state/c0/ (`c0.sha256` f40f8b6a..., 973), c2/ (56cd3173..., 419), r1-probe/
(8acdb953..., 239), r1-probe-32/ (67e83466..., 224); each manifest run from its parent directory.
Every certificate below recertified on pyramidon in both verdicts (`certify/`, `certify.log`).

- **The thermostat's second probe (D2'', `--thermostat --thermostat-screen 0.95`; round 1's six
  failure-A runs; 1425fe9, pyramidon, 16:16 to 19:31): 0 of 6 certified; 4 of 6 formed a state**
  (best shares 0.62-0.74: S3 seeds 3, 4, 5, move-to-front-4 seed 3) **and every one eroded within 1-2
  steps of the commitment's return** (ends 0.01-0.10 on 3-5 codes). Not its registered success; the
  thermostat retired (the owner's choice, made before).
- **Annealing (`--anneal`, the commitment-control workplan's stage 3, D6-D11; #886):** arm 1 (the six
  runs at width 16; cf036ab, crate a3e7604, pyramidon, 19:32 to 23:49; control: round 1 repeated by
  bits, 12 rows) **0 of 6, 1 formed** (move-to-front-4 seed 3, 0.537): the factors hardened at steps
  2-5 before any state, the commitment ramped back, the runs ended as round 1's. Arm 2 (move-to-front-5
  seeds 0, 1 at width 32; 8e88c9d, stone, 18:26 to 22:51; control: C0 repeated by bits, 12 rows) **0 of
  2, none formed** (0.466, 0.389; frozen from step 6 on 15 and 9 codes). **Arm 2's codes merged:**
  23-27 of a factor's 32 codes identical (squared distance under 1e-6) by step 2-3, hardness stalled
  near 0.30 until `tau` 1/64: deterministic annealing above its critical temperature (Rose 1998; the
  literature review, 2026-10-20). Not a remedy as built. #886's tier-1 A/B (`epoch_op_profile`, ONNO
  then NO): profiled totals 2.0-7.4% slower on conv paths its crate diff doesn't touch; merged at the
  owner's word.
- **The commitment never on (`--commitment-from 1000`):** arm 1 (S3 seeds 0, 3, round 1's cell;
  8e88c9d, stone; control: the diagnostic repeated by bits, 14 rows) **both certified** (steps 6 and
  8; seed 3 lost its state at steps 5-6 and re-formed it); arm 2 (S3 seeds 1, 4, 5, move-to-front-4
  seed 3; 377e648, pyramidon; the diagnostic by bits, 12 rows) **S3 seed 5 and move-to-front-4 seed 3
  certified, S3 seeds 1 and 4 formed nothing** (0.31, 0.48; 3-4 codes): **4 of the 6 failure-A runs**,
  against at most 1 of 6 for the diagnostic and both thermostat probes. Their certificates are thin
  (2-3 near ties each under the strict ratio). Arm 3 (move-to-front-5 seeds 0, 1 at 32; stone) 0 of 2:
  states formed (0.86, 0.88, held 13-17 steps at 4.6-7 pairs per code) and lost. With R3/R4 (never-r34,
  the same seeds) 0 of 2 (0.90, 0.92 at best, 3.4 pairs per code). **Where the commitment worked, harm**
  (stopped at the redirection): S4 seeds 0, 1 at 32 uncertified at steps 22 and 30 (C0 certified them at
  11 and 12); division 13 seeds 0, 1 at 32 at 73-108 pairs per code by steps 11-12 (W1 certified both;
  control W1 repeated by bits, 8 rows); S5 seeds 0, 1 at 32 with R3/R4 at 18-20 (C2 certified seed 1).
  **So the commitment hurts before a state forms and helps once one exists** (Zhao et al. 2026's
  account); S3 seeds 1 and 4 collapse under the snap alone.
- **C0 (the one-code-one-state workplan's width claim; c452ed9, jebel, 16:24 to 23:11; controls by
  bits, 12-14 rows): S4 3 of 6, move-to-front-4 2 of 6, S5 0 of 6, move-to-front-5 0 of 6** (width 16:
  4, 4, 0, 0). No harm by its rule (harm only at 0 of 6): width 32 the base. Failures: S5 all
  bottleneck 2 (pairs per code about 120, share 0.88-0.98); move-to-front-5 five failure A, seed 2 a
  state at 1.6; move-to-front-4 seeds 2, 3, 4 failure A; S4 seeds 2, 5 at 20-23 pairs per code.
- **C2 (R3/R4 at width 32; c452ed9, jebel, 23:25 to 01:58; controls C0 by bits, 14 and 12 rows): S5 2
  of 6 (seeds 1, 4), move-to-front-5 3 of 6 (seeds 1, 2, 5)** against C0's 0 of 6 each: works, not
  helps (helps at 4). The failures came close (minimum pairs per code 1.0-1.9, S5 seed 0 5.4); move-to-
  front-5 seed 1, failure A in C0, certified. Two certificates thin (9 and 11 near ties).
- **R1 (the answers' own branch; 8824a70, stone): at 16, 0 of 6** (division at 23-44 pairs per code:
  the branch disturbs the state too); **at 32, 0 of 6**, both division seeds failing where W1 certified
  them. Shelved; C1 not run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import functools
import json
import math
import multiprocessing
import sys
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.specs import window_presets as wp

from indrajala_studies import chow, inference
from indrajala_studies import machine_reading as mr
from indrajala_studies import modular_family as mf
from indrajala_studies import window_diagnostics as wd
from indrajala_studies import window_recurrent_certificate as wc
from indrajala_studies import window_recurrent_study as wr
from indrajala_studies import window_tasks as wt
from indrajala_studies.collapse import RULES, SIGNALS, Trigger

GUARD_ARMS = {False: "unguarded", True: "guarded"}  # --guard both's directories
ARMS = {
    "family": mf.FAMILY,
    "quotient": mf.QUOTIENT,
    "sum": mf.SUM,
    "product": mf.PRODUCT,
    "state": mf.STATE,
    "suffix": mf.SUFFIX,
    "token": mf.TOKEN,
    "planted": mf.PLANTED,
    "cover": mf.COVER,
}
MACHINE_ARMS = ("planted", "cover")  # a machine's queries (wt.query_partitions); the rest are division's


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one probe run")
    run.add_argument(
        "--arm",
        choices=sorted(ARMS),
        default="family",
        help="division's queries, or a machine's (planted: a planted machine's approximation, T10's P2; cover: T11's coarsest cover, D20)",
    )
    seeds = run.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seed", type=int, help="one run, into --out")
    seeds.add_argument("--first-seed", type=int, help="--seeds runs from this seed, at once, into --out/seed<s>")
    seeds.add_argument(
        "--runs",
        nargs="+",
        metavar="TASK:SEED",
        help="these runs alone (a machine's task and a seed each), at once, into --out/seed<s> or, for several "
        "tasks, --out/<task>/seed<s>: chosen seeds replayed (not with --task)",
    )
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument(
        "--p", type=int, nargs="+", default=[5], help="the modulus; several run in one pool, into --out/p<p>"
    )
    run.add_argument(
        "--task",
        nargs="+",
        help="a general algebra's machines in place of division (word-problem-s<n>, move-to-front-<n>, "
        "random-<n>-<seed>); several run in one pool, into --out/<task>",
    )
    run.add_argument(
        "--codes-ratio",
        type=Fraction,
        help="each of --factors factors ceil(R sqrt(q)) codes for q target states, width --factor-width: "
        "ceil(R p) for division (the division ladder's D2, the general algebras' D3: 32/11)",
    )
    run.add_argument(
        "--factor-width",
        type=int,
        help="each --codes-ratio factor's width s (default 16; W1's and the paper's cell: 32)",
    )
    run.add_argument("--factors", type=int, default=2, help="factors of 16 codes, s 16 each")
    run.add_argument(
        "--factor",
        type=int,
        nargs=2,
        action="append",
        metavar=("S", "M"),
        help="a factor of width S, M codes (for --factors)",
    )
    run.add_argument(
        "--reads", choices=["before_state", "output"], default="before_state", help="the poolings' tokens (D10)"
    )
    run.add_argument(
        "--budget-minutes", type=float, help="a wall-clock runaway guard only (default none; its stop is censored)"
    )
    run.add_argument("--budget-epochs", type=int, required=True, help="epochs trained in all: the binding budget")
    run.add_argument("--workers", type=int, help="runs at once (default: every seed at once)")
    run.add_argument(
        "--fork", action="store_true", help="snapshot and fork on collapse (the collapse-fork workplan, retired)"
    )
    _trigger_arguments(run)
    run.add_argument(
        "--guard",
        choices=["off", "on", "both"],
        default="off",
        help="the certificate guard on the quotient, armed after a perfect quotient screen (the guard workplan); "
        "both: each seed with and without, into --out/unguarded and --out/guarded",
    )
    run.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.QUERIES),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--resume",
        action="store_true",
        help="the same command again: each run goes on from its checkpoint, an ended one is skipped, one not "
        "begun starts (the code-consolidation workplan's stage 3)",
    )
    run.add_argument(
        "--long-tail",
        type=int,
        metavar="F",
        help="B (the code-consolidation workplan's D4): every other example at lengths j+1..F j, capped at the "
        "exclusion bound",
    )
    run.add_argument(
        "--consolidate",
        action="store_true",
        help="A (the code-consolidation workplan's D1): at each diagnosis, the network's d-equivalent codes "
        "pulled to their representatives",
    )
    run.add_argument(
        "--restore-best",
        action="store_true",
        help="P2 (the code-consolidation workplan's stage 2c): a step ending on a plateau goes back to the network "
        "at its best screen before the step's end and the next step",
    )
    run.add_argument(
        "--commitment-from",
        type=int,
        metavar="J",
        help="the commitment-control workplan's diagnostic: every factor's commitment 0 for the steps before J, "
        "its specs' from J",
    )
    run.add_argument(
        "--window-width",
        type=int,
        default=wp.D,
        metavar="D",
        help="the window's residual stream (W11's 16), its FFN twice it (the window branches workplan's W1)",
    )
    run.add_argument(
        "--branches",
        action="store_true",
        help="the window branches workplan's window: a shared trunk, then a branch a factor and one for the answers "
        "(needs --reads output)",
    )
    run.add_argument(
        "--branch-heads",
        type=int,
        default=1,
        metavar="H",
        help="each of --branches' blocks' attention heads (1, the retired design's; the post-paper workplan's "
        "branches-32 has 2, D8)",
    )
    run.add_argument(
        "--recombine",
        action="store_true",
        help="the post-paper workplan's recombining window (D7): a two-block shared trunk, then a two-head branch a "
        "factor, their ends summed into one more block before the output (needs --reads output)",
    )
    run.add_argument(
        "--answer-branch",
        action="store_true",
        help="the one-code-one-state workplan's R1: a shared trunk, then the state branch (W11's last block) every "
        "factor reads and the answers' own single-head branch (needs --reads output)",
    )
    run.add_argument(
        "--baseline",
        choices=mf.BASELINES,
        help="the paper's T3: 'linear', E6's window without its blocks; 'continuous', every factor's state "
        "continuous, no codes (no certificate: judged by long tests)",
    )
    run.add_argument(
        "--thermostat",
        action="store_true",
        help="the commitment-control workplan's usage thermostat (stage 2): each factor's commitment off below "
        "--thermostat-floor effective codes, ramped back over --thermostat-ramp epochs once they reach "
        "--thermostat-restore of its codebook",
    )
    run.add_argument("--thermostat-floor", type=float, default=mf.Thermostat.floor, metavar="CODES")
    run.add_argument("--thermostat-restore", type=float, default=mf.Thermostat.restore, metavar="SHARE")
    run.add_argument("--thermostat-ramp", type=int, default=mf.Thermostat.ramp, metavar="EPOCHS")
    run.add_argument(
        "--thermostat-screen",
        type=float,
        metavar="SCREEN",
        help="D2'': restore a factor's commitment once the step's best screen reaches SCREEN, in place of half its codebook",
    )
    run.add_argument(
        "--anneal",
        action="store_true",
        help="the commitment-control workplan's annealing (stage 3): each snapped factor's codes soft in training at "
        "--anneal-start halved every --anneal-halving epochs, its commitment 0, until 99%% of the step's screen's "
        "windows weight one code above 0.99; then hard, its commitment ramped back over --anneal-ramp epochs",
    )
    run.add_argument("--anneal-start", type=float, default=mf.Anneal.start, metavar="TEMPERATURE")
    run.add_argument("--anneal-halving", type=int, default=mf.Anneal.halving, metavar="EPOCHS")
    run.add_argument("--anneal-ramp", type=int, default=mf.Anneal.ramp, metavar="EPOCHS")
    run.add_argument("--anneal-tail", type=int, default=mf.Anneal.tail, metavar="EPOCHS")
    run.add_argument(
        "--anneal-never-commit",
        action="store_true",
        help="the commitment stays 0 once hard (the state formation workplan's D23)",
    )
    run.add_argument(
        "--whole",
        action="store_true",
        help="no curriculum (the state formation workplan's D22): every epoch over every length to the cap, every "
        "step's screen over a fixed spread of them; a step a block of epochs ending in its certificate",
    )
    run.add_argument(
        "--crucible",
        action="store_true",
        help="the state formation workplan's D26 (with --warm-up): the material held at the warm-up's last step's "
        "lengths through the whole profile, then, every factor hard, the whole distribution's (a seed, then growth)",
    )
    run.add_argument(
        "--state-bound",
        type=int,
        help="the state formation workplan's D35: stop at the first step whose snapped machine answers right "
        "with at most this many states, minimised (the certificate only judging); else at the first certificate",
    )
    run.add_argument(
        "--margin-wait",
        type=int,
        help="the state formation workplan's D59 (with --state-bound): D35's stop waits until no near tie splits "
        "the snapped machine, at most this many steps after its first pass",
    )
    run.add_argument(
        "--cutoff-epochs",
        type=int,
        help="the batch runner workplan's D1 (with --continuous): an attempt not certified at both verdicts by "
        "the first step's end with this many epochs trained stops there, \"cut\" (restarts at Luby's cutoff "
        "t*, in epochs; a certified run goes on)",
    )
    run.add_argument(
        "--coupling-cap",
        type=float,
        default=1.0,
        help="the state formation workplan's D38: every factor's coupling capped (0: the codes never in the "
        "forward path)",
    )
    run.add_argument(
        "--continuous",
        action="store_true",
        help="the state formation workplan's D48 (with --crucible and --melt-shape exponential, no --handover or "
        "--melt-gate): every switch after the warm-up continuous: the cool's rate from the screen (D49), the "
        "coupling from the codes' sharpness (D50), soft to the end with a straight-through gradient (D51), growth "
        "from the crucible (D52), a step an epoch with its readings (D53)",
    )
    run.add_argument(
        "--coupling-pace",
        action="store_true",
        help="the state formation workplan's D61 (b) (with --continuous): each coupling moves toward its "
        "sharpness's at most an allowance an epoch, the allowance its first rise, then doubled while the screen "
        "holds and halved when the clock stops (times 2 ** (2 rate - 1))",
    )
    run.add_argument(
        "--reheat-slowing",
        action="store_true",
        help="the state formation workplan's D66 (with --reheat): the cool's forward rate halved after each "
        "reheat, the re-cool through the zone at half the pace of the pass before; a run that never reheats is "
        "unchanged",
    )
    run.add_argument(
        "--reheat",
        action="store_true",
        help="the state formation workplan's D63 (a) (with --continuous): the cool's rate extended below its low "
        "mark to -1, a fallen screen running the clock back (never past the cool's start), the temperatures it "
        "passes back over released; growth only forward",
    )
    run.add_argument(
        "--incremental-reheat",
        action="store_true",
        help="the incremental reheat workplan's D1-D7 (with --continuous, not --reheat): a hold lasting the cool's "
        "e-folding time runs the clock back along the cool's path, an epoch of clock an epoch, to just above the "
        "latest transition (a new most code groups in a factor), holds there as long, and undoes the stage "
        "before only if the screen hasn't recovered; none left, the run ends",
    )
    run.add_argument(
        "--sharpness-smoothing",
        action="store_true",
        help="the state formation workplan's D61 (c) (with --continuous): the coupling read from the sharpness "
        "smoothed by a one-sided Savitzky-Golay filter, its line over the cool's last e-folding time of clock "
        "(--melt-halving / ln 2)",
    )
    run.add_argument(
        "--warm-up",
        action="store_true",
        help="the state formation workplan's option 1 (with --anneal): every snapped factor continuous, its "
        "commitment 0, until the end of the first step whose best screen reaches --warm-up-screen or of the "
        "--warm-up-cap-th step, its distinct states at least --warm-up-distinct times its codes; then its codes by "
        "k-means of the states seen, heated to --melt-hot times the whole's critical temperature over --melt-heat "
        "epochs, held --melt-hold, cooled halving every --melt-halving, thermal noise of variance --melt-kappa T",
    )
    run.add_argument("--warm-up-screen", type=float, default=mf.WarmUp.screen, metavar="SHARE")
    run.add_argument("--warm-up-cap", type=int, default=mf.WarmUp.cap, metavar="STEPS")
    run.add_argument("--warm-up-distinct", type=float, default=mf.WarmUp.distinct, metavar="RATIO")
    run.add_argument("--melt-hot", type=float, default=mf.WarmUp.hot, metavar="MULTIPLE")
    run.add_argument("--melt-heat", type=int, default=mf.WarmUp.heat, metavar="EPOCHS")
    run.add_argument("--melt-hold", type=int, default=mf.WarmUp.hold, metavar="EPOCHS")
    run.add_argument("--melt-halving", type=int, default=mf.WarmUp.halving, metavar="EPOCHS")
    run.add_argument("--melt-kappa", type=float, default=mf.WarmUp.kappa, metavar="KAPPA")
    run.add_argument(
        "--melt-shape",
        choices=["exponential", "adaptive", *sorted(mf.COOLS)],
        default=mf.WarmUp.shape,
        help="the cool's shape on a log-temperature scale (D24): exponential halves every --melt-halving epochs; the "
        "others fall --melt-octaves in --melt-cool epochs, then halve as it; adaptive (D33) steps each factor's "
        "temperature once at equilibrium (--cool-within) or after --cool-settle epochs, by T exp(-pace / sqrt(C))",
    )
    run.add_argument("--cool-pace", type=float, default=mf.WarmUp.pace, metavar="LAMBDA")
    run.add_argument("--cool-settle", type=int, default=mf.WarmUp.settle, metavar="EPOCHS")
    run.add_argument("--cool-within", type=float, default=mf.WarmUp.within, metavar="SIGMAS")
    run.add_argument(
        "--cool-rise",
        type=float,
        default=mf.WarmUp.rise,
        metavar="SHARE",
        help="the adaptive cool's step no larger than raises a factor's coupling by SHARE (D34; 0: unbounded)",
    )
    run.add_argument("--melt-cool", type=int, default=mf.WarmUp.cool, metavar="EPOCHS")
    run.add_argument("--melt-octaves", type=float, default=mf.WarmUp.octaves, metavar="OCTAVES")
    run.add_argument(
        "--handover",
        action="store_true",
        help="the gradual handover (the state formation workplan's D25): each soft factor's next state (1 - l) h + "
        "l q, l 0 through the heat and the hold, then log-linear in T from its melting point to its last transition",
    )
    run.add_argument(
        "--melt-gate",
        type=float,
        default=mf.WarmUp.gate,
        metavar="SHARE",
        help="a quasi-static cool (the state formation workplan's D29): the profile's clock holds through an epoch "
        "whose screen before it is below SHARE times the hold's best (0: off)",
    )
    run.add_argument(
        "--handover-octaves",
        type=float,
        default=mf.WarmUp.window,
        metavar="OCTAVES",
        help="each factor's coupling window at least OCTAVES below its melting point (D30; 0: the curve's alone)",
    )
    run.add_argument(
        "--keep-machine",
        action="store_true",
        help="the one-code-one-state workplan's R3: keep each step's network with the fewest pairs per code (read at "
        "each diagnosis) and restore it at a plateau (not with --restore-best)",
    )
    run.add_argument(
        "--counterexamples",
        type=float,
        nargs="?",
        const=0.25,
        metavar="SHARE",
        help="the one-code-one-state workplan's R4: SHARE of each epoch's examples (0.25 if no SHARE) from the words "
        "reaching the codes that are wrong for their states, read at each diagnosis",
    )
    certify = commands.add_parser("certify", help="a saved family network's certificate, one slot")
    certify.add_argument("network", type=Path)
    certify.add_argument(
        "--task", required=True, help="modular-division-p (the quotient), modular-sum-p (the sum) or a machine's"
    )
    certify.add_argument("--arm", choices=sorted(ARMS), default="family")
    certify.add_argument("--snap-ratio", type=float, default=wc.SNAP_RATIO)
    certify.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.QUERIES),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    certify.add_argument("--out", type=Path, required=True)
    long = commands.add_parser("long-test", help="saved family networks' long tests, one slot")
    long.add_argument("networks", type=Path, nargs="+")
    long.add_argument(
        "--task", required=True, help="modular-division-p (the quotient), modular-sum-p (the sum) or a machine's"
    )
    long.add_argument("--arm", choices=sorted(ARMS), default="family")
    long.add_argument("--n", type=int, nargs="+", default=list(wr.LONG_LENGTHS))
    long.add_argument("--properties", nargs="+", default=list(mf.LONG_PROPERTIES))
    long.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.QUERIES),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    long.add_argument("--out", type=Path, required=True)
    trace = commands.add_parser("trace", help="runs' logs, a line per diagnosis, collapses marked")
    trace.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    progress = commands.add_parser("progress", help="runs' logs, running or ended, a line per run: how far each got")
    progress.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    population = commands.add_parser(
        "population", help="seeded crystallisations' logs (D26, D27), a line per run: early observables, the outcome"
    )
    population.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    stop = commands.add_parser(
        "answers-stop",
        help="runs' logs, a line per run: an answers-only stop under a state bound, judged by the certificate",
    )
    stop.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    stop.add_argument("--task", help="a machine's name: the bound its states, and the encoding for --snapped")
    stop.add_argument("--bound", type=int, help="the state bound (division's: its p squared); else --task's states")
    stop.add_argument("--slack", type=float, default=1.0, help="the bound times this (a loose bound)")
    stop.add_argument(
        "--snapped", action="store_true", help="re-read answers on each step's saved network (a coupled run's log)"
    )
    stop.add_argument(
        "--logged",
        action="store_true",
        help="each step's snapped answers as its row logged them (answers_stop, D53: every epoch of a continuous run)",
    )
    stop.add_argument("--arm", choices=sorted(ARMS), default="family", help="the run's, for the strict verdict")
    stop.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.QUERIES),
        default=list(mf.VOCABULARY),
        help="the run's, for the strict verdict (a row with none: its saved step network's, with --task)",
    )
    proof = commands.add_parser(
        "answers-proof",
        help="runs' logs: the answers-only stop under each bound, then Chow's test set on the stop's saved network",
    )
    proof.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl, its step networks beside it")
    proof.add_argument("--task", required=True, help="modular-division-p (the quotient) or a machine's")
    proof.add_argument("--arm", choices=sorted(ARMS), default="family")
    proof.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.QUERIES),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    proof.add_argument("--slack", type=float, nargs="+", default=[1.0, 1.1], help="each bound: the states times this")
    proof.add_argument(
        "--limit", type=int, default=10_000_000, help="the most test words to query; above, out of reach"
    )
    proof.add_argument("--out", type=Path, help="a directory for each run's <run>.answers-proof.json")
    capacity = commands.add_parser(
        "heat-capacity", help="profiled runs' logs (D32), a line per run: each factor's splits and capacity peak"
    )
    capacity.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    formation = commands.add_parser(
        "formation",
        help="runs' logs, a line per run: when the state formed and was kept, the cost by a kill's step, the "
        "first melting curve's crowding (the scaling conjecture's reading)",
    )
    formation.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    formation.add_argument("--by", type=int, default=200, help="the kill's step: seconds spent by it")
    cool = commands.add_parser(
        "cool",
        help="runs' logs, a line per run: the coupling's jumps against the screen's fall (their order), the "
        "longest freeze, the reheats",
    )
    cool.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    stages = commands.add_parser(
        "transitions",
        help="runs' logs, a line per run: the cool's stages of crystallisation (a factor's most code groups), "
        "its holds and their recoveries, its reheats and the stages they crossed (the incremental reheat's stage 0)",
    )
    stages.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    prefix = commands.add_parser(
        "same-prefix",
        help="a paired arm's logs against their controls', a line per pair: the rows identical, the cool's start, "
        "the first difference",
    )
    prefix.add_argument("pairs", type=Path, nargs="+", help="LOG CONTROL, LOG CONTROL, ...: each run's log.jsonl")
    prefix.add_argument("--ignore", nargs="*", default=[], help="the arm's own anneal fields (D61's target ...)")
    restarts = commands.add_parser(
        "restart-cost",
        help="runs' logs of one backdrop: Luby's expected cost per certificate of fresh-seed restarts at a "
        "fixed cutoff, the optimal cutoff t* (D63 (e)'s window, the 2026-10-22 literature review)",
    )
    restarts.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    restarts.add_argument(
        "--clock",
        choices=("epochs", "steps", "seconds"),
        default="epochs",
        help="the cost's unit: epochs trained (the time axis), steps, or seconds (money, not time)",
    )
    restarts.add_argument("--curve", action="store_true", help="every cutoff's q(t) and l(t), a line each")
    order = commands.add_parser(
        "order",
        help="machines' order from answers alone: the answer's entropy given the last k tokens against random "
        "machines of the state bound's size (the order note)",
    )
    order.add_argument("tasks", nargs="+", help="machines' names")
    order.add_argument("--bound", type=int, help="the state bound (default: each machine's states)")
    order.add_argument("--ensemble", type=int, default=30, help="random machines drawn")
    drops = commands.add_parser("drops", help="a collapse trigger scored against runs' logs, offline")
    drops.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl (runs made without the fork)")
    _trigger_arguments(drops)
    epoch = commands.add_parser("guard-epoch", help="one epoch from a saved network, unguarded and guarded, timed")
    epoch.add_argument("network", type=Path)
    target = epoch.add_mutually_exclusive_group(required=True)
    target.add_argument("--p", type=int)
    target.add_argument("--task", help="a general algebra's machine")
    epoch.add_argument("--step", type=int, required=True, help="the step whose lengths the epoch trains (1..step)")
    epoch.add_argument("--arm", choices=sorted(ARMS), default="family")
    separable = commands.add_parser(
        "separability", help="whether a saved network's answers are linearly separable over its output layer's input"
    )
    separable.add_argument("network", type=Path)
    separable.add_argument(
        "--task", required=True, help="modular-division-p (the quotient), modular-sum-p (the sum) or a machine's"
    )
    separable.add_argument("--arm", choices=sorted(ARMS), default="family")
    separable.add_argument("--steps", type=int, default=60_000, help="the softmax fit's full-batch steps")
    separable.add_argument("--passes", type=int, default=20_000, help="the perceptron's passes, at most")
    separable.add_argument("--out", type=Path, required=True, help="the result's JSON")
    tune = commands.add_parser("readout-tune", help="a saved network's output layer trained alone, to its certificate")
    tune.add_argument("network", type=Path)
    tune.add_argument(
        "--task", required=True, help="modular-division-p (the quotient), modular-sum-p (the sum) or a machine's"
    )
    tune.add_argument("--arm", choices=sorted(ARMS), default="family")
    tune.add_argument("--step", type=int, required=True, help="the step whose lengths it trains (1..step)")
    tune.add_argument("--epochs", type=int, default=30, help="at most, stopping at the first certified")
    tune.add_argument("--out", type=Path, required=True)
    for name, summary in (
        ("by-length", "saved networks' exact accuracy over every word of each length, against the task's machine"),
        ("breaks", "saved networks' broken transitions, from each state's first code and its extra codes"),
        (
            "approximations",
            "what each factor's code determines of the task's machine exactly, and whether they cover it",
        ),
        (
            "chow",
            "saved networks' machines against the task's by Chow's test set under a state bound (answers-only, exact)",
        ),
    ):
        reading = commands.add_parser(name, help=summary)
        reading.add_argument("networks", type=Path, nargs="+")
        reading.add_argument("--task", required=True, help="modular-division-p (the quotient) or a machine's")
        reading.add_argument("--arm", choices=sorted(ARMS), default="family")
        reading.add_argument(
            "--vocabulary",
            nargs="+",
            choices=list(mf.QUERIES),
            default=list(mf.VOCABULARY),
            help="the operations with a token (the approximation claim's: sum difference product)",
        )
        reading.add_argument("--d", type=int, help="a right code's word length (default the task's distinguishing + 1)")
        reading.add_argument("--out", type=Path, help="a directory for each network's <network>.<command>.json")
        if name == "by-length":
            reading.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
        if name == "chow":
            reading.add_argument("--bound", type=int, help="the state bound (default the task's states)")
            reading.add_argument("--slack", type=float, default=1.0, help="the bound times this, rounded down")
            reading.add_argument("--limit", type=int, default=10_000_000, help="the most test words to query")
    shares = commands.add_parser(
        "factor-shares", help="each factor's share of each of a machine's named congruences, weighted as training draws"
    )
    shares.add_argument("networks", type=Path, nargs="+")
    shares.add_argument("--task", required=True, help="a machine's (word-problem-s3, move-to-front-4, ...)")
    shares.add_argument("--show", type=float, default=0.8, help="a congruence on the line at this share or more")
    shares.add_argument("--out", type=Path, help="a directory for each network's <network>.factor-shares.json")
    spread = commands.add_parser(
        "state-spread",
        help="saved networks' state spread: the outputs' critical temperature and effective dimension, the codes' groups",
    )
    spread.add_argument("networks", type=Path, nargs="+")
    spread.add_argument("--task", required=True, help="modular-division-p (the quotient) or a machine's")
    spread.add_argument("--arm", choices=sorted(ARMS), default="family")
    spread.add_argument("--step", type=int, required=True, help="the curriculum step whose screen lengths it samples")
    spread.add_argument("--cases", type=int, default=200, help="cases a length (the screen's)")
    norms = commands.add_parser(
        "weight-norms",
        help="saved networks' weight norms, a line each: the whole, the readout, each factor's cell and codes "
        "(plasticity's correlates and the slingshot, offline)",
    )
    norms.add_argument("networks", type=Path, nargs="+")
    melting = commands.add_parser(
        "melting-curve",
        help="saved networks' melting curves (D21): Rose's deterministic annealing of each factor's codes on its "
        "outputs, cooled from twice the critical temperature, the codes' groups and transitions at each temperature",
    )
    melting.add_argument("networks", type=Path, nargs="+")
    melting.add_argument("--task", required=True, help="modular-division-p (the quotient) or a machine's")
    melting.add_argument("--arm", choices=sorted(ARMS), default="family")
    melting.add_argument("--step", type=int, required=True, help="the curriculum step whose screen lengths it samples")
    melting.add_argument("--cases", type=int, default=200, help="cases a length (the screen's)")
    separation = commands.add_parser(
        "state-separation",
        help="whether saved networks' cell outputs determine the target machine's state (nearest neighbour and "
        "centroid accuracy against chance; the state formation workplan's feasibility check)",
    )
    separation.add_argument("networks", type=Path, nargs="+")
    separation.add_argument("--task", required=True, help="a machine's task")
    separation.add_argument("--arm", choices=sorted(ARMS), default="family")
    separation.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    separation.add_argument("--cases", type=int, default=50, help="words a length")
    separation.add_argument(
        "--live", action="store_true", help="every snapped factor read continuous (a warm-up's network, as it trained)"
    )
    congruences = commands.add_parser(
        "congruences", help="each task's machine: its nontrivial principal congruences (its approximations' atoms)"
    )
    congruences.add_argument("tasks", nargs="+", help="task names (wt.task_named)")
    cover = commands.add_parser(
        "cover-pair",
        help="each task's machine: its coarsest cover by two approximations (the paper workplan's D20 rule, "
        "wt.coarsest_cover; minutes at move-to-front-4), against the registered pair where there is one",
    )
    cover.add_argument("tasks", nargs="+", help="task names (wt.task_named)")
    bars = commands.add_parser(
        "rank-bars",
        help="a machine-level comparison's attainable p values (the paper workplan's D19): the exact rank-sum "
        "p for each count of machines succeeding in the tested arm, against each count in the control",
    )
    bars.add_argument("--machines", type=int, default=15, help="machines an arm")
    bars.add_argument("--control", type=int, nargs="+", default=[0, 1, 2, 3], help="control machines succeeding")
    measures = commands.add_parser(
        "measures",
        help="each task's machine measured before any run (the paper workplan's D19): congruences, order, "
        "Chow's k, the crucible depth, token ranks, synchronising",
    )
    measures.add_argument("tasks", nargs="+", help="task names (wt.task_named)")
    measures.add_argument("--ensemble", type=int, default=30, help="random machines drawn for order")
    summary = commands.add_parser("long-summary", help="a study's long tests: a line per run, failures, all")
    summary.add_argument("root", type=Path, help="long-test's outputs, a <run>/<network>.long-test.json each")
    readout = commands.add_parser("readout", help="a registered comparison: each arm's runs, counts, Fisher's test")
    readout.add_argument("root", type=Path, help="the study's directory, a <arm>/seed<s> per run")
    readout.add_argument("--arms", nargs="+", default=["family", "quotient"], help="the first is tested as better")
    readout.add_argument("--certify", type=Path, help="the recertifications (default: root/certify)")
    readout.add_argument(
        "--long-tests",
        type=Path,
        help="long-test's outputs (<run>/<network>.long-test.json): count a run only certified and long-tested "
        "(a run without codes: its last network long-tested)",
    )
    readout.add_argument(
        "--pairs",
        nargs="+",
        metavar="BETTER:OTHER",
        help="the registered tests, each arm against another (Holm over them); the arms are theirs, in order",
    )
    args = parser.parse_args(argv)
    if args.command == "drops":
        trigger = _trigger(args)
        print(f"trigger: {trigger.describe()}")
        scores = [
            mf.score_trigger([json.loads(line) for line in path.read_text().splitlines()], trigger)
            for path in args.logs
        ]
        for path, score in zip(args.logs, scores, strict=True):
            print(
                f"{path}: {score.points} points, {score.drops} large drops, {score.caught} caught, "
                f"{score.unrecovered_caught}/{score.unrecovered} unrecovered caught, {score.false_alarms} false alarms, "
                f"{score.early} early"
            )
        print(
            f"{len(scores)} runs: {sum(s.drops for s in scores)} large drops, {sum(s.caught for s in scores)} caught; "
            f"{sum(s.unrecovered_caught for s in scores)} of {sum(s.unrecovered for s in scores)} never recovered "
            f"caught; {sum(s.false_alarms for s in scores)} false alarms near a solution, "
            f"{sum(s.early for s in scores)} firings while still learning"
        )
        return 0
    if args.command == "guard-epoch":
        encoding = _target(args.task or f"modular-division-{args.p}", args)
        print(json.dumps(mf.guard_epoch(args.network, encoding, args.step)))
        return 0
    if args.command == "separability":
        result = mf.separability(args.network, _target(args.task, args), args.task, args.steps, args.passes)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({key: value for key, value in result.items() if key != "wrong"}))
        return 0
    if args.command == "readout-tune":
        target = _target(args.task, args)
        for row in mf.tune_readout(args.network, target, args.task, args.step, args.out, args.epochs):
            print(json.dumps(row), flush=True)
        return 0
    if args.command == "readout":
        pairs = [(better, other) for better, other in (pair.split(":") for pair in args.pairs)] if args.pairs else None
        arms = list(dict.fromkeys(arm for pair in pairs for arm in pair)) if pairs else args.arms
        print("\n".join(mf.readout(args.root, arms, args.certify or args.root / "certify", pairs, args.long_tests)))
        return 0
    if args.command == "congruences":
        for name in args.tasks:
            print(congruence_line(name), flush=True)
        return 0
    if args.command == "cover-pair":
        for name in args.tasks:
            print(cover_line(name), flush=True)
        return 0
    if args.command == "measures":
        for name in args.tasks:
            print(measures_line(name, args.ensemble), flush=True)
        return 0
    if args.command == "rank-bars":
        print("\n".join(rank_bars(args.machines, args.control)))
        return 0
    if args.command == "long-summary":
        print("\n".join(mf.long_summary(args.root)))
        return 0
    if args.command == "state-spread":
        encoding = _target(args.task, args)
        lengths = sorted({1, (args.step + 1) // 2, args.step})
        rows = [mf.windows_drawn(encoding, n, args.cases, f"family-screen-{args.step}-{n}") for n in lengths]
        for path in args.networks:
            spread = mf.state_spread(mf.load_network(str(path)), rows)
            print(json.dumps({"network": str(path), "step": args.step, "factors": spread}), flush=True)
        return 0
    if args.command == "state-separation":
        encoding = _target(args.task, args)
        rows = [mf.windows_drawn(encoding, n, args.cases, f"separation-{n}") for n in args.lengths]
        for path in args.networks:
            net = mf.load_network(str(path))
            if args.live:
                net.set_continuous([f.codes is not None for f in net.recurrence.factors])
            separation = mf.state_separation(net, rows, encoding.task.machine)
            print(json.dumps({"network": str(path), "live": args.live, **separation}), flush=True)
        return 0
    if args.command == "melting-curve":
        encoding = _target(args.task, args)
        lengths = sorted({1, (args.step + 1) // 2, args.step})
        rows = [mf.windows_drawn(encoding, n, args.cases, f"family-screen-{args.step}-{n}") for n in lengths]
        for path in args.networks:
            curves = mf.melting_curves(mf.load_network(str(path)), rows)
            print(json.dumps({"network": str(path), "step": args.step, "factors": curves}), flush=True)
        return 0
    if args.command == "factor-shares":
        task = wt.task_named(args.task)
        for path in args.networks:
            shares = wd.congruence_shares(wd.samples(mf.load_network(str(path)), task), task.machine)
            held: list[str] = []
            for g, factor in enumerate(shares["factors"]):
                factor_named: dict[str, float] = factor["named"]
                ranked = sorted(factor_named.items(), key=lambda kv: -kv[1])
                shown = ", ".join(f"{k} {v:.2f}" for k, v in ranked if v >= args.show)
                held.append(f"factor {g}: state {factor['state']:.2f}, {shown or '-'}")
            print(f"{path}: state {shares['state']:.2f}; {'; '.join(held)}", flush=True)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                result = {"network": str(path), "task": args.task, **shares}
                (args.out / f"{path.parent.name}.{path.stem}.factor-shares.json").write_text(
                    json.dumps(result, indent=1)
                )
        return 0
    if args.command in ("by-length", "breaks", "approximations", "chow"):
        for path in args.networks:
            print(read_network(path, args), flush=True)
        return 0
    if args.command == "trace":
        for path in args.logs:
            print(f"== {path}")
            print("\n".join(mf.trace([json.loads(line) for line in path.read_text().splitlines()])))
        return 0
    if args.command == "progress":
        for path in args.logs:
            print(f"{path}: {mf.progress([json.loads(line) for line in path.read_text().splitlines()])}", flush=True)
        return 0
    if args.command == "population":
        for path in args.logs:
            checkpoint = path.parent / "checkpoint.json"  # its run record, for the trained length (D42)
            run = json.loads(checkpoint.read_text())["run"] if checkpoint.exists() else None
            course = mf.crucible_course([json.loads(line) for line in path.read_text().splitlines()], run=run)
            print(f"{path}: {mf.crucible_summary(course)}", flush=True)
        return 0
    if args.command == "answers-stop":
        encoding = None if args.task is None else mf.target_named(args.task)
        if args.bound is not None:
            bound = args.bound * args.slack
        elif encoding is not None:
            bound = encoding.task.machine.states * args.slack
        else:
            raise SystemExit("answers-stop: --bound or a machine's --task")
        if args.snapped and encoding is None:
            raise SystemExit("answers-stop: --snapped needs --task")
        tally: dict[str, int] = {}
        for path in args.logs:

            def snapped(step: int, path: Path = path) -> dict[str, float] | None:
                saved = path.parent / f"step-{step}.network.json"
                if encoding is None or not saved.exists():
                    return None
                return mf.snapped_answers(mf.load_network(str(saved)), encoding)

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            logged = {
                int(row["step"]): row["answers_stop"].get("answers")
                for row in rows
                if "end" in row and isinstance(row.get("answers_stop"), dict)
            }
            reader = logged.get if args.logged else snapped if args.snapped else None
            strict = None if args.task is None else strict_reader(path, args.task, args)
            verdict = mf.answers_stop(rows, bound, reader, strict)
            tally[verdict["verdict"]] = tally.get(verdict["verdict"], 0) + 1
            print(f"{path}: {verdict['verdict']}, stop {verdict['stop']}, certified {verdict['certified']}", flush=True)
        print(f"bound {bound:g}: {tally}")
        return 0
    if args.command == "answers-proof":
        tallies: dict[float, dict[str, int]] = {}
        for path in args.logs:
            result = answers_proof(path, args)
            for b in result["bounds"]:
                proof = b.get("chow")
                outcome = (
                    "-"
                    if proof is None
                    else "network missing"
                    if "missing" in proof
                    else "out of reach"
                    if "refused" in proof
                    else "proved"
                    if proof["equivalent"]
                    else "refuted"
                )
                tally = tallies.setdefault(b["slack"], {})
                key = f"{b['verdict']} {outcome}"
                tally[key] = tally.get(key, 0) + 1
                detail = "" if proof is None or "missing" in proof else f" ({chow_line(proof)})"
                print(
                    f"{path}: x{b['slack']:g} bound {b['bound']}: {b['verdict']}, stop {b['stop']}, "
                    f"certified {b['certified']}; {outcome}{detail}",
                    flush=True,
                )
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / f"{path.parent.name}.answers-proof.json").write_text(json.dumps(result, indent=1))
        for slack, tally in tallies.items():
            print(f"x{slack:g}: {tally}")
        return 0
    if args.command == "weight-norms":
        for path in args.networks:
            w = mf.weight_norms(mf.load_network(str(path)))
            cells = ", ".join(f"{c:.2f}" for c in w["cells"])
            codes = ", ".join("-" if c is None else f"{c:.2f}" for c in w["codes"])
            print(
                f"{path}: total {w['total']:.2f}, readout {w['readout']:.3f}, cells {cells}, codes {codes}", flush=True
            )
        return 0
    if args.command == "restart-cost":
        courses = [
            mf.restart_course([json.loads(line) for line in path.read_text().splitlines()], args.clock)
            for path in args.logs
        ]
        for path, course in zip(args.logs, courses, strict=True):
            if course is None:
                print(f"{path}: no step ends, left out", flush=True)
        cost = mf.restart_cost([c for c in courses if c is not None])
        if args.curve:
            for t, q, cost_t in cost["curve"]:
                print(f"t {t:g}: q {q:.3f}, l {cost_t:.1f}", flush=True)
        print(mf.restart_summary(cost, args.clock), flush=True)
        return 0
    if args.command == "cool":
        for path in args.logs:
            course = mf.cool_course([json.loads(line) for line in path.read_text().splitlines()])
            print(f"{path}: {mf.cool_summary(course)}", flush=True)
        return 0
    if args.command == "transitions":
        for path in args.logs:
            course = mf.transitions_course([json.loads(line) for line in path.read_text().splitlines()])
            print(f"{path}: {mf.transitions_summary(course)}", flush=True)
        return 0
    if args.command == "same-prefix":
        if len(args.pairs) % 2:
            raise SystemExit("same-prefix: give logs in pairs, LOG CONTROL")
        for path, control in zip(args.pairs[::2], args.pairs[1::2], strict=True):
            reading = mf.same_prefix(
                [json.loads(line) for line in path.read_text().splitlines()],
                [json.loads(line) for line in control.read_text().splitlines()],
                args.ignore,
            )
            verdict = "intact" if reading["intact"] else "BROKEN"
            print(
                f"{path} against {control}: {verdict}, {reading['same']} rows identical, the cool from row "
                f"{reading['before_cool']}, first difference at step {reading['differs']}",
                flush=True,
            )
        return 0
    if args.command == "formation":
        for path in args.logs:
            course = mf.formation_course([json.loads(line) for line in path.read_text().splitlines()], args.by)
            print(f"{path}: {mf.formation_summary(course)}", flush=True)
        return 0
    if args.command == "order":
        for name in args.tasks:
            machine = mf.target_named(name).task.machine
            reading = mf.order_against_random(machine, args.bound or machine.states, args.ensemble)
            print(f"{name}: {json.dumps(reading)}", flush=True)
        return 0
    if args.command == "heat-capacity":
        for path in args.logs:
            courses = mf.heat_course([json.loads(line) for line in path.read_text().splitlines()])
            print(f"{path}: {mf.heat_summary(courses)}", flush=True)
        return 0
    if args.command == "long-test":
        args.out.mkdir(parents=True, exist_ok=True)
        for path in args.networks:
            print(long_test_file(path, _target(args.task, args), args), flush=True)
        return 0
    if args.command == "certify":
        result = mf.certify_file(args.network, _target(args.task, args), args.task, args.snap_ratio)
        args.out.write_text(json.dumps(result, indent=1))
        print(wc.summary(result))
        return 0
    if args.codes_ratio is not None and args.factor:
        parser.error("--codes-ratio sets every factor's codes: not with --factor")
    if args.factor_width is not None and args.codes_ratio is None:
        parser.error("--factor-width sets --codes-ratio's factors' width: only with --codes-ratio")
    if (args.branches or args.answer_branch or args.recombine) and args.reads != "output":
        parser.error("--branches, --answer-branch and --recombine: each factor reads a branch's end, so --reads output")
    if sum((args.branches, args.answer_branch, args.recombine)) > 1:
        parser.error("--branches, --answer-branch or --recombine, one at most")
    if args.branch_heads != 1 and not args.branches:
        parser.error("--branch-heads: the heads of --branches' blocks")
    if args.keep_machine and args.restore_best:
        parser.error("--keep-machine or --restore-best, not both")
    if args.thermostat and (args.commitment_from is not None or args.consolidate):
        parser.error("--thermostat sets the commitment itself: not with --commitment-from or --consolidate")
    if args.anneal and (args.thermostat or args.commitment_from is not None or args.consolidate):
        parser.error("--anneal sets the commitment itself: not with --thermostat, --commitment-from or --consolidate")
    anneal = (
        mf.Anneal(
            args.anneal_start,
            args.anneal_halving,
            mf.Anneal.hard,
            mf.Anneal.share,
            args.anneal_ramp,
            args.anneal_tail,
            not args.anneal_never_commit,
        )
        if args.anneal
        else None
    )
    if args.continuous and not (
        args.warm_up and args.crucible and args.melt_shape == "exponential" and not args.handover and not args.melt_gate
    ):
        parser.error("--continuous (D48) is a crucible's exponential cool: no --handover or --melt-gate of its own")
    if (args.coupling_pace or args.sharpness_smoothing or args.reheat) and not args.continuous:
        parser.error(
            "--coupling-pace, --sharpness-smoothing (D61) and --reheat (D63) set the continuous profile: with --continuous"
        )
    if args.cutoff_epochs is not None and (not args.continuous or args.cutoff_epochs < 1):
        parser.error("--cutoff-epochs (the batch runner's D1) cuts a continuous run's attempt: with --continuous, >= 1")
    if args.incremental_reheat and (not args.continuous or args.reheat):
        parser.error("--incremental-reheat sets the continuous cool's reheat: with --continuous, not --reheat")
    if args.reheat_slowing and not args.reheat:
        parser.error("--reheat-slowing (D66) slows the re-cool after a reheat: with --reheat")
    if args.coupling_pace and args.coupling_cap < 1.0:
        parser.error("--coupling-pace (D61) paces the coupling D38's cap would bound: not both")
    if args.margin_wait is not None and args.state_bound is None:
        parser.error("--margin-wait (D59) waits on D35's stop: with --state-bound")
    if args.warm_up and not args.anneal:
        parser.error("--warm-up ends in an annealed snap: with --anneal")
    if args.warm_up and (args.fork or args.restore_best or args.keep_machine):
        parser.error("--warm-up crystallises: never with --fork, --restore-best or --keep-machine (D18)")
    warm_up = (
        mf.WarmUp(
            args.warm_up_screen,
            args.warm_up_cap,
            mf.WarmUp.iterations,
            args.warm_up_distinct,
            args.melt_hot,
            args.melt_heat,
            args.melt_hold,
            args.melt_halving,
            args.melt_kappa,
            args.melt_shape,
            args.melt_cool,
            args.melt_octaves,
            args.handover,
            args.melt_gate,
            args.handover_octaves,
            args.cool_pace,
            args.cool_settle,
            args.cool_within,
            args.cool_rise,
        )
        if args.warm_up
        else None
    )
    thermostat = (
        mf.Thermostat(args.thermostat_floor, args.thermostat_restore, args.thermostat_ramp, args.thermostat_screen)
        if args.thermostat
        else None
    )
    # the targets by name: the machines', or division's at each p, and each one's directory when several
    if args.runs is not None and args.task:
        parser.error("--runs names its tasks: not with --task")
    named: list[str] = args.runs or []
    chosen = [(task, int(seed)) for task, _, seed in (run.rpartition(":") for run in named)]
    if args.runs is not None:
        args.task = list(dict.fromkeys(task for task, _ in chosen))
    names = args.task or [f"modular-division-{p}" for p in args.p]
    folders = dict(zip(names, args.task or [f"p{p}" for p in args.p], strict=True))

    def cell(name: str) -> tuple[mf.Target, list[tuple[int, int]]]:
        encoding = _target(name, args)
        if args.codes_ratio is not None:
            return encoding, mf.scaled_factors(
                encoding.target_states, args.codes_ratio, args.factors, args.factor_width or 16
            )
        return encoding, [tuple(f) for f in args.factor] if args.factor else [(16, 16)] * args.factors

    def probe(name: str) -> functools.partial[None]:
        encoding, factors = cell(name)
        return functools.partial(
            mf.run,
            encoding,
            factors,
            budget_seconds=None if args.budget_minutes is None else args.budget_minutes * 60,
            reads=args.reads,
            budget_epochs=args.budget_epochs,
            fork=mf.Fork(trigger=_trigger(args)) if args.fork else None,
            resume=args.resume,
            long_tail=args.long_tail,
            consolidate=args.consolidate,
            restore_best=args.restore_best,
            commitment_from=args.commitment_from,
            width=args.window_width,
            branches=args.branches,
            answer_branch=args.answer_branch,
            recombine=args.recombine,
            branch_heads=args.branch_heads,
            baseline=args.baseline,
            thermostat=thermostat,
            keep_machine=args.keep_machine,
            counterexamples=args.counterexamples,
            anneal=anneal,
            warm_up=warm_up,
            whole=args.whole,
            crucible=args.crucible,
            state_bound=args.state_bound,
            coupling_cap=args.coupling_cap,
            continuous=(
                mf.Continuous(
                    pace=args.coupling_pace,
                    smooth=args.sharpness_smoothing,
                    reheat=args.reheat,
                    slowing=args.reheat_slowing,
                    incremental=args.incremental_reheat,
                )
                if args.continuous
                else None
            ),
            margin_wait=args.margin_wait,
            cutoff_epochs=args.cutoff_epochs,
        )

    for name in names:
        print(feasibility(name, *cell(name), args.long_tail, args.consolidate), flush=True)
    guards = {"off": [False], "on": [True], "both": [False, True]}[args.guard]
    with machine_lock.study():
        if args.seed is not None:
            if len(guards) > 1 or len(names) > 1:
                parser.error("--guard both and several --p or --task run --first-seed's seeds, not one --seed")
            probe(names[0])(seed=args.seed, out=args.out, guard=guards[0])
            return 0
        seeds = range(args.first_seed, args.first_seed + args.seeds) if args.runs is None else range(0)
        # each seed's targets and arms in turn, so a round's rungs and pairs end about together
        jobs = [(name, s, guard) for s in seeds for name in names for guard in guards]
        jobs += [(name, s, guard) for name, s in chosen for guard in guards]
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(
            min(args.workers or len(jobs), len(jobs)), mp_context=context
        ) as pool:
            runs: dict[concurrent.futures.Future[str], str] = {}
            for name, s, guard in jobs:
                out = args.out / folders[name] if len(names) > 1 else args.out
                out = out / GUARD_ARMS[guard] if len(guards) > 1 else out
                out.mkdir(parents=True, exist_ok=True)
                runs[pool.submit(logged_run, functools.partial(probe(name), guard=guard), s, out, args.resume)] = (
                    f"{out.relative_to(args.out)}/seed{s}" if out != args.out else f"seed {s}"
                )
            for done in concurrent.futures.as_completed(runs):
                print(f"{runs[done]}: {done.result()}", flush=True)
    return 0


def feasibility(
    name: str, encoding: mf.Target, factors: list[tuple[int, int]], long_tail: int | None, consolidate: bool
) -> str:
    """
    A cell's feasibility line against its config (experiment-design.md, section 1), refused when the
    joint codes can't hold the target's states: q, codes a factor, m, the cap (the exclusion bound,
    the longest training length), s0 (D3) and, with --consolidate, A's d (D2).
    """
    q = encoding.target_states
    m = math.prod(codes for _, codes in factors)
    assert m >= q, f"{name}: {m} joint codes can't hold {q} states"
    machine = encoding.task.machine
    line = (
        f"{name}: q {q}, codes {' x '.join(str(codes) for _, codes in factors)}, m {m} ({m / q:.1f} q), "
        f"cap {mf.exclusion_bound(q, factors)}, s0 {mf.first_length(machine)}"
    )
    if long_tail is not None:
        line += f", long tail to {long_tail} j"
    return line + (f", d {mr.word_length(machine)}" if consolidate else "")


def chow_reading(machine: mf.QueryMachine, target: wt.Mealy, bound: int, limit: int) -> dict[str, Any]:
    """A network's machine against the target by Chow's test set under `bound`: its codes, minimised states, the
    verdict, the words queried and a counterexample; or "refused" when the test set exceeds `limit`."""
    walked = mr.walk(machine, target.tokens)
    try:
        checked = chow.check(mr.mealy(walked), chow.machine_query(target), bound, limit=limit)
    except ValueError as refused:
        return {"codes": len(walked.codes), "bound": bound, "refused": str(refused)}
    return {
        "codes": len(walked.codes),
        "states": checked.states,
        "bound": bound,
        "equivalent": checked.equivalent,
        "words": checked.words,
        "counterexample": None if checked.counterexample is None else list(checked.counterexample),
    }


def chow_line(result: dict[str, Any]) -> str:
    """chow_reading's result, a line."""
    if "refused" in result:
        return f"bound {result['bound']}, {result['refused']}"
    counterexample = result["counterexample"]
    return (
        f"{result['codes']} codes, minimised {result['states']}, bound {result['bound']}: "
        f"{'equivalent' if result['equivalent'] else 'not equivalent'} after {result['words']:,} words"
        + ("" if counterexample is None else f", counterexample of {len(counterexample)} tokens")
    )


def strict_reader(log: Path, task: str, args: argparse.Namespace) -> Callable[[int], bool | None]:
    """
    A run's step's strict verdict (snap ratio 1.0, D59) from its saved network, for logs whose rows
    carry no `certificates` (answers_stop's `strict`); None where the step's network wasn't saved.
    """

    def strict(step: int) -> bool | None:
        saved = log.parent / f"step-{step}.network.json"
        if not saved.exists():
            return None
        machine = mf.QueryMachine(mf.load_network(str(saved)), _target(task, args), mf.slot(task))
        return wc.certify(machine, wt.task_named(task), snap_ratio=1.0).certified

    return strict


def answers_proof(log: Path, args: argparse.Namespace) -> dict[str, Any]:
    """
    T6's reading of one run (the paper workplan): at each bound (the target's states times a slack),
    the answers-only stop from the log (answers_stop: the logged answers all right, the minimised
    machine within the bound; the certificate only judges it), then Chow's test set on the stop's
    saved network under the same bound, the target a black box: the proof needing no target in the
    loop. "network missing" where the stop's step wasn't saved.
    """
    target = wt.task_named(args.task).machine
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    bounds: list[dict[str, Any]] = []
    for slack in args.slack:
        bound = int(target.states * slack)
        verdict = mf.answers_stop(rows, bound, strict=strict_reader(log, args.task, args))
        reading: dict[str, Any] = {"slack": slack, **verdict, "bound": bound}
        if verdict["stop"] is not None:
            saved = [log.parent / f"{kind}step-{verdict['stop']}.network.json" for kind in ("", "certified-")]
            found = next((s for s in saved if s.exists()), None)
            if found is None:
                reading["chow"] = {"bound": bound, "missing": str(saved[0])}
            else:
                machine = mf.QueryMachine(mf.load_network(str(found)), _target(args.task, args), mf.slot(args.task))
                reading["chow"] = {"network": str(found), **chow_reading(machine, target, bound, args.limit)}
        bounds.append(reading)
    return {"log": str(log), "task": args.task, "bounds": bounds}


def read_network(path: Path, args: argparse.Namespace) -> str:
    """One network's by-length or breaks (machine_reading), a line, and its JSON into --out."""
    target = wt.task_named(args.task).machine
    d = args.d or mr.word_length(target)
    machine = mf.QueryMachine(mf.load_network(str(path)), _target(args.task, args), mf.slot(args.task))
    walk = mr.product(machine, target)
    if args.command == "chow":
        bound = int((args.bound or target.states) * args.slack)
        result = {"network": str(path), "task": args.task, **chow_reading(machine, target, bound, args.limit)}
        line = f"{path}: {chow_line(result)}"
    elif args.command == "by-length":
        rows = mr.by_length(walk, sorted(args.lengths), d)
        result: dict[str, Any] = {"network": str(path), "task": args.task, "d": d, "lengths": rows}
        line = " ".join(f"{r['n']}:{r['exact']:.3f}/{r['named']:.3f}" for r in rows)
        line = f"{path}: d {d}, exact/named by length {line}"
    elif args.command == "approximations":
        result = {
            "network": str(path),
            "task": args.task,
            **mr.approximations(walk, target, wt.planted_labels(args.task)),
        }
        top = ", ".join(str(g) for g in result["cascade"]) or "-"
        line = (
            f"{path}: {result['states']} states, meet {result['meet']}, cover {result['cover']}, cascade top {top}; "
            + "; ".join(
                f"factor {g}: {f['codes']} codes, approximation {len(f['approximation'])} blocks "
                f"(congruence {f['congruence']}, autonomous {f['autonomous']:.3f}), determines {', '.join(f['determines']) or '-'}"
                for g, f in enumerate(result["factors"])
            )
        )
    else:
        result = {"network": str(path), "task": args.task, **mr.breaks(walk, d)}

        def share(kind: str) -> str:
            broken, transitions = result[kind]["broken"], result[kind]["transitions"]
            return f"{broken}/{transitions} ({broken / transitions:.2f})" if transitions else "0/0"

        line = (
            f"{path}: d {d}, {result['codes']} codes, {result['pairs']} pairs on {result['states']} states, "
            f"{result['right_pairs']} right; from right codes first {share('first_right')} extra "
            f"{share('extra_right')}; from every code first {share('first')} extra {share('extra')}"
        )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{path.parent.name}.{path.stem}.{args.command}.json").write_text(json.dumps(result, indent=1))
    return line


def _target(task: str, args: argparse.Namespace) -> mf.Target:
    """
    A task's target: division's or the modular sum's family encoding at its p (the arm's queries), else
    the machine's own, with a machine arm's queries (any other arm: none).
    """
    arm = getattr(args, "arm", "family")
    if task.startswith("modular-"):
        if arm in MACHINE_ARMS:
            raise SystemExit(f"--arm {arm} asks a machine's approximation: not {task}'s")
        p = int(task.rsplit("-", 1)[1])
        return mf.Encoding(p, ARMS[arm], tuple(getattr(args, "vocabulary", mf.VOCABULARY)))
    return mf.target_named(task, ARMS[arm] if arm in MACHINE_ARMS else ())


def _trigger_arguments(command: argparse.ArgumentParser) -> None:
    default = Trigger()
    command.add_argument("--trigger-signal", choices=SIGNALS, default=default.signal, help="the trigger's signal")
    command.add_argument(
        "--trigger-rule",
        choices=RULES,
        default=default.rule,
        help="ratio: below threshold x peak; fall: threshold below it",
    )
    command.add_argument("--trigger-threshold", type=float, default=default.threshold)
    command.add_argument(
        "--trigger-floor", type=float, default=default.floor, help="fire only once the step's peak is this"
    )


def _trigger(args: argparse.Namespace) -> Trigger:
    return Trigger(args.trigger_signal, args.trigger_rule, args.trigger_threshold, args.trigger_floor)


def logged_run(probe: functools.partial[None], seed: int, out: Path, resume: bool = False) -> str:
    """
    One of several runs, into out/seed<s>/, its output and errors to out/seed<s>.log (added to when
    resuming); how it ended.
    """
    mode = "a" if resume else "w"
    with (out / f"seed{seed}.log").open(mode) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        probe(seed=seed, out=out / f"seed{seed}")
    rows = [json.loads(line) for line in (out / f"seed{seed}" / "log.jsonl").read_text().splitlines()]
    last = next(r for r in reversed(rows) if "end" in r)
    verdict = "certified" if last["certified"] else last["end"]
    return f"step {last['step']}, {verdict}, {last['seconds'] / 60:.1f} min"


def cover_line(name: str) -> str:
    """
    A task's coarsest cover (wt.coarsest_cover): each approximation's blocks and quotient monoid, and
    whether it is the registered pair's (wt.query_partitions, up to the blocks' numbering).
    """
    machine = wt.task_named(name).machine
    pair = wt.coarsest_cover(machine)
    parts = ", ".join(f"{max(labels) + 1} blocks (monoid {machine.quotient_monoid(labels)})" for labels in pair)
    known = wt.query_partitions(name)
    if "cover-a" not in known:
        return f"{name}: {parts}; no registered pair"
    registered = tuple(wt.canonical(known[q]) for q in ("cover-a", "cover-b"))
    return f"{name}: {parts}; {'the registered pair' if registered == pair else 'NOT the registered pair'}"


def congruence_line(name: str) -> str:
    """
    A task's machine's principal congruences: how many, and their blocks (how many of each); none, said so:
    the machine is primitive (Rystsov and Szykuła 2023: its transition monoid has no nontrivial congruence).
    """
    machine = wt.task_named(name).machine
    found = machine.principal_congruences()
    if not found:
        return f"{name}: {machine.states} states, primitive: no nontrivial congruence (no approximation but itself)"
    blocks = Counter(max(labels) + 1 for labels in found)
    listed = ", ".join(f"{b} blocks x{n}" for b, n in sorted(blocks.items(), reverse=True))
    noun = "congruence" if len(found) == 1 else "congruences"
    return f"{name}: {machine.states} states, {len(found)} nontrivial principal {noun} ({listed})"


def rank_bars(machines: int, control: list[int]) -> list[str]:
    """
    For each count of control machines succeeding once, each count of tested machines succeeding once:
    the exact one-sided rank-sum p (inference.rank_sum_greater); a machine succeeding twice ranks above
    one succeeding once, so these are the bars when every success is a machine's only one.
    """
    lines: list[str] = []
    for c in control:
        ps = [
            inference.rank_sum_greater([1] * k + [0] * (machines - k), [1] * c + [0] * (machines - c))
            for k in range(machines + 1)
        ]
        listed = ", ".join(f"{k}: {p:.4f}" for k, p in enumerate(ps) if k > c and p < 0.2)
        lines.append(f"{machines} against {machines}, control {c} succeeding: {listed}")
    return lines


def measures_line(name: str, ensemble: int) -> str:
    """
    A task's machine measured before any run (D19): its congruences (congruence_line), its order (rms z
    against random machines of its size), its deepest state's depth plus Chow's k (the crucible depth:
    the feasibility register's row 15), each token's rank, and whether it synchronises.
    """
    machine = wt.task_named(name).machine
    order = mf.order_against_random(machine, machine.states, ensemble)["rms_z"]
    depth, k = machine.depth(), machine.distinguishing_length()
    crucible = "unbounded (not minimal)" if k is None else str(depth + k)
    ranks = ", ".join(str(rank) for rank in machine.token_ranks())
    resets = "synchronising" if machine.synchronising() else "not synchronising"
    return (
        f"{congruence_line(name)}; order {order}; depth {depth}, Chow's k {k}, crucible depth {crucible}; "
        f"token ranks {ranks}; {resets}"
    )


def long_test_file(path: Path, encoding: mf.Target, args: argparse.Namespace) -> str:
    """One network's long tests into args.out/<network>.long-test.json, unless there; a line of its cases and failures."""
    target = args.out / f"{path.stem}.long-test.json"
    if target.exists():
        return f"{path.stem}: done before"
    long = mf.long_test(path, encoding, args.task, args.n, args.properties)
    target.write_text(json.dumps(long, indent=1))
    cases, failures = wr.long_test_failures(long)
    seconds = sum(length["seconds"] for length in long["lengths"].values())
    return f"{path.stem}: {cases:,} cases, " + (", ".join(failures) or "every case passed") + f"; {seconds:.0f} s"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
