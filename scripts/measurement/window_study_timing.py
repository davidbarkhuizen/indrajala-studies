"""
The window-recurrent studies timed end to end on the Rust backend (the window-program workplan's
stages 0 and 6): the cases that run the studies' own code, beside indrajala-ml's
`scripts/measurement/window_recurrent_timing.py`, whose parsing, timing, breakdown and command line
this reuses (the research-separation workplan, D3). A case is a name, its kind first:

- `long-test-family-p<p>-f<m>-U1-n<n>-c<c>`: the long test end to end at n columns, c cases of U1
  through `QueryModel.predict` and `task.evaluate` (the examples, the network, the labels, the
  checking), the network built from seed 0 (the time doesn't depend on its answers).
- `study-family-p<p>-f<m>-e<e>`: a short study run end to end, `modular_family.run` of the family
  arm (two factors of m codes, the window's output read) from seed 0 for e epochs in all, into a
  temporary directory (a real run, not one step).

The learn and forward cases are window_recurrent_timing.py's; given here they run as there.

    python scripts/measurement/window_study_timing.py [--cases CASE ...] [--repeats 5]
    python scripts/measurement/window_study_timing.py --breakdown CASE [CASE ...]

It prints the probe contract's lines (ml's scripts/measurement/ab.py, `--bench cmd`). ab.py runs it
by its absolute path, so each side's ml tree (on PYTHONPATH) supplies the library and
window_recurrent_timing.py:

    python scripts/measurement/ab.py remote --host jebel run --bench cmd --old OLD --new NEW --order ONNO \\
        -- /home/david/code/indrajala-studies/scripts/measurement/window_study_timing.py
"""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import indrajala_ml

# window_recurrent_timing.py from the ml tree that supplies the library (scripts/ isn't a package)
sys.path.insert(0, str(Path(indrajala_ml.__path__[0]).parent / "scripts" / "measurement"))
import window_recurrent_timing as timing

DEFAULT_CASES = ["long-test-family-p7-f32-U1-n1024-c1023"]
FAMILY_QUERIES = ("sum", "difference")


def long_test_step(fields: dict[str, int], prop: str) -> Callable[[], object]:
    """A long test at n columns: c cases of the property, examples to checked results."""
    from indrajala_ml.model.specs.window_recurrence import OUTPUT

    from indrajala_studies import addition_data as ad
    from indrajala_studies import modular_family as mf
    from indrajala_studies import window_recurrent_study as wr
    from indrajala_studies import window_tasks as wt

    encoding = mf.Encoding(fields["p"], FAMILY_QUERIES)
    network = mf.network(encoding, [(timing.FAMILY_STATE, fields["f"])] * 2, 0, reads=OUTPUT)
    task = wt.task_named(f"modular-division-{fields['p']}")
    catalogue = task.catalogue(fields["c"], ad.RELATION_SAMPLED, ad.PER_STRATUM, doubling=True)
    tested = [p for p in wr.test_properties(task, catalogue) if p.id == prop]
    model = mf.QueryModel(network, encoding, task, mf.slot(task.name))
    n = fields["n"]
    return lambda: task.evaluate(model.predict(n), n, tested)


def study_step(fields: dict[str, int]) -> Callable[[], object]:
    """A short family probe run of e epochs, each run into its own temporary directory."""
    from indrajala_ml.model.specs.window_recurrence import OUTPUT

    from indrajala_studies import modular_family as mf

    encoding = mf.Encoding(fields["p"], FAMILY_QUERIES)
    factors = [(timing.FAMILY_STATE, fields["f"])] * 2

    def step() -> None:
        with tempfile.TemporaryDirectory() as out:
            mf.run(encoding, factors, 0, 1e9, Path(out), reads=OUTPUT, budget_epochs=fields["e"])

    return step


def case_step(case: str) -> tuple[Callable[[], object], bool]:
    """A case's step, and whether it is an inference step (one loop a repeat)."""
    kind, _, fields, prop = timing.parse_case(case)
    if kind == "study":
        return study_step(fields), True
    if kind == "long-test":
        assert prop is not None, f"{case}: a long test names its property"
        return long_test_step(fields, prop), True
    return timing.case_step(case)


if __name__ == "__main__":
    timing.main(__doc__, DEFAULT_CASES, case_step)
