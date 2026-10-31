"""scripts/paper/paper_tables.py: the paper's tables regenerated and checked against a manuscript."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "paper"))
import paper_tables as script  # scripts/ isn't a package

TEX = r"""
\begin{table}[t]
\caption{Another.}\label{tab:other}
\begin{tabular}{l}\toprule a \\ \midrule x & 1 \\ \bottomrule\end{tabular}
\end{table}
\begin{table}[t]
\caption{The one.}
\label{tab:toy}
\begin{tabular}{lr}
\toprule
arm & n \\
\midrule
first & 1 \\
second & 2 \\
\bottomrule
\end{tabular}
\end{table}
"""


def test_span_writes_ranges_as_the_manuscripts_do():
    assert script.span([]) == "--" and script.span([7]) == "7" and script.span([12, 6, 9]) == "6--12"


def rows(*lines: str) -> script.Table:
    return lambda data: list(lines)


def test_check_reads_the_labelled_tables_rows_and_flags_a_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    assert script.manuscript_rows(TEX, "tab:toy") == ["first & 1 \\\\", "second & 2 \\\\"]
    assert script.manuscript_rows(TEX, "tab:missing") is None
    tex = tmp_path / "paper.tex"
    tex.write_text(TEX)
    monkeypatch.setattr(script, "TABLES", {"tab:toy": rows("first & 1 \\\\", "second & 2 \\\\")})
    assert script.main(["check", str(tex)]) == 0
    monkeypatch.setattr(script, "TABLES", {"tab:toy": rows("first & 1 \\\\", "second & 3 \\\\")})
    assert script.main(["check", str(tex)]) == 1
    printed = capsys.readouterr().out
    assert "tab:toy: 2 rows, the same" in printed and "regenerated: second & 3" in printed


def test_median_is_whole_or_to_a_half():
    assert script.median([157, 182, 200, 394]) == "191" and script.median([152, 184, 207, 309]) == "195.5"
    assert script.median([95, 120, 230]) == "120"


def arm_run(epochs: int, end: str = "certified", certified: bool = True, long: bool | None = True, peak: float = 0.5):
    run = script.mf.Outcome(3, end, epochs, 1.0, peak, 2, 1 if peak < 0.5 else 0, 0, 0)
    return script.mf.ArmRun(300, run, certified, {}, True, long)


def test_rate_and_arm_tables_count_only_successes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runs = [
        arm_run(100),
        arm_run(300),
        arm_run(200, long=False),  # certified, a long test failed: a failure, but not an uncertified one
        arm_run(600, "spent", False, None, script.mf.NEAR),
        arm_run(600, "spent", False, None, 0.4),
    ]

    def batch_runs(data: Path, test: str, arm: str) -> list[script.mf.ArmRun]:
        return runs

    monkeypatch.setattr(script, "batch_runs", batch_runs)
    assert script.rates_table([("s4", 24)])(tmp_path) == [
        "\\texttt{s4} & 24 & 2 of 5 & 0.12--0.77 & 100--300, 200 & 1 of 3 \\\\"
    ]
    assert script.arms_table("t1", [("sum", "sum")])(tmp_path) == ["sum & 2 of 5 & 0.12--0.77 & 1 of 3 & 1 \\\\"]


def test_the_groups_table_tests_each_algebra_against_its_answers_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    given = [arm_run(100)] * 16 + [arm_run(600, "spent", False, None, 0.4)] * 4
    alone = [arm_run(100)] * 4 + [arm_run(600, "spent", False, None, 0.4)] * 16

    def batch_runs(data: Path, test: str, arm: str) -> list[script.mf.ArmRun]:
        return given if test == "t11/batch" else alone

    monkeypatch.setattr(script, "batch_runs", batch_runs)
    monkeypatch.setattr(script, "GROUPS", [("s4", 24)])

    def paper_family(data: Path) -> dict[str, tuple[float, float]]:
        return {"t11-s4": (1.8e-4, 0.0014)}  # Holm the family's

    monkeypatch.setattr(script, "paper_family", paper_family)
    assert script.groups(tmp_path) == [
        "\\texttt{s4} & 24 & 4 of 20 & 16 of 20 & 0.58--0.92 & $1.8 \\times 10^{-4}$ & $0.0014$ \\\\"
    ]


def test_the_paper_family_adjusts_over_every_registered_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def wins(data: Path, test: str, arm: str) -> tuple[int, list[script.mf.ArmRun]]:
        return 0, []

    def greater(first: object, second: object) -> float:
        return 0.01

    def machine_runs(data: Path, test: str, arms: list[str]) -> dict[str, dict[int, list[script.mf.ArmRun]]]:
        return {arm: {} for arm in arms}

    def machine_successes(runs: dict[int, list[script.mf.ArmRun]]) -> list[int]:
        return []

    def rank_sum_greater(first: list[int], second: list[int]) -> float:
        return 0.02

    def state_test(data: Path) -> tuple[list[list[script.mf.ArmRun]], float]:
        return [], 0.08

    def group_tests(data: Path) -> list[tuple[str, int, list[script.mf.ArmRun], list[script.mf.ArmRun], float]]:
        return [(b, q, [], [], 0.001) for b, q in script.GROUPS]

    for name, stub in [("wins", wins), ("greater", greater), ("machine_runs", machine_runs)]:
        monkeypatch.setattr(script, name, stub)
    for name, stub in [("state_test", state_test), ("group_tests", group_tests)]:
        monkeypatch.setattr(script, name, stub)
    monkeypatch.setattr(script.mf, "machine_successes", machine_successes)
    monkeypatch.setattr(script.inference, "rank_sum_greater", rank_sum_greater)
    script.paper_family.cache_clear()
    family = script.paper_family(tmp_path)
    script.paper_family.cache_clear()
    tests = ["t1a", "t1b", "t2b", "t3a", "t3b", "t10a", "t10b", "t10r", "t12", "t11-s3", "t11-s4", "t11-mtf4"]
    assert list(family) == tests
    assert family["t11-s4"] == (0.001, pytest.approx(0.012))  # 12 tests, the three smallest
    assert family["t1a"] == (0.01, pytest.approx(0.09))  # five 0.01s after them: 9 times
    assert family["t12"] == (0.08, pytest.approx(0.09))  # 0.08 alone, but Holm never steps down


def test_the_state_table_and_numbers_test_the_cover_against_the_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cover = [arm_run(100)] * 11 + [arm_run(600, "spent", False, None, 0.4)] * 9
    state = [arm_run(200), arm_run(300), arm_run(200, long=False)] + [arm_run(600, "spent", False, None, 0.4)] * 17

    def batch_runs(data: Path, test: str, arm: str) -> list[script.mf.ArmRun]:
        assert test == "t12/batch"
        return cover if arm == "d5-cover" else state

    monkeypatch.setattr(script, "batch_runs", batch_runs)
    assert script.state(tmp_path) == [
        "the sum and the difference (cover) & 11 of 20 & 0.34--0.74 & 100 & 100 \\\\",
        "the state, $A p + B$ & 2 of 20 & 0.03--0.30 & 200--300 & 250 \\\\",
    ]
    approximations = tmp_path / "paper" / "t12" / "batch" / "approximations"
    approximations.mkdir(parents=True)
    line = "n.json: 25 states, meet 5, cover False, cascade top 0; factor 0: 13 codes, approximation {} blocks (x), determines {}"
    (approximations / "d5-cover.txt").write_text(line.format(5, "A - B") + "\n" + line.format(1, "-") + "\n")
    (approximations / "d5-state.txt").write_text(line.format(5, "A + B") + "\n")  # a line, not a coordinate
    made = script.paper_state(tmp_path)
    assert made["t12.cover"] == "11" and made["t12.state"] == "2"
    assert made["t12.p"] == script.p_value(script.inference.fisher_exact(11, 9, 2, 18))
    assert (made["t12.cover-networks"], made["t12.cover-lines"]) == ("2", "1")
    assert (made["t12.state-networks"], made["t12.state-lines"]) == ("1", "none")


def test_a_factor_holds_a_congruence_only_with_its_blocks():
    reading = script.Reading(24, False, [(6, ["pairings", "parity"]), (2, ["parity"])])
    assert script.holds(reading, "parity", 2)
    assert script.holds(reading, "pairings", 6)
    assert not script.holds(reading, "places {0}", 4)
    assert not script.holds(script.Reading(24, False, [(6, ["parity"])]), "parity", 2)


def test_the_planted_table_and_numbers_count_machines_and_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # T10 (D19): the machine the unit; P2's machine 1 succeeding twice is one machine, two runs
    fail = arm_run(600, "spent", False, None)
    machines = {
        "p0": {0: [arm_run(200), fail], 1: [fail, fail]},
        "p1": {0: [fail, fail], 1: [fail, fail]},
        "p2": {0: [arm_run(80), fail], 1: [arm_run(100), arm_run(120)]},
    }

    def machine_runs(data: Path, test: str, arms: list[str]) -> dict[str, dict[int, list[script.mf.ArmRun]]]:
        return machines

    monkeypatch.setattr(script, "machine_runs", machine_runs)
    assert script.planted(tmp_path) == [
        "T10 P0, primitive & 1 of 2 & 1 of 4 & 200 \\\\",
        "T10 P1, planted, the target alone & 0 of 2 & 0 of 4 & -- \\\\",
        "T10 P2, planted, its congruence quotient an output & 2 of 2 & 3 of 4 & 80--120, 100 \\\\",
        "T10r P0, primitive & 1 of 2 & 1 of 4 & 200 \\\\",  # T10r (D24): its own rows, P0 and P2 only
        "T10r P2, planted, its congruence quotient an output & 2 of 2 & 3 of 4 & 80--120, 100 \\\\",
    ]
    numbers = script.paper_planted(tmp_path)
    assert numbers["t10.p2-machines"] == "2" and numbers["t10.machines"] == "2"
    assert numbers["t10.b-p"] == f"{script.inference.rank_sum_greater([1, 2], [1, 0]):.4f}"
    assert numbers["t10.b-once-p"] == f"{script.inference.rank_sum_greater([1, 1], [1, 0]):.4f}"
    assert numbers["t10r.p2-runs"] == "3" and numbers["t10r.runs"] == "4"
    assert numbers["t10r.p"] == numbers["t10.b-p"]  # the same counts, one test with no Holm
    assert numbers["t10r.fisher-p"] == f"{script.inference.fisher_exact_greater(3, 1, 1, 3):.4f}"


LINE = (
    "/x/certified-step-19.network.json: 24 states, meet 4, cover False, cascade top 0; factor 0: 11 codes,"
    " approximation 4 blocks (congruence False, autonomous 0.545), determines places {3}; factor 1: 10 codes,"
    " approximation 1 blocks (congruence False, autonomous 0.440), determines -"
)


def test_a_decomposition_row_from_text_and_json_readings(tmp_path: Path):
    (tmp_path / "s4.txt").write_text(LINE + "\n")
    factors = [{"approximation": [5] * 5, "determines": [line]} for line in ("A + B", "A - B")]
    net = {"states": 25, "cover": True, "factors": factors}
    (tmp_path / "family").mkdir()
    (tmp_path / "family" / "a.json").write_text(script.json.dumps(net))
    assert script.readings(tmp_path / "s4.txt") == [script.Reading(24, False, [(4, ["places {3}"]), (1, [])])]
    rows = script.decomposition_row("s4", script.readings(tmp_path / "s4.txt") + script.readings(tmp_path / "family"))
    assert rows == "s4 & 2 & 3/4 & 2 (1) & 1 & \\texttt{A + B} 1, \\texttt{A - B} 1, \\texttt{places \\{3\\}} 1 \\\\"


def test_proof_outcomes_name_the_stop_and_chows_verdict():
    chow = {"equivalent": False, "words": 4096, "counterexample": [1, 0]}
    assert script.proof_outcome({"verdict": "silent"}) == "silent"
    assert script.proof_outcome({"verdict": "false-stop", "chow": chow}) == "false-stop refuted"
    assert script.proof_outcome({"verdict": "kept", "chow": {"refused": "limit"}}) == "kept out of reach"
    assert (
        script.proof_outcome({"verdict": "strict-only", "chow": {**chow, "equivalent": True}}) == "strict-only proved"
    )
    assert script.natural("d7-family-300") < script.natural("d11-family-300")


def test_numbers_are_written_as_the_manuscript_writes_them():
    assert script.count(7) == "7" and script.count(20480) == "20{,}480" and script.count(19680) == "19{,}680"
    assert script.words(1, 3) == "1 to 3" and script.words(3, 3) == "3"
    assert script.none_or(0) == "none" and script.none_or(2) == "2"
    assert script.every(20, 20) == "every" and script.every(19, 20) == "19 of 20"
    assert script.listed([0, 1, 2, 4, 5]) == "0, 1, 2, 4 and 5" and script.listed([3]) == "3"
    assert script.p_value(0.014182) == "0.014" and script.p_value(0.042547) == "0.043"
    assert [script.base3_digits(x) for x in (0, 2, 3, 13119)] == [1, 1, 2, 9]


PROSE = r"\newcommand{\pn}[2]{#2} Of \pn{toy.runs}{105} networks, \pn{toy.wrong}{19{,}680}; \pn{toy.words}{1 to 3}."


def test_check_and_fill_hold_the_prose_numbers_to_their_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    tex = tmp_path / "paper.tex"
    tex.write_text(PROSE)
    monkeypatch.setattr(script, "TABLES", {})
    made = {"toy.runs": "105", "toy.wrong": "19{,}680", "toy.words": "1 to 3"}

    def source(data: Path) -> script.Numbers:
        return dict(made)

    monkeypatch.setattr(script, "NUMBER_SOURCES", [source])
    assert script.main(["check", str(tex)]) == 0
    assert "numbers: 3, the same" in capsys.readouterr().out
    made["toy.wrong"] = "13{,}119"
    assert script.main(["check", str(tex)]) == 1
    assert "toy.wrong: manuscript 19{,}680, regenerated 13{,}119" in capsys.readouterr().out
    assert script.main(["fill", str(tex)]) == 0
    assert tex.read_text() == PROSE.replace("19{,}680", "13{,}119")
    assert script.main(["check", str(tex)]) == 0
    tex.write_text(PROSE + r" \pn{toy.missing}{1}")
    assert script.main(["check", str(tex)]) == 1 and script.main(["fill", str(tex)]) == 1
    assert "no source for toy.missing" in capsys.readouterr().out


def test_numbers_refuses_a_key_from_two_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def one(data: Path) -> script.Numbers:
        return {"a": "1"}

    def two(data: Path) -> script.Numbers:
        return {"a": "2"}

    monkeypatch.setattr(script, "NUMBER_SOURCES", [one, two])
    with pytest.raises(ValueError, match="two sources"):
        script.numbers(tmp_path)
