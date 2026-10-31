"""
Stand-in adders for the addition catalogue's tests and the study harness's: one correct, and each
broken one wrong in a known way, so the properties it must fail are known.
"""

from collections.abc import Sequence

from indrajala_studies.addition_data import BASE, Adder, Triple, column_sums, from_digits


def correct(triples: Sequence[Triple]) -> list[int]:
    return [sum(t) for t in triples]


def no_carry(n: int) -> Adder:
    """Each column's digit sum mod 3, the top digit 0: commutative and associative, and wrong."""

    def f(triples: Sequence[Triple]) -> list[int]:
        return [from_digits([s % BASE for s in column_sums(t, n)]) for t in triples]

    return f


def reach(n: int, length: int) -> Adder:
    """
    Each column's carry composed from carry 0 at length + 1 columns below it: right up to a carry
    distance of length, M3's L* = length.
    """

    def f(triples: Sequence[Triple]) -> list[int]:
        answers: list[int] = []
        for t in triples:
            sums = (*column_sums(t, n), 0)
            out: list[int] = []
            for column in range(n + 1):
                k = 0
                for s in sums[max(0, column - length - 1) : column]:
                    k = (s + k) // BASE
                out.append((sums[column] + k) % BASE)
            answers.append(from_digits(out))
        return answers

    return f


def drops_third(triples: Sequence[Triple]) -> list[int]:
    return [a + b for a, b, _c in triples]
