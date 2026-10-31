"""CI's shards (copied from ml's tests/conftest.py, the studies-repo workplan's D5): `--shard I/N`."""

from typing import cast

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        help="run only shard I of N (0-based): every Nth test from the Ith in collection order, so CI's "
        "parallel jobs split tests/ between them, the slow parametrized tests spread across all N",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = cast("str | None", config.getoption("--shard"))
    if shard is None:
        return
    index, count = (int(x) for x in shard.split("/"))
    assert 0 <= index < count, f"--shard I/N needs 0 <= I < N; got {shard}"
    kept = [item for i, item in enumerate(items) if i % count == index]
    config.hook.pytest_deselected(items=[item for i, item in enumerate(items) if i % count != index])
    items[:] = kept
