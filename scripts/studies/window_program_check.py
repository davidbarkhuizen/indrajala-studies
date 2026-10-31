"""
Saved window-recurrent networks' inference through the window program against the per-layer path
(the window-program workplan, stage 3): for each network, rows of random tokens from its
embedding's vocabulary at n columns, in the long tests' chunks (`CHUNK_TOKENS // ((n + 1) * k)`
rows of n + 1 windows), every output compared by bits, and each window's states' codes. Prints a
line a network and exits 1 if any differs.

    python scripts/studies/window_program_check.py NETWORK.json [...] [--n 1024] [--rows 2000] [--seed 0]
"""

import argparse
import sys
import time

import numpy as np
from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.networks.window_recurrent_array_network import per_layer_path
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies.window_recurrent_study import CHUNK_TOKENS


def check(path: str, n: int, rows: int, seed: int) -> tuple[bool, str]:
    """Whether the program's outputs and codes are the per-layer path's on rows rows at n columns."""
    network = load_network(path)
    k = network.columns
    vocabulary = network.layers[0].vocabulary
    chunk = max(1, CHUNK_TOKENS // ((n + 1) * k))
    rng = np.random.default_rng(seed)
    times = {"program": 0.0, "per layer": 0.0}
    for start in range(0, rows, chunk):
        X = RUST.matrix(rng.integers(0, vocabulary, (min(chunk, rows - start), (n + 1) * k)).astype(float).tolist())
        begun = time.perf_counter()
        program = network.forward_windows(X).tolist(), network.state_codes(X)
        times["program"] += time.perf_counter() - begun
        with per_layer_path(network):
            begun = time.perf_counter()
            per_layer = network.forward_windows(X).tolist(), network.state_codes(X)
            times["per layer"] += time.perf_counter() - begun
        outputs = np.array(program[0], dtype=np.float64).tobytes() == np.array(per_layer[0], dtype=np.float64).tobytes()
        if not outputs or program[1] != per_layer[1]:
            return False, f"rows {start}-{start + chunk - 1} differ ({'codes' if outputs else 'outputs'})"
    seconds = ", ".join(f"{name} {value:.1f} s" for name, value in times.items())
    return True, f"{rows} rows at {n} columns, outputs and codes by bits; {seconds}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("networks", nargs="+")
    parser.add_argument("--n", type=int, default=1024, help="columns a row (default 1024)")
    parser.add_argument("--rows", type=int, default=2000, help="rows a network (default 2000)")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    same = True
    for path in arguments.networks:
        ok, line = check(path, arguments.n, arguments.rows, arguments.seed)
        same &= ok
        print(f"{'same' if ok else 'DIFFERENT'}: {path}: {line}", flush=True)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
