# indrajala-studies

The research studies run on [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml): their
code, scripts, configs and tests, moved out of ml so the library carries less of what a reuser never
runs. The findings, the papers and the research workplans are in
[indrajala-research](https://github.com/davidbarkhuizen/indrajala-research).

Since 2026-10-31 every piece of the window-recurrent research lives here (ml's research-separation
workplan): the harness, its tasks and configs, the certificate, the readings, the modular family and
its probe, the batch runner, and the addition and window-recurrent study scripts. ml keeps the
library (the window-recurrent network and its window presets, `indrajala_ml/model/specs/window_presets.py`)
and the primitive-era studies that validate its features.

- `indrajala_studies/`: the modules (`window_recurrent_study`, `window_tasks`, `modular_family`,
  `batch`, `addition_data`, ...).
- `scripts/studies/`, `scripts/paper/`, `scripts/runner/`, `scripts/measurement/`, `scripts/data/`:
  the command lines; `scripts/studies/configs/` the runs' configurations.
- `docs/working-notes.md`: the gotchas of running and reading studies (ml's own notes cover git,
  the machines and the shell).
- `data/` (gitignored): every study's results and networks, with their manifests.

**Timing:** ml's `ab.py` runs `scripts/measurement/window_study_timing.py` (the long test and a study
run end to end) by its absolute path, each side's ml tree supplying the library:
`ab.py remote --host jebel run --bench cmd ... -- /home/david/code/indrajala-studies/scripts/measurement/window_study_timing.py`.

**Paper 1** is reproduced from the tags `paper1-v1` and `paper1-v1.1` in ml and the crate, where the
paper's scripts still live under their old paths, and from its datapack
[doi:10.5281/zenodo.23733575](https://doi.org/10.5281/zenodo.23733575).

## Setup

```sh
git clone --recurse-submodules git@github.com:davidbarkhuizen/indrajala-studies.git
cd indrajala-studies
./cli setup   # the venv, ml (editable) and its pinned dependencies, the crate built from ml/rust
./cli test
./cli lint    # ruff and strict pyright, at ml's pins
```

The build needs [rustup](https://rustup.rs); `ml/rust/rust-toolchain.toml` pins the toolchain.

## Its relation to ml

- **ml is a git submodule at `ml/`**, with the crate inside it at `ml/rust/`, so a studies commit
  pins both. A study's provenance records this repository's commit and whether the tree is dirty
  (a moved submodule makes it so).
- **Code here may import ml; ml never imports this.**
- **A change that needs new ml code merges in ml first**, then a "Bump ml/" PR here:
  `git submodule update --remote ml`, then `./cli setup` and `./cli test`.

## Licence

MIT, as ml.
