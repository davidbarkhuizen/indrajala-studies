# indrajala-studies

The research studies run on [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml): their
code, scripts, configs and tests, moved out of ml so the library carries less of what a reuser never
runs. The findings, the papers and the research workplans are in
[indrajala-research](https://github.com/davidbarkhuizen/indrajala-research).

It is a first step, not the whole split: the code moved is what nothing left in ml needs (ml's
`docs/workplans/infrastructure/studies-repo-workplan.md`). The window-recurrent harness, its tasks,
the certificate, `chow`, `window_replay` and the batch runner stay in ml for now, and so do the
window-recurrent configs: run them from `ml/scripts/studies/configs/`.

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
