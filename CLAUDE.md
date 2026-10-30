# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code studying **injectivity of transformer MLP blocks** (`x -> W2 · ReLU(W1 x + b1)`). The core question: do distinct inputs in different ReLU linear regions collapse to the same output? The repo trains toy transformers on synthetic tasks, counts "collapsing" region pairs during training, and experiments with projecting `W2` during training (the `gamma` parameter) to keep the block closer to injective. A scaled-up version of the projection experiment runs on a NanoGPT speedrun derivative in `nanogpt_train/`.

There is no package structure, build system, or linter config. Python 3.12+ is required (`main.py` uses nested same-quote f-strings); the environment is conda (see `.vscode/settings.json`). Most scripts assume a CUDA GPU.

## Commands

```bash
python main.py            # train toy transformer(s); config is hard-coded in main()
python analysisv2.py      # standalone collapse analysis; edit main() (random=True/False, model_file)
python data.py            # sanity-print the synthetic datasets
```

There is no test suite and no pytest-style tests. The only validation script is for the cuOpt extension. It is a plain script with no `test_*` functions, so `pytest` collects nothing; run it directly:

```bash
cd cuopt_linprog && CUOPT_ROOT=/path/to/cuopt python test_cuopt_linprog.py
```

Building that extension: `cd cuopt_linprog && CUOPT_ROOT=/path/to/cuopt pip install .` (see `cuopt_linprog/README.md`). The CPU `testing_stub/` the README mentions is only inside `torch-cuopt-lp.zip` and isn't unpacked in the repo. The extension is currently not used by the main pipeline — the import in `analysisv2.py` is commented out.

## Architecture

**Training loop (`main.py`)** — `RunConfig` dataclass holds everything; experiments are set up by editing `main()` directly (seeds, task, gamma sweep). `run()` builds data via `data.make_dataset`, a `toy_model.Transformer`, and trains with AdamW. Every `log_freq` epochs it logs loss/accuracy and runs collapse analysis on each block's FFN. `RunConfig.file_name(dir)` generates the path stem used for `logs/`, `figs/` and `models/` outputs (all gitignored, although some older files are tracked) — the stem embeds the task kwargs `repr`, which is why filenames contain `{'n_keys': 128}` etc.

**Tasks (`data.py`)** — `modular`, `kv`, `kv_seq`, `boolean`. All share one framing: input is a token sequence, the target is read from a single position `meta.target_pos`, and the loss is cross-entropy over `meta.vocab_size`. `kv` tasks are pure memorization, so they report *train* accuracy (`main.py` switches on `config.task`); other tasks report test accuracy.

**Model (`toy_model.py`)** — a small pre-norm transformer. Two hooks matter for the research:
- `realized=True` makes each `FFN` stash its post-ReLU activations (`realized_sample`); `Transformer.get_realized()` returns one per block. `main.py` converts these to the set of unique sign patterns = the linear regions actually visited by the data.
- `proj(gamma)` replaces `l2.weight` with `projection.project_kernel_into_safe_region(...)`, applied after each optimizer step when `gamma < 1`.
- The model is `torch.compile`d, so saved state dicts have an `_orig_mod.` prefix; loaders in the analysis scripts try both key forms.

**Collapse analysis (`analysisv2.py`)** — for `A = W1`, `b = b1`, `B = W2`:
1. `region_patterns` enumerates every linear region (±1 sign vector) by vertex enumeration over `C(m, n)` hyperplane subsets. This only scales to tiny `dim` (the toy default is `dim=4, h_dim=16`).
2. `region_maps` builds the per-region affine maps `B·D_r·A`.
3. For each region pair, an LP asks whether two distinct points (one in each region) map to the same output. `find_n_pairs_collapsing_torch` solves these as **batched LPs in pure PyTorch** (a custom dense simplex: `_lp_*`, `batched_linprog`), checks witnesses, and returns `(n_hits, R)`. `alpha_check` is the fraction of the `C(R,2)` pairs that get sampled; pass `patterns=` to restrict the analysis to the realized regions.

`analysis.py` is the older scipy/multiprocessing version that `initialisation_test.py` still imports; `analysisv2.py` is a superset of it and is the one `main.py` uses.

**Projection (`projection.py`, `proj_v2.py`, `proj_v3.py`)** — successive versions of "find the nearest `W2` whose kernel stays at a bounded angle from `col(W1)`". The toy model uses `projection.project_kernel_into_safe_region`; `nanogpt_train` uses `proj_v3.nearest_with_kernel_angle_bound_fast` (the math is in the `proj_v3.py` docstring). `nanogpt_train/` keeps **its own copies** of `projection.py` and `proj_v3.py` — changes to the root copies do not propagate.

**Plotting (`grapher.py`)** — `create_graphs(met1, met2, ...)` draws a twin-axis plot of two metric series (or lists of series, one per seed/layer) and saves it to `{plot_file}_{met1}_{met2}.png`. The rest of the file is a large pasted log string used by its `main()`.

**Theory (`notes/*.tex`)** — the maths behind the code. Read these before changing an algorithm:
- `Algorithms (2).tex`: region enumeration via vertices and the pairwise and k-way overlap LPs (what `region_patterns`, `distinct_collapse*` and `find_kway_collisions` implement).
- `bound (1).tex`: why the projection works. `B` is injective on the ReLU image when `ker(B) ⊆ ker(Aᵀ)`, and this stays true while the largest principal angle between the two kernels is below `π/2 − α(A, c)`. This is the "safe region" that `gamma` controls.
- `tojohn (1).tex`: the nearest-matrix-with-angle-bounded-kernel problem and the algorithm behind `proj_v3.py`.

Other top-level scripts (`temp.py`, `test.py`, `random_walk.py`, `matrix_analysis.py`, `distributions.ipynb`) are scratch experiments.
