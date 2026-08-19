# Datasets

This repository does not bundle the raw/processed dataset files (they are not
ours to redistribute, and several are 10s–100s of MB). To reproduce the
experiments, place each dataset under `data/<name>/` following the layout
`info.json` + `train.csv` + `test.csv` (+ cached `.npy` splits), matching the
convention used by TabDDPM / TabSyn-style tabular benchmarks.

Datasets used in this project (all public, standard UCI-derived benchmarks
used throughout the tabular generative modeling literature — download from
the UCI Machine Learning Repository or the original TabDDPM/TabSyn repos'
data-preparation scripts):

- `adult` — UCI Adult / Census Income
- `bank` — UCI Bank Marketing
- `default` — UCI Default of Credit Card Clients
- `magic` — UCI MAGIC Gamma Telescope
- `shoppers` — UCI Online Shoppers Purchasing Intention
- `beijing` — Beijing PM2.5 (regression)
- `news` — UCI Online News Popularity (regression)
- `diabetes` — Pima Indians Diabetes

`info.json` in each dataset directory records the column layout (`num_col_idx`,
`cat_col_idx`, `target_col_idx`, `task_type`) that `process_dataset.py` and the
training/eval scripts expect. We did not independently re-verify the exact
redistribution license terms of each source dataset for this note — check the
original UCI page for each dataset before redistributing the raw files
yourself.

`baselines/<dataset>/samples.csv` (also not bundled here) holds generated
samples from comparison methods (e.g. TabDiff) used for the cross-model
comparisons in the paper; regenerate via each baseline's own repository.
