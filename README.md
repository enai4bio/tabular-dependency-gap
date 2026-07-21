# Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models

Code and result files for the paper *"Measuring the Dependency Gap: Diagnosing Inter-Column
Fidelity in Tabular Generative Models"* (under review, TMLR).

The paper introduces a **dependency-aware fidelity diagnostic** for tabular generative models:
a strong classifier two-sample test (XGBoost-C2ST) decomposed into **marginal / dependency /
numerical–categorical cross** components, anchored between a worst-case fully-factorized
reference (FF, all dependency destroyed) and a best-case real-data oracle. Using it we show
that (i) widely used metrics (LR-C2ST, Trend) are largely blind to destroyed dependency,
(ii) a SOTA flow-matching generator (TabbyFlow/EF-VFM) carries a real, significant dependency
gap, and (iii) the gap is neither a structural limitation (recovery theorem + XOR control) nor
a capacity problem (16× parameters, paired t-tests all p>0.05), and no cheap intervention
closes it — pointing to the objective's lack of direct dependency supervision.

## Repository layout

```
ef_vfm/                 EF-VFM training code (fork of andresguzco/ef-vfm) incl. the
                        P3 cross-coupling variant (models/flow_model_fix.py,
                        modules/main_modules_fix.py) and width-sweep configs (configs/*.toml)
eval/                   downstream-utility evaluator (minority-class F1/recall)
main.py                 training/sampling entry point (--fix --mechanism p3 for the P3 variant)
process_dataset.py      dataset download/preprocessing (adult/default/bank/magic/shoppers, UCI)
proper_metrics.py       THE diagnostic: XGB-C2ST + marginal/dep/dep_cross decomposition,
                        FF/oracle references, Trend & LR-C2ST (blindness panel only)
litmus_check.py         sampler-level check that a model modification actually enters sample()
check_c2a_moment_match.py  second-order cross-moment check (residual gap is higher-order)
scripts/                run_*.py evaluation drivers and queue_*.sh training queues per experiment
toy/                    controlled XOR study (toy_blindspot.py) + its raw output (out_decisive.txt)
posthoc_copula/         post-hoc copula study (Gaussian/vine/empirical) referenced in Sec. 6
results/                all step_*.csv result files backing the paper's tables and figures
figs/                   paper figures; regenerate with make_paper_figs.py (no training needed)
```

Checkpoints, generated samples, and raw training logs are not included (hundreds of GB);
all numbers in the paper are reproducible from `results/*.csv`, and every experiment can be
re-run from scratch with the scripts below.

## Environment

Python 3.10, PyTorch (CUDA), and:

```
xgboost==3.2  scikit-learn==1.7  sdmetrics  imbalanced-learn==0.14  pandas  scipy  matplotlib
```

## Reproducing the paper

**0. Data.** `python process_dataset.py --dataname adult` (same for `default`, `bank`,
`magic`, `shoppers`; downloads from UCI and writes `data/<ds>/{train,test,info}`).

**1. Train baselines** (5 seeds each): `bash scripts/queue_adult_base.sh` etc.; deterministic
training via `python main.py --dataname <ds> --mode train --exp_name <name> --deterministic --seed <s>`.

**2. Diagnostic (Table 1, Figs. 1–2).** `python scripts/run_051_<ds>_seed5.py` computes
full/marg/dep/dep_cross + FF/oracle anchors + minority-F1 per seed;
`python scripts/run_051_blindness.py` computes the Trend/LR-C2ST blindness panel;
`python make_paper_figs.py` regenerates both figures from `results/`.

**3. Capacity sweep (Sec. 5).** `bash scripts/queue_051_capacity.sh` trains width
1×/2×/4× (dim_t 1024/2048/4096; ≈10.6M/42M/168M params), then
`python scripts/run_051_capsweep5_eval.py`. Convergence gate: per-seed final training loss
must reach the 1× level (default fails this at every width and is excluded — see paper footnote).

**4. P3 in-model fix (Sec. 6).** `python litmus_check.py` first (the mechanism must enter
`sample()`); then `bash scripts/queue_051_p3.sh` and `python scripts/run_051_p3_eval.py`
(paired, same-seed differences).

**5. Second-order / SMOTE / post-hoc copula (Sec. 6).**
`python check_c2a_moment_match.py`, `python scripts/run_048_smote.py`,
and `posthoc_copula/copula_fix.py` (prior controlled study, same protocol).

**6. XOR control (Sec. 5).** `python toy/toy_blindspot.py`; the paper's 0.98/1.00 numbers
are in `toy/out_decisive.txt`.

## Paper ↔ files map

| Paper item | Script | Result file |
|---|---|---|
| Table 1 (diagnosis, 4 datasets) | `scripts/run_051_{adult,bank,default}_seed5.py` | `results/step_*_diag_seed5_051*.csv`, `results/step_multidataset_diag.csv` |
| Blindness values (Sec. 4, Fig. 2) | `scripts/run_051_blindness.py` | `results/step_{bank,magic}_blindness_051*.csv`, `results/step1_decomp_summary.csv` (adult/shoppers) |
| Capacity sweep + paired t (Sec. 5) | `scripts/run_051_capsweep5_eval.py` | `results/step_capacity_sweep5_051*.csv`, `results/step_capacity_trainloss.csv` |
| P3 paired deltas (Sec. 6) | `scripts/run_051_p3_eval.py`, `scripts/run_b_eval.py` | `results/step_{bank,default}_p3_051*.csv` |
| Second-order moments (Sec. 6) | `check_c2a_moment_match.py` | `results/step_*_order_seed5_051*.csv` |
| SMOTE comparison (Sec. 6) | `scripts/run_048_smote.py` | `results/step_smote.csv` |
| XOR control (Sec. 5) | `toy/toy_blindspot.py` | `toy/out_decisive.txt` |
| Figures 1–2 | `make_paper_figs.py` | `figs/` |

## Attribution

The training code under `ef_vfm/` is a fork of
[EF-VFM / TabbyFlow](https://github.com/andresguzco/ef-vfm)
(Guzmán et al., *Exponential Family Variational Flow Matching for Tabular Data Generation*,
ICML 2025). Please check and respect the upstream repository's license terms.
All diagnostic, evaluation, and experiment-driver code in this repository is ours.
