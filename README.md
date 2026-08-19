# Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models

Code and result files for the paper *"Measuring the Dependency Gap: Diagnosing Inter-Column
Fidelity in Tabular Generative Models"* (under review, TMLR).

The paper introduces a **dependency-aware fidelity diagnostic** for tabular generative models:
a strong classifier two-sample test (XGBoost-C2ST) decomposed into **marginal / dependency /
numerical–categorical cross** components, anchored between a worst-case fully-factorized
reference (FF, all dependency destroyed) and a best-case real-data oracle. Applied to a
flow-matching generator (TabbyFlow/EF-VFM) and a diffusion generator (TabDiff) on four
quantitative datasets (adult/default/bank/magic, plus shoppers qualitatively), we show that
(i) widely used metrics (LR-C2ST, Trend) are largely blind to destroyed dependency, (ii) both
generator families carry a real, significant dependency gap of the same order, (iii) the gap
is neither a structural limitation (recovery theorem + XOR control) nor a capacity problem
(16× parameters, paired t-tests all p>0.05), and (iv) no cheap intervention we test — an
explicit discrete-aware cross-coupling head, second-order moment matching, or a post-hoc
Gaussian copula — closes it, pointing to the training objective's lack of direct dependency
supervision rather than to model capacity or architecture.

## Repository layout

```
ef_vfm/                 EF-VFM training code (fork of andresguzco/ef-vfm) incl. the
                        P3 cross-coupling variant (models/flow_model_fix.py,
                        modules/main_modules_fix.py) and width-sweep configs (configs/*.toml)
tabdiff/                patch on top of the official TabDiff repo (Xu et al., MIT license):
                        just the NFE/step-count sweep script we added, see tabdiff/README.md
eval/                   downstream-utility evaluator (minority-class F1/recall)
main.py                 training/sampling entry point (--fix --mechanism p3 for the P3 variant)
process_dataset.py      dataset download/preprocessing (adult/default/bank/magic/shoppers, UCI)
proper_metrics.py       THE diagnostic: XGB-C2ST + marginal/dep/dep_cross decomposition,
                        FF/oracle references, Trend & LR-C2ST (blindness panel only)
litmus_check.py         sampler-level check that a model modification actually enters sample()
                        (necessary, not sufficient, for the modification to add expressive
                        power -- see the P3 discussion in the paper and in scripts/)
check_c2a_moment_match.py  second-order cross-moment check (residual gap is higher-order)
posthoc_copula/         post-hoc Gaussian copula study (Sec. 6)
scripts/                run_*.py evaluation drivers, eval_*.py metric computers, and
                        queue_*.sh training queues, one set per experiment
toy/                    controlled XOR study (toy_blindspot.py) + its raw output (out_decisive.txt)
results/                all step_*.csv result files backing the paper's tables and figures
figs/                   paper figures (Fig. 1 metric-blindness panel, Fig. 2 gap decomposition
                        across adult/default/bank/magic); regenerate with
                        scripts/run_063_fig_gap_decomposition.py and
                        scripts/run_061_fig_metric_blindness.py (no training needed)
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
`python scripts/run_061_fig_metric_blindness.py` computes the Trend/LR-C2ST blindness panel
and regenerates Fig. 1; `python scripts/run_063_fig_gap_decomposition.py` regenerates Fig. 2
across all four datasets.

**3. Capacity sweep, up and down (Sec. 5).** Upward: `bash scripts/queue_051_capacity.sh`
trains width 1×/2×/4× (dim_t 1024/2048/4096; ≈10.6M/42M/168M params), then
`python scripts/run_051_capsweep5_eval.py`. Downward: `bash scripts/queue_056_capacity_down.sh`
(1/8×–1/2×), evaluated with `scripts/eval_056_capacity_down.py`. Convergence gate: per-seed
final training loss must reach the 1× level (default fails this at every width and is
excluded from the comparison — see paper footnote, range 18.8–19.3 across all width-sweep runs).

**4. P3 in-model fix (Sec. 6).** `python litmus_check.py` first (the mechanism must enter
`sample()` — this is a necessary "is the path alive" check, not a test of expressive power,
see the paper's discussion of this distinction); then `bash scripts/queue_adult_fix_p3.sh`
(and the bank/default equivalents) and `python scripts/run_051_p3_eval.py` /
`scripts/run_063_p3_adult_eval.py` (paired, same-seed differences).

**5. Second-order / SMOTE / post-hoc copula (Sec. 6).**
`python check_c2a_moment_match.py`, `python scripts/run_048_smote.py`,
and `posthoc_copula/copula_fix.py`.

**6. NFE / step-count sweep (Sec. 5, Appendix).** EF-VFM:
`python scripts/run_056_nfe_resample.py` + `scripts/eval_056_nfe.py`. TabDiff (requires the
official TabDiff repo, see `tabdiff/README.md`): `python tabdiff/run_056_nfe_resample_tabdiff.py`
+ `scripts/eval_056_nfe_tabdiff.py`.

**7. TabDiff cross-validation (Sec. 5).** The same diagnostic run on TabDiff (diffusion,
crossing the flow-matching/diffusion boundary): `scripts/run_tabdiff_diag.py`,
`scripts/run_tabdiff_marginal.py`, `scripts/run_tabdiff_robust.py`.

**8. XOR control (Sec. 5).** `python toy/toy_blindspot.py`; the paper's 0.98/1.00 numbers
are in `toy/out_decisive.txt`.

## Paper ↔ files map

| Paper item | Script | Result file |
|---|---|---|
| Table 1 (diagnosis, 4 datasets) | `scripts/run_051_{adult,bank,default}_seed5.py`, `scripts/run_magic_diag5.py` | `results/step_{adult,bank,default}_diag_seed5_051*.csv`, `results/step_magic_diag_seed5*.csv` |
| Blindness panel (Sec. 4, Fig. 1) | `scripts/run_061_fig_metric_blindness.py`, `scripts/run_060_trend_ff.py` | `results/step_061_fig_metric_blindness*.csv`, `results/step_060_trend_ff*.csv` |
| Gap decomposition (Fig. 2) | `scripts/run_063_fig_gap_decomposition.py` | `results/step_063_gap_decomp_summary.csv` |
| Capacity sweep up + paired t (Sec. 5) | `scripts/run_051_capsweep5_eval.py` | `results/step_capacity_sweep5_051*.csv`, `results/step_capacity_trainloss.csv` |
| Capacity sweep down (Sec. 5) | `scripts/eval_056_capacity_down.py` | `results/step_056_capacity_down*.csv` |
| P3 paired deltas (Sec. 6) | `scripts/run_051_p3_eval.py`, `scripts/run_063_p3_adult_eval.py`, `scripts/run_b_eval.py` | `results/step_{adult,bank,default}_p3_051*.csv` |
| Second-order moments (Sec. 6) | `check_c2a_moment_match.py` | `results/step_*_order_seed5_051*.csv`, `results/step_c2a_moment_match.csv` |
| SMOTE comparison (Sec. 6) | `scripts/run_048_smote.py` | `results/step_smote*.csv` |
| Post-hoc copula (Sec. 6) | `posthoc_copula/copula_fix.py` | `posthoc_copula/phase2_copula_compare.csv` |
| NFE / step-count sweep (Sec. 5, App.) | `scripts/run_056_nfe_resample.py`, `tabdiff/run_056_nfe_resample_tabdiff.py` | `results/step_056_nfe_sweep*.csv` |
| TabDiff cross-validation (Sec. 5) | `scripts/run_tabdiff_diag.py`, `scripts/run_tabdiff_marginal.py`, `scripts/run_tabdiff_robust.py` | `results/step_tabdiff_*.csv` |
| XOR control (Sec. 5) | `toy/toy_blindspot.py` | `toy/out_decisive.txt` |
| Figures 1–2 | `scripts/run_061_fig_metric_blindness.py`, `scripts/run_063_fig_gap_decomposition.py` | `figs/` |

## Attribution

The training code under `ef_vfm/` is a fork of
[EF-VFM / TabbyFlow](https://github.com/andresguzco/ef-vfm)
(Guzmán et al., *Exponential Family Variational Flow Matching for Tabular Data Generation*,
ICML 2025). Please check and respect the upstream repository's license terms.

`tabdiff/` is a patch on top of [TabDiff](https://github.com/MinkaiXu/TabDiff)
(Shi et al., MIT license); only our own added script is included here, see `tabdiff/README.md`.

All diagnostic, evaluation, and experiment-driver code in this repository is ours.
