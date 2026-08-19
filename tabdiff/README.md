# TabDiff patch

This directory holds only the script we added on top of the official TabDiff
repository (Xu et al., MIT license) to run the NFE (number-of-function-
evaluations / diffusion step count) sweep reported in the paper's Appendix.

To reproduce: clone the official TabDiff repository
(https://github.com/MinkaiXu/TabDiff), set up its environment per its own
README, then drop `run_056_nfe_resample_tabdiff.py` into the repo root
(same layout as this directory). It reuses TabDiff's own `main.py`/model
loading and only adds the step-count resampling loop.

`ef-vfm-fix/eval_056_nfe_tabdiff.py` (in the sibling `ef-vfm-fix/` directory
of this repo) evaluates the resulting samples with the same
`proper_metrics.py` diagnostic used throughout the paper, so the TabDiff NFE
numbers are directly comparable to the EF-VFM ones.

We did not modify TabDiff's own model/training code; the sweep only varies
the sampler's step count at inference time.
