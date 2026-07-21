import torch
from ef_vfm.main import main as ef_vfm_main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of EF-VFM (TabbyFlow) for tabular data generation')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name dataset, one of those in data/ dir')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--method', type=str, default='ef_vfm', help='Currently we only release our model EF-VFM. Baselines will be released soon.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, used to name log directories and the wandb run name')
    parser.add_argument('--deterministic', action='store_true', help='Whether to make the entire process deterministic, i.e., fix global random seeds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (used when --deterministic is set)')
    parser.add_argument('--config_path', type=str, default=None, help='Override path to the TOML config (047 capacity check: width x4 config). Default keeps the original 1x toml.')
    
    # Configs for testing ef_vfm
    parser.add_argument('--num_samples_to_generate', type=int, default=None, help='Number of samples to be generated while testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint to be tested')
    parser.add_argument('--report', action='store_true', help="Report testing mode: this mode sequentially runs <num_runs> test runs and report the avg and std")
    parser.add_argument('--num_runs', type=int, default=20, help="Number of runs to be averaged in the report testing mode")

    # Phase B fix flags
    parser.add_argument('--fix', action='store_true', help='Use Phase B fixed model (P1+P3): low-rank cov + num→cat cross-coupling')
    parser.add_argument('--rank', type=int, default=2, help='Low-rank factor dimension r for P1 (default 2)')
    parser.add_argument('--mechanism', type=str, default='p1p3', choices=['p1p3', 'p3', 'p1'],
                        help="Which fix mechanism to enable: 'p3'=cross_head only (P1 disabled, isotropic cov), "
                             "'p1'=L_head only, 'p1p3'=both. Only used with --fix. "
                             "Use 'p3' to cleanly attribute results to P3 (num->cat cross-coupling).")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    ef_vfm_main(args)