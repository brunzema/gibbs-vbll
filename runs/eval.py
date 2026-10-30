# Add the parent directory to the Python path (same pattern as train.py)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.models.vbll_model import VBLLModel
from src.models import VBLLDeltaModel
from src.utils.model_management import load_model_and_config, get_activation_fn
from src.utils.normalization import NormalizedModelWrapper
from src.training import compute_dtw_distance, compute_mmd, compute_energy_distance


# ─── Model reconstruction ────────────────────────────────────────────────────

def reconstruct_model(saved: dict):
    """Rebuild a VBLLDeltaModel or VBLLModel from a saved pickle dict."""
    cfg  = saved['config']
    mode = saved['training_metadata']['training_loss']
    n_tr = saved['training_metadata']['n_train_samples']

    state_dim   = len(cfg.environment.initial_state_bounds)
    hidden_dims = tuple(
        [cfg.model.model.hidden_dim] * cfg.model.model.num_hidden_layers
    )
    reg_weight  = 1. / n_tr
    # reg_weight  = cfg.model.model.regularization_weight / n_tr

    activation  = get_activation_fn(cfg.model.model.activation_fn)

    if 'VBLLDeltaModel' in cfg.model.model._target_:
        ModelClass = VBLLDeltaModel
    else:
        ModelClass = VBLLModel

    model = ModelClass(
        input_dim             = state_dim,
        output_dim            = state_dim,
        hidden_dims           = hidden_dims,
        feature_dim           = cfg.model.model.feature_dim,
        activation_fn         = activation,
        regularization_weight = reg_weight,
        prior_scale           = cfg.model.model.prior_scale,
        wishart_scale         = cfg.model.model.wishart_scale,
        rngs                  = nnx.Rngs(0),
    )
    nnx.update(model, saved['model_state'])
    nnx.display(model)
    return model


# ─── Environment rollout helpers ─────────────────────────────────────────────

def _rollout_det(env, x0, n_steps: int) -> np.ndarray:
    """Deterministic rollout; prefers env.rollout_deterministic if available."""
    if hasattr(env, 'rollout_deterministic'):
        return env.rollout_deterministic(np.array(x0), n_steps)
    dummy_rng = np.random.RandomState(0)
    traj = [np.array(x0)]
    cur  = np.array(x0)
    for _ in range(n_steps):
        cur = env.step(cur, dummy_rng)
        traj.append(cur.copy())
    return np.array(traj)


def _rollout_noisy(env, x0, n_steps: int, seed: int) -> np.ndarray:
    rng  = np.random.RandomState(seed)
    traj = [np.array(x0)]
    cur  = np.array(x0)
    for _ in range(n_steps):
        cur = env.step(cur, rng)
        traj.append(cur.copy())
    return np.array(traj)


# ─── Coverage ────────────────────────────────────────────────────────────────

def compute_empirical_coverage(
    samples_all: list,
    noisy_gt_all: list,
    n_test: int,
    nominal_level: float = 0.90,
):
    alpha_lo = (1.0 - nominal_level) / 2.0 * 100.0
    alpha_hi = (1.0 + nominal_level) / 2.0 * 100.0

    coverages_per_t_list = []
    for i in range(n_test):
        model_arr = np.array(samples_all[i])   # (n_noisy, T+1, D)
        gt_arr    = np.array(noisy_gt_all[i])  # (n_noisy, T+1, D)

        lo = np.percentile(model_arr, alpha_lo, axis=0)  # (T+1, D)
        hi = np.percentile(model_arr, alpha_hi, axis=0)  # (T+1, D)

        # Per-dimension indicator: (n_noisy, T+1, D)
        inside_per_dim = (gt_arr >= lo[None]) & (gt_arr <= hi[None])

        # Average over GT samples AND dimensions → marginal coverage: (T+1,)
        # Previously used np.all(..., axis=2) which gave joint coverage,
        # systematically underestimating calibration for D > 1.
        coverages_per_t_list.append(inside_per_dim.mean(axis=(0, 2)))

    coverages_per_t = np.array(coverages_per_t_list)      # (n_test, T+1)
    coverages_mean  = coverages_per_t[:, 1:].mean(axis=1) # exclude trivial t=0

    return coverages_mean, coverages_per_t


# ─── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(config_path='../config', config_name='eval_variable_speed_limit_cycle', version_base='1.3')
def main(cfg: DictConfig) -> None:
    print('=' * 80)
    print('VBLL Model Evaluation')
    print('=' * 80)
    print(OmegaConf.to_yaml(cfg))
    
    import time as _time
    timestamp = _time.strftime('%Y%m%d_%H%M%S')

    save_dir = Path(cfg.eval.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and reconstruct models ───────────────────────────────────────────
    models_data     = []
    first_saved_cfg = None
    skip_missing_models = bool(cfg.eval.get('skip_missing_models', False))
    missing_model_paths = []

    for i, mc in enumerate(cfg.models):
        model_path = Path(mc.path)
        if not model_path.exists():
            msg = f'Model file not found: {model_path}'
            if skip_missing_models:
                print(f'\n[WARN] {msg} (skipping)')
                missing_model_paths.append(str(model_path))
                continue
            raise FileNotFoundError(msg)

        print(f'\nLoading model {i + 1}/{len(cfg.models)}: {model_path}')
        saved  = load_model_and_config(model_path)
        model  = reconstruct_model(saved)
        label  = mc.get('label', saved['training_metadata']['training_loss'])
        if first_saved_cfg is None:
            first_saved_cfg = saved['config']
        # Wrap with normalization if the model was trained on normalized states
        in_norm  = saved.get('input_normalization')
        out_norm = saved.get('output_normalization')
        if in_norm and in_norm.get('enabled'):
            model = NormalizedModelWrapper(model, in_norm, out_norm)
            print(f'  → Normalization wrapper applied')
        models_data.append({'model': model, 'label': label, 'saved': saved})
        print(f'  → Label: {label}  '
              f'(mode: {saved["training_metadata"]["training_loss"]})')

    if not models_data:
        raise RuntimeError(
            'No models were loaded for evaluation. '
            f'Checked {len(cfg.models)} configured paths.'
        )

    if missing_model_paths:
        print('\n[WARN] Missing model paths:')
        for p in missing_model_paths:
            print(f'  - {p}')

    # ── Reconstruct environment from first model's config ────────────────────
    env_oc    = first_saved_cfg.environment.environment
    env_dict  = OmegaConf.to_container(env_oc, resolve=True)
    noise_std = float(env_dict.get('noise_std', 0.01))

    det_env   = hydra.utils.instantiate({**env_dict, 'noise_std': 0.0})
    noisy_env = hydra.utils.instantiate({**env_dict, 'noise_std': noise_std})

    initial_state_bounds = np.array(first_saved_cfg.environment.initial_state_bounds)
    state_dim = initial_state_bounds.shape[0]
    print(f'\nEnvironment: {first_saved_cfg.environment.name}  (dim={state_dim}, noise_std={noise_std})')
    
    # ── Build save directory: base / env_name / timestamp ─────────────────────
    env_name_str = first_saved_cfg.environment.get(
        'name',
        env_dict.get('_target_', 'unknown_env').rsplit('.', 1)[-1],
    )
    print(f'\nEnvironment: {env_name_str}  (dim={state_dim}, noise_std={noise_std})')
    save_dir = Path(cfg.eval.save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Save directory: {save_dir}')

    # ── Generate test initial states ──────────────────────────────────────────
    rollout_steps = cfg.eval.rollout_steps
    n_test        = cfg.eval.n_test_states
    n_vis         = cfg.eval.n_vis_states
    n_noisy       = cfg.eval.n_noisy_samples
    seed          = cfg.eval.seed

    np.random.seed(seed + 9999)
    test_init_all = [
        jnp.array(np.random.uniform(
            initial_state_bounds[:, 0], initial_state_bounds[:, 1]
        ))
        for _ in range(n_test)
    ]
    # Formerly picked the plotted states; kept so the global RNG stream, and with
    # it the subsampled MMD / energy-distance metrics, match earlier evaluations.
    np.random.choice(n_test, size=n_vis, replace=False)

    # ── Ground truth ──────────────────────────────────────────────────────────
    print('\nGenerating noiseless ground-truth trajectories...')
    gt_all = [
        _rollout_det(det_env, np.array(x0), rollout_steps)
        for x0 in test_init_all
    ]

    print(f'Generating {n_noisy} noisy GT samples per initial state...')
    noisy_gt_all = []
    for idx, x0 in enumerate(test_init_all):
        if idx % 20 == 0:
            print(f'  {idx}/{n_test} ...')
        noisy_gt_all.append([
            _rollout_noisy(
                noisy_env, np.array(x0), rollout_steps,
                seed + 5000 + idx * 1000 + si,
            )
            for si in range(n_noisy)
        ])

    # ── Roll out each model and compute metrics ───────────────────────────────
    def _flatten_samples(samples_list):
        return np.array([s.flatten() for s in samples_list])

    gt_flat = [_flatten_samples(noisy_gt_all[i]) for i in range(n_test)]
    
    # Shared MMD bandwidth from GT only (same ruler for all methods).
    # Use scipy.spatial.distance.pdist to avoid the O(n² × D) intermediate
    # tensor that np.linalg.norm broadcasting would allocate (can be tens of GB
    # for high-D flattened trajectories like glycolytic_oscillator at dim=1407).
    from scipy.spatial.distance import pdist
    all_gt_flat = np.concatenate(gt_flat, axis=0)
    subsample_idx = np.random.choice(len(all_gt_flat),
                                     size=min(500, len(all_gt_flat)),
                                     replace=False)
    gt_sub = all_gt_flat[subsample_idx]
    dists = pdist(gt_sub)          # condensed 1-D array of pairwise distances
    shared_bw = float(np.median(dists[dists > 0]))

    # Terminal-state bandwidth (used for terminal MMD)
    gt_term_flat = np.concatenate(
        [np.array([traj[-1] for traj in noisy_gt_all[i]]) for i in range(n_test)],
        axis=0,
    )
    term_sub_idx = np.random.choice(len(gt_term_flat),
                                    size=min(500, len(gt_term_flat)),
                                    replace=False)
    term_sub = gt_term_flat[term_sub_idx]
    term_dists = pdist(term_sub)
    terminal_bw = float(np.median(term_dists[term_dists > 0]))

    for mi, mentry in enumerate(models_data):
        model = mentry['model']
        label = mentry['label']
        print(f'\nRolling out model {mi + 1}/{len(models_data)}: {label}')

        # Mean trajectories (all test states)
        mean_trajs = [
            model.rollout_mean_trajectory(x0, rollout_steps)
            for x0 in test_init_all
        ]

        # Stochastic samples (all test states)
        samples_all = []
        for idx, x0 in enumerate(test_init_all):
            if idx % 20 == 0:
                print(f'  {idx}/{n_test} ...')
            keys  = jax.random.split(
                jax.random.PRNGKey(seed + (mi + 1) * 2000 + idx * 1000),
                n_noisy,
            )
            samps = jax.vmap(
                lambda k: model.simulate(x0, rollout_steps, k)
            )(keys)
            samples_all.append([np.array(s) for s in samps])

        # ── Metrics ───────────────────────────────────────────────────────────
        gt_final   = np.stack([g[-1] for g in gt_all])
        m_final    = np.stack([np.array(m)[-1] for m in mean_trajs])
        final_l2   = np.linalg.norm(m_final - gt_final, axis=1)

        cumul_l2   = np.array([
            np.sum(np.linalg.norm(
                np.array(m)[1:] - np.array(g)[1:], axis=1
            ))
            for m, g in zip(mean_trajs, gt_all)
        ])

        dtw_dist   = np.array([
            compute_dtw_distance(np.array(m), np.array(g))
            for m, g in zip(mean_trajs, gt_all)
        ])

        m_flat     = [_flatten_samples(samples_all[i]) for i in range(n_test)]
        mmd_vals   = np.array([
            compute_mmd(m_flat[i], gt_flat[i], bandwidth=shared_bw) for i in range(n_test)
        ])
        ed_vals    = np.array([
            compute_energy_distance(m_flat[i], gt_flat[i]) for i in range(n_test)
        ])

        terminal_mmd_vals = np.array([
            compute_mmd(
                np.array([s[-1] for s in samples_all[i]]),
                np.array([traj[-1] for traj in noisy_gt_all[i]]),
                bandwidth=terminal_bw,
            )
            for i in range(n_test)
        ])
        terminal_ed_vals = np.array([
            compute_energy_distance(
                np.array([s[-1] for s in samples_all[i]]),
                np.array([traj[-1] for traj in noisy_gt_all[i]]),
            )
            for i in range(n_test)
        ])

        # Empirical coverage at multiple nominal levels
        coverage_levels = [0.50, 0.70, 0.90, 0.95]
        coverage_data = {}
        for level in coverage_levels:
            cov_mean, cov_per_t = compute_empirical_coverage(
                samples_all, noisy_gt_all, n_test, nominal_level=level,
            )
            key_suffix = f'{int(level * 100)}'
            coverage_data[f'coverage_{key_suffix}']       = cov_mean
            coverage_data[f'coverage_per_t_{key_suffix}'] = cov_per_t

        mentry.update({
            'mean_trajs':      mean_trajs,
            'samples_all':     samples_all,
            'final_l2':        final_l2,
            'cumul_l2':        cumul_l2,
            'dtw':             dtw_dist,
            'mmd':             mmd_vals,
            'energy_dist':     ed_vals,
            'terminal_mmd':    terminal_mmd_vals,
            'terminal_energy': terminal_ed_vals,
            **coverage_data,
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('EVALUATION SUMMARY')
    print('=' * 80)
    for mentry in models_data:
        mode = mentry['saved']['training_metadata']['training_loss']
        print(f"\n{mentry['label']}  ({mode}):")
        for key, lbl in [
            ('final_l2',        'Final L2    '),
            ('cumul_l2',        'Cumul L2    '),
            ('dtw',             'DTW         '),
            ('mmd',             'MMD²        '),
            ('energy_dist',     'Energy Dist '),
            ('terminal_mmd',    'Term. MMD²  '),
            ('terminal_energy', 'Term. Energy'),
            ('coverage_90',     'Coverage90% '),
            ('coverage_50',     'Coverage50% '),
            ('coverage_70',     'Coverage70% '),
            ('coverage_95',     'Coverage95% '),
        ]:
            v = mentry[key]
            print(f'  {lbl}: mean={np.mean(v):.4f}  '
                  f'median={np.median(v):.4f}  '
                  f'std={np.std(v):.4f}')

    # ── Save statistics pkl ────────────────────────────────────────────────────
    import pickle

    metric_keys = [
        'final_l2', 'cumul_l2', 'dtw', 'mmd', 'energy_dist',
        'terminal_mmd', 'terminal_energy',
        'coverage_50', 'coverage_70', 'coverage_90', 'coverage_95',
        'coverage_per_t_50', 'coverage_per_t_70', 'coverage_per_t_90', 'coverage_per_t_95',
    ]
    stats_records = []
    for mentry in models_data:
        mode = mentry['saved']['training_metadata']['training_loss']
        record = {
            'label': mentry['label'],
            'training_loss': mode,
        }
        for key in metric_keys:
            v = mentry[key]
            arr = np.array(v)
            record[key] = arr
            record[f'{key}_mean']   = float(np.mean(arr))
            record[f'{key}_median'] = float(np.median(arr))
            record[f'{key}_std']    = float(np.std(arr))
            record[f'{key}_q05']    = float(np.percentile(arr, 5))
            record[f'{key}_q95']    = float(np.percentile(arr, 95))
        stats_records.append(record)

    stats_path = save_dir / 'stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats_records, f)
    print(f'Statistics saved to: {stats_path.resolve()}')


if __name__ == '__main__':
    main()
