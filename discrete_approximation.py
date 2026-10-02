from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer
from pathlib import Path

import uqpfn

DEFAULT_B_POSTSAMPLES = 50
DEFAULT_T_FWDSAMPLES = 500
DEFAULT_RHO = 0.99
FONG_ALPHA_C = 2.0
FONG_ALPHA_BETA = 1.0

def get_posterior(y_grid: np.ndarray,
                  n_train: int,
                  B_postsamples: int,
                  T_fwdsamples: int,
                  cdf_conditionals: np.ndarray,
                  discrete: bool = False,
                  rho: float = DEFAULT_RHO,
                  seed: int = 100,
                  alpha_schedule=None,
                  alpha_dimension: int | None = None,
                  n_eff = None,
                  shared_uniform: bool = True,
                  use_blowup: bool = True,) -> list[dict]:
        '''adds discrete, indicating whether the outcome is continuous, and thus
        the grid is just arbitrary evaluation points of Y, or if it is discrete,
        meaning the grid actually indicates all values {1,..,K}; importantly it also
        tells how to draw u'''
    
        cdf_conditionals = np.asarray(cdf_conditionals, dtype = float)
        K, n_eval = cdf_conditionals.shape
        support = np.asarray(y_grid).ravel()
    
        if n_eff is None:
            n_eff_arr = None
        else:
            n_eff_arr = np.asarray(n_eff, dtype = float)
            if n_eff_arr.ndim == 0:
                n_eff_arr = np.full(n_eval, float(n_eff_arr))
            elif n_eff_arr.shape != (n_eval,):
                raise ValueError(
                    f'n_eff must be scalar or shape ({n_eval},), got {n_eff_arr.shape}'
                )
    
        if alpha_schedule is None:
            if alpha_dimension is None:
                raise ValueError(
                    'alpha_dimension is required when alpha_schedule is omitted; '
                    'get_posterior no longer assumes d = 1.'
                )
            if alpha_dimension <= 0:
                raise ValueError(f'alpha_dimension must be positive, got {alpha_dimension}')

        rho_arr = np.array([[float(rho)]], dtype = 'float64')
        if discrete:
            cop = pv.Bicop(family = pv.BicopFamily.gaussian, rotation = 0, parameters = rho_arr,
                           var_types = ['d', 'd'])
        else:
            cop = pv.Bicop(family = pv.BicopFamily.gaussian, rotation = 0, parameters = rho_arr)
        eps = 1e-6
    
        N = T_fwdsamples + n_train
        ysim_all = np.zeros((n_eval, N, B_postsamples))
        cdf_all = np.zeros((n_eval, K, B_postsamples))
    
        if use_blowup and alpha_dimension is not None:
            alpha_blowup = blowup_integral(n_train, alpha_dimension, T_fwdsamples)
        else:
            alpha_blowup = 1.0

        if discrete:
            for b in range(B_postsamples):
                np.random.seed(seed + b)
                cdf_b = np.clip(cdf_conditionals.T.copy(), eps, 1.0 - eps)
                cdf_prev = np.zeros((n_eval, K))
                cdf_prev[:, 1:] = cdf_b[:, :-1]
                ysim = np.zeros((n_eval, N))

                for i in range(n_train + 1, N):
                    if alpha_schedule is None:
                        alpha = default_alpha_schedule(i, alpha_dimension)
                    else:
                        alpha = alpha_schedule(i)

                    alpha = np.asarray(alpha, dtype = float)
                    if alpha.ndim == 0:
                        alpha = np.full((n_eval, 1), float(alpha))
                    elif alpha.shape == (n_eval,):
                        alpha = alpha[:, None]
                    elif alpha.shape != (n_eval, 1):
                        raise ValueError(
                            'alpha_schedule must return a scalar, shape (n_eval,) or '
                            f'shape (n_eval, 1); got {alpha.shape}'
                        )
                    alpha = alpha * alpha_blowup
                    
                    if shared_uniform:
                        u_val = np.random.uniform()
                        yind = (cdf_b < u_val).sum(axis = 1)
                    else:
                        u_vals = np.random.uniform(size = n_eval)
                        yind = (cdf_b < u_vals[:, None]).sum(axis = 1)

                    yind = np.clip(yind, 0, K - 1)
                    rows = np.arange(n_eval)
                    drawn_us = cdf_b[rows, yind]
                    prev_us = np.where(yind > 0, cdf_b[rows, yind - 1], eps)
                    u2 = np.broadcast_to(drawn_us[:, None], (n_eval, K)).copy()
                    v2 = np.broadcast_to(prev_us[:, None], (n_eval, K)).copy()
                    
                    u = np.stack([np.clip(cdf_b, eps, 1.0 - eps), np.clip(u2, eps, 1.0 - eps),
                                 np.clip(cdf_prev, eps, 1.0 - eps), np.clip(v2, eps, 1.0 - eps)], 
                                 axis = -1).reshape(-1, 4)
                    
                    h = cop.hfunc2(u).reshape(n_eval, K)

                    cdf_b = (1 - alpha) * cdf_b + alpha * h
                    cdf_b /= np.maximum(np.max(cdf_b, axis = 1, keepdims = True), eps)
                    cdf_b = np.clip(cdf_b, eps, 1.0 - eps)

                    ysim[:, i] = support[yind].squeeze()

                ysim_all[:, :, b] = ysim
                cdf_all[:, :, b] = cdf_b
        else:
            for b in range(B_postsamples):
                np.random.seed(seed + b)
                cdf_b = np.clip(cdf_conditionals.T.copy(), eps, 1.0 - eps)
                ysim = np.zeros((n_eval, N))

                for i in range(n_train + 1, N):
                    if alpha_schedule is None:
                        alpha = default_alpha_schedule(i, alpha_dimension)
                    else:
                        alpha = alpha_schedule(i)

                    alpha = np.asarray(alpha, dtype = float)
                    if alpha.ndim == 0:
                        alpha = np.full((n_eval, 1), float(alpha))
                    elif alpha.shape == (n_eval,):
                        alpha = alpha[:, None]
                    elif alpha.shape != (n_eval, 1):
                        raise ValueError(
                            'alpha_schedule must return a scalar, shape (n_eval,) or '
                            f'shape (n_eval, 1); got {alpha.shape}'
                        )
                    alpha = alpha * alpha_blowup
                
                    if shared_uniform:
                        u_val = np.random.uniform()
                        u2 = np.full((n_eval, K), u_val)
                    else:
                        u_vals = np.random.uniform(size = n_eval)
                        u2 = np.repeat(u_vals[:, None], K, axis = 1)
                    u = np.stack([np.clip(cdf_b, eps, 1.0 - eps), np.clip(u2, eps, 1.0 - eps)], axis = -1).reshape(-1, 2)
                    h = cop.hfunc2(u).reshape(n_eval, K)
                    
                    cdf_b = (1 - alpha) * cdf_b + alpha * h
                    cdf_b /= np.maximum(np.max(cdf_b, axis = 1, keepdims = True), eps)
                    cdf_b = np.clip(cdf_b, eps, 1.0 - eps)

                    if shared_uniform:
                        idx = np.array([np.searchsorted(cdf_b[j], u_val) for j in range(n_eval)])
                        idx = np.clip(idx, 0, K - 1)[:, None]
                    else:
                        idx = np.searchsorted(cdf_b, u_vals[:, None])
                        idx = np.clip(idx, 0, K - 1)

                    ysim[:, i] = support[idx].squeeze()
    
                ysim_all[:, :, b] = ysim
                cdf_all[:, :, b] = cdf_b
    
        return [{'ysim': ysim_all[j], 'cdf': cdf_all[j]} for j in range(n_eval)]

def blowup_integral(n, d, T):
    gamma = 4.0 / (1.1 * d+ 4.0)
    return 1.0 / np.sqrt(1.0 - (1.0 + T / n) ** (-gamma))
    
def _build_alpha_schedule(C: float, beta: float):
    return lambda i: C * (i + 1) ** (-beta)

def default_alpha_schedule(i: int, d: int) -> float:
    """
    Return the dimension-adaptive default martingale learning rate.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    beta = 0.5 + 2 / (1.1 * d + 4.0)
    C = 2 ** beta
    return C * (i + 1) ** (-beta)

def build_design(
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    n_basis: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    d = X_fit.shape[1]
    blocks_f, blocks_e = [], []
    
    for j in range(d):
        st = SplineTransformer(degree = degree, n_knots = n_basis - degree + 1,
                               include_bias = False)
        st.fit(X_fit[:, [j]])
        Bf = st.transform(X_fit[:, [j]])
        Be = st.transform(X_eval[:, [j]])
        cm = Bf.mean(0, keepdims = True)
        blocks_f.append(Bf - cm)
        blocks_e.append(Be - cm)

    Z_f = np.c_[np.ones(len(X_fit)), np.hstack(blocks_f)]
    Z_e = np.c_[np.ones(len(X_eval)), np.hstack(blocks_e)]

    return Z_f, Z_e

def quantize(
    f_train: np.ndarray,
    f_test: np.ndarray,
    disc_classes: int = 10,
    noise_scale: float = 0.5,
    seed: int,
)-> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    rng = np.random.default_rng(seed)
    f = np.concatenate([f_train, f_test])
    thresholds = np.linspace(f.min(), f.max(), disc_classes  + 1)[1:-1]
    eps = rng.logistic(0, noise_scale, f_train.shape[0])
    y_train = np.digitize(f_train + eps, thresholds) + 1
    f_test_class = np.digitize(f_test, thresholds) + 1

    return y_train, f_test_class

def disc_martingale_cis(results, y_grid, q = 0.5, lo = 5.0, hi = 95.0):
    y1d = np.asarray(y_grid).ravel()
    out = []
    for entry in results:
        cdf = entry['cdf']
        qb = y1d[np.clip([np.searchsorted(cdf[:,b], q, side = 'right') for b in range(cdf.shape[1])], 
                         0, len(y1d) - 1)]
        out.append(np.percentile(qb, [lo, hi], method = 'inverted_cdf'))
    return np.stack(out)

def cs_metrics(cs1: np.ndarray, cs2: np.ndarray, class_width: float = 1.0) -> dict:
    if class_width <= 0:
        raise ValueError(f'class_width must be positive, got {class_width} instead.')
    lo1, hi1 = cs1
    lo2, hi2 = cs2

    lo1 = np.asarray(lo1)
    lo2 = np.asarray(lo2)
    hi1 = np.asarray(hi1)
    hi2 = np.asarray(hi2)

    cardin1 = ((hi1 - lo1) // class_width) + 1
    cardin2 = ((hi2 - lo2) // class_width) + 1

    inter_card = ((np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), a_min = 0, a_max = None)) // class_width) + 1
    union_card = (np.clip(np.maximum(hi1, hi2) - np.minimum(lo1, lo2), a_min = 0, a_max = None) // class_width) + 1

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        iou = inter_card / union_card
        overlap = inter_card / cardin1
        card_diff = np.abs(cardin1 - cardin2) / cardin1
        lowdiff = (np.abs(lo1 - lo2) // class_width) / cardin1
        highdiff = (np.abs(hi1 - hi2) // class_width) / cardin1

    bounds = flags[str(K)]
    low_iou = (iou < bounds['iou']).mean()
    low_overlap = (overlap < bounds['overlap']).mean()
    high_carddiff = (card_diff > bounds['card_diff']).mean()
    high_lowdiff = (lowdiff > bounds['bound_diff']).mean()
    high_highdiff = (highdiff > bounds['bound_diff']).mean()
    overconf = (cardin1 > cardin2).mean()

    return {'iou': iou, 'overlap': overlap,
           'card_diff': card_diff, 'low_diff': lowdiff,
           'high_diff': highdiff,
           'failures':{'iou': low_iou, 'overlap': low_overlap,
                      'card_diff': high_carddiff, 'low_diff': high_lowdiff,
                      'high_diff': high_highdiff, 'over_confidence': overconf,}}


@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    yield
    print(f'   [{name}] {time.perf_counter() - t0:.2f}s', flush = True)
    
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(uqpfn.__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_reps",  type=int,   default=10,   help="Outer repetitions")
    p.add_argument("--n_test",  type=int,   default=200,  help="Test set size")
    p.add_argument("--B",       type=int,   default=100,  help="Martingale posterior draws")
    p.add_argument("--T",       type=int,   default=200,  help="Martingale forward steps")
    p.add_argument("--N",       type=int,   default=1000, help="TabPFN forward samples")
    p.add_argument("--n_train", type=int,   default=100)
    p.add_argument("--d_true",  type=int,   default=5)
    p.add_argument("--d_total", type=int,   default=10)
    p.add_argument("--sigma",   type=float, default=1.0)
    p.add_argument("--tau",     type=float, default=1.0)
    p.add_argument("--n_basis", type=int,   default=20)
    p.add_argument("--degree",  type=int,   default=3)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out",     default= Path('tests') / "artifacts" / "cont_approx.csv")
    return p.parse_args(args = [])

def main():
    args = parse_args()

    from uqpfn import tabicl

    class_number = [20, 50, 100, 400]
    sigma = {'20': 0.6, '50': 0.6, '100': 0.6, '400': 0.8}
    flags = {
        '20': {'iou': 0.9, 'overlap': 1.0, 'card_diff': 0.1, 'bound_diff':0.075},
        '50':  {'iou': 0.94, 'overlap': 1.0, 'card_diff': 0.05, 'bound_diff':0.04},
        '100':  {'iou': 0.97, 'overlap' :0.98, 'card_diff': 0.03, 'bound_diff':0.025},
        '400':  {'iou': 0.98, 'overlap': 0.99, 'card_diff': 0.02, 'bound_diff': 0.015},
    }
    train_sizes = [50, 100, 200]
    train_max = 1000
    features_number = [5, 20]
    records = []

    for features in features_number:
        true_features = features #have J = d

        for rep in range(args.n_reps):
            print(f'\n=== Rep {rep + 1}/{args.n_reps} for {features} features ===', flush = True)
            rng = np.random.default_rng(args.seed + rep * 432)

            #generate dataset
            X_tr_raw = rng.standard_normal((train_max * 3, features))
            x1 = X_tr_raw[:, 0]
            mask = ~(
                ((x1 >= -1.2) & (x1 <= -0.9)) |
                ((x1 >= 0.9) & (x1 <= 1.2))
            )
            X_train = X_tr_raw[mask][: train_max]

            X_test = rng.standard_normal((args.n_test, features))
    
            Z_tr, Z_te = build_design(
                X_train[:, : true_features],
                X_test[:, : true_features],
                args.n_basis,
                args.degree,
            )
            p = Z_tr.shape[1]
        
            theta_true = args.tau * rng.standard_normal(p)
            theta_true[0] = 0.0
            f_test = Z_te @ theta_true #get true vals for coverage assessment
            f_train = Z_tr @ theta_true 

            for disc_class in class_number:
                #quantize the outcome
                new_y_grid = np.linspace(1, disc_class, disc_class)
                y_train_class, _ = quantize(y_train, f_test, 
                                            disc_classes = disc_class,
                                            noise_scale = sigma[str(disc_class)]
                                            seed = args.seed + rep * 432
                                           )
                
                for sample_size in train_sizes:
                    #subset the data
                    if disc_class < 50:
                        X_train_now, _, y_train_now, _ = train_test_split(
                            X_train, y_train_class, train_size = sample_size, 
                            stratify = y_train_class,
                            random_state = args.seed + rep * 432
                        )
                    else:
                        X_train_now, _, y_train_now, _ = train_test_split(
                            X_train, y_train_class, train_size = sample_size, 
                            random_state = args.seed + rep * 432
                        )

                    #get TabICL estimate from the quantized dataset
                    with timed('CDF'):
                        model = tabicl.TabICLQuantileRegressor(n_estimators = 4)
                        model.fit(X_train_now, y_train_now)
                        cdf_arr = model.predict_cdf(X_test, new_y_grid, np.linspace(0.001, 0.999, 999))
                        cdf_arr[-1,:] = 1.0
    
                    #get continuous AMP
                    with timed('get continuous AMP'):
                        cont_res = get_posterior(
                            new_y_grid, sample_size, args.B, args.T, cdf_arr,
                            discrete = False, alpha_dimension = features,
                        )
        
                    #get discrete AMP
                    with timed('get discrete AMP'):
                        disc_res = get_posterior(
                            new_y_grid, sample_size, args.B, args.T, cdf_arr,
                            discrete = True, alpha_dimension = features,
                        )

                    with timed('calculating results...'):
                        #get confidence intervals
                        cont_cis = disc_martingale_cis(cont_res, new_y_grid)
                        disc_cis = disc_martingale_cis(disc_res, new_y_grid)
        
                        cont_lo, cont_hi = cont_cis[:, 0], cont_cis[:, 1]
                        disc_lo, disc_hi = disc_cis[:, 0], disc_cis[:, 1]
        
                        #evaluate differences
                        diffs = cs_metrics([disc_lo, disc_hi], [cont_lo, cont_hi])
                        
                        records.append({'K': disc_class, 'n': sample_size, 
                                        'd': features, 'rep': rep, 
                                       })
                        for entry, val in diffs.items():
                            if entry != 'failures':
                                records[len(records) - 1][entry] = val.mean()
                            else:
                                for failed_entry, summary in val.items():
                                    records[len(records) - 1][f'{failed_entry}_fails'] = summary

            df = pd.DataFrame(records)
            df.to_csv(args.out, index = False)
            summ = df.groupby(['K', 'n'])[[
                'iou', 'overlap', 'card_diff',
                'low_diff', 'high_diff',
                'iou_fails', 'overlap_fails',
                'card_diff_fails', 'low_diff_fails',
                'high_diff_fails',
                'over_confidence_fails', 
            ]].mean()
            print(f'\n Summary after rep {rep + 1}:\n{summ}\n', flush = True)

        df = pd.DataFrame(records)
        df.to_csv(args.out, index = False)
        print('\n' + '=' * 70)
        print('FINAL SUMMARY (mean over reps)')
        print('=' * 70)
        print(df.groupby(['K', 'n'])[[
             'iou', 'overlap', 'card_diff',
                'low_diff', 'high_diff',
                'iou_fails', 'overlap_fails',
                'card_diff_fails', 'low_diff_fails',
                'high_diff_fails',
                'over_confidence_fails',
        ]].mean().to_string())
        print(f'operation finished for {features} number of features')

    print(f'operation completed. results saved in {args.out}')

if __name__ == '__main__':
    main()
