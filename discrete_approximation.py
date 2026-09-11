from __future__ import annotations

import numpy as np
import pyvinecopulib as pv

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
