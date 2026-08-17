import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, nnls


def bspline_basis(wavelengths, n_splines, degree=3):
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError(f'wavelengths must be 1D with >= 2 points, got shape {wl.shape}')
    if n_splines < degree + 1:
        raise ValueError(f'n_splines {n_splines} must be at least degree+1 = {degree + 1}')
    lo = float(wl.min())
    hi = float(wl.max())
    n_interior = n_splines - degree - 1
    if n_interior > 0:
        interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    else:
        interior = np.array([], dtype=float)
    knots = np.concatenate((np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)))
    return BSpline.design_matrix(np.clip(wl, lo, hi), knots, degree).toarray()


def reconvolve(decay, irf):
    return np.convolve(decay, irf)[:decay.shape[0]]


def poisson_weights(data):
    return 1.0 / np.sqrt(np.maximum(data, 1.0))


def _trapz(y, x):
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return integrate(y, x)


def temporal_templates(taus, t, irf):
    out = np.empty((len(taus), t.shape[0]), dtype=float)
    for k, tau in enumerate(taus):
        out[k] = reconvolve(np.exp(-t / max(float(tau), 1e-15)), irf)
    return out


def _design(templates, basis, fit_slice, add_bg):
    tpl = templates[:, fit_slice]
    n_comp = templates.shape[0]
    n_spl = basis.shape[1]
    cols = []
    for comp in range(n_comp):
        for j in range(n_spl):
            cols.append(np.outer(basis[:, j], tpl[comp]).ravel())
    a = np.stack(cols, axis=1)
    if add_bg:
        a = np.hstack((a, np.ones((a.shape[0], 1))))
    return a


def _nnls_maxiter(n_cols):
    return max(500, 5 * n_cols)


def fitMufle(data, wavelengths, tcspc_res, irf, n_components,
             n_splines=8, tau_bounds=(1e-10, 1e-8), fit_start=0, fit_end=None,
             fit_background=True, spline_degree=3, max_nfev=300):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f'data must be (n_channels, n_bins), got shape {data.shape}')
    n_chan, n_bins = data.shape
    wl = np.asarray(wavelengths, dtype=float)
    if wl.shape != (n_chan,):
        raise ValueError(f'wavelengths shape {wl.shape} does not match n_channels {n_chan}')
    if n_components < 1:
        raise ValueError(f'n_components must be >= 1, got {n_components}')

    t = np.arange(n_bins, dtype=float) * float(tcspc_res)
    irf = np.asarray(irf, dtype=float)
    total = irf.sum()
    if total > 0:
        irf = irf / total

    basis = bspline_basis(wl, n_splines, spline_degree)
    fit_end = n_bins if fit_end is None else fit_end
    fit_slice = slice(fit_start, fit_end)
    d_fit = data[:, fit_slice]
    w2 = poisson_weights(d_fit)
    w = w2.ravel()
    d_w = d_fit.ravel() * w

    lo = np.log10(tau_bounds[0])
    hi = np.log10(tau_bounds[1])
    if n_components == 1:
        p0 = np.array([0.5 * (lo + hi)])
    else:
        p0 = np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), n_components)

    state = {}

    def residual(log_taus):
        taus = 10.0 ** log_taus
        templates = temporal_templates(taus, t, irf)
        a = _design(templates, basis, fit_slice, fit_background)
        a_w = a * w[:, None]
        coeffs, _ = nnls(a_w, d_w, maxiter=_nnls_maxiter(a_w.shape[1]))
        state['a'] = a
        state['coeffs'] = coeffs
        return a_w @ coeffs - d_w

    fit = least_squares(residual, p0, bounds=(lo, hi), max_nfev=max_nfev)

    taus = 10.0 ** fit.x
    order = np.argsort(taus)
    taus = taus[order]

    coeffs = state['coeffs']
    spec_coeffs = coeffs[:n_components * n_splines].reshape(n_components, n_splines)[order]
    spectra = spec_coeffs @ basis.T
    bg = float(coeffs[-1]) if fit_background else 0.0

    model_fit = (state['a'] @ coeffs).reshape(n_chan, d_fit.shape[1])
    resid = d_fit - model_fit
    n_params = n_components * n_splines + n_components + (1 if fit_background else 0)
    dof = max(d_fit.size - n_params, 1)
    chi2 = float(np.sum((resid * w2) ** 2))
    amplitudes = np.array([_trapz(np.maximum(s, 0.0), wl) for s in spectra])

    return {
        'taus': taus,
        'spectra': spectra,
        'amplitudes': amplitudes,
        'background': bg,
        'wavelengths': wl,
        'model': model_fit,
        'residual': resid,
        'fit_slice': fit_slice,
        'chi2': chi2,
        'chi2_reduced': chi2 / dof,
        'n_components': n_components,
        'success': bool(fit.success),
        'message': fit.message,
    }
