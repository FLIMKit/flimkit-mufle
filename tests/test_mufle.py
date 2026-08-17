import numpy as np

from flimkit_mufle.synth import syntheticSpectralTemporal
from flimkit_mufle.model import fitMufle


def _correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def test_recovers_two_components():
    data, wl, res, irf, truth = syntheticSpectralTemporal(seed=1)
    out = fitMufle(data, wl, res, irf, n_components=2)
    assert out['success']

    order = np.argsort(truth['taus'])
    tru_taus = truth['taus'][order]
    tru_spectra = truth['spectra'][order]

    rel = np.abs(out['taus'] - tru_taus) / tru_taus
    assert np.all(rel < 0.15), (out['taus'], tru_taus, rel)

    for k in range(2):
        corr = _correlation(out['spectra'][k], tru_spectra[k])
        assert corr > 0.95, (k, corr)

    assert out['chi2_reduced'] < 2.0, out['chi2_reduced']


def test_single_component_recovers_tau():
    comp = ({'tau_ns': 2.0, 'peak_nm': 500.0, 'width_nm': 40.0, 'height': 1.0},)
    data, wl, res, irf, truth = syntheticSpectralTemporal(components=comp, seed=2)
    out = fitMufle(data, wl, res, irf, n_components=1)
    rel = abs(out['taus'][0] - truth['taus'][0]) / truth['taus'][0]
    assert rel < 0.1, (out['taus'][0], truth['taus'][0])


def test_array_accessor_passthrough():
    from flimkit_mufle.accessor import spectral_stack
    data, wl, res, irf, truth = syntheticSpectralTemporal(seed=3)
    out_wl, stack, tcspc = spectral_stack(data, wavelengths=wl)
    assert stack.shape == data.shape
    assert tcspc is None
    assert np.allclose(out_wl, wl)
