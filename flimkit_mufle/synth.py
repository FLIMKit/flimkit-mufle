import numpy as np

from flimkit_mufle.model import reconvolve


def gaussian_irf(t, centre, width):
    g = np.exp(-0.5 * ((t - centre) / width) ** 2)
    total = g.sum()
    return g / total if total > 0 else g


def gaussian_spectrum(wavelengths, peak, width, height):
    return height * np.exp(-0.5 * ((wavelengths - peak) / width) ** 2)


DEFAULT_COMPONENTS = (
    {'tau_ns': 0.5, 'peak_nm': 460.0, 'width_nm': 35.0, 'height': 1.0},
    {'tau_ns': 3.0, 'peak_nm': 540.0, 'width_nm': 45.0, 'height': 0.8},
)


def syntheticSpectralTemporal(components=DEFAULT_COMPONENTS, n_channels=40,
                              wl_range=(420.0, 620.0), n_bins=256, tcspc_res_s=50e-12,
                              irf_centre_ns=0.6, irf_width_ns=0.12, photon_scale=4000.0,
                              background=5.0, seed=0):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(wl_range[0], wl_range[1], n_channels)
    t = np.arange(n_bins, dtype=float) * tcspc_res_s
    t_ns = t * 1e9
    irf = gaussian_irf(t_ns, irf_centre_ns, irf_width_ns)

    clean = np.zeros((n_channels, n_bins), dtype=float)
    truth_spectra = np.zeros((len(components), n_channels), dtype=float)
    truth_taus = np.zeros(len(components), dtype=float)
    for k, comp in enumerate(components):
        tau_s = comp['tau_ns'] * 1e-9
        spectrum = gaussian_spectrum(wavelengths, comp['peak_nm'],
                                     comp['width_nm'], comp['height'])
        decay = reconvolve(np.exp(-t / tau_s), irf)
        clean += photon_scale * np.outer(spectrum, decay)
        truth_spectra[k] = photon_scale * spectrum
        truth_taus[k] = tau_s

    clean += background
    noisy = rng.poisson(np.maximum(clean, 0.0)).astype(float)

    truth = {
        'taus': truth_taus,
        'spectra': truth_spectra,
        'wavelengths': wavelengths,
        'irf': irf,
        'background': background,
        'photon_scale': photon_scale,
    }
    return noisy, wavelengths, tcspc_res_s, irf, truth
