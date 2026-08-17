import numpy as np


def _open(source):
    if hasattr(source, 'summed_decay'):
        return source
    from flimkit.formats import FLIMFile
    return FLIMFile(str(source))


def spectral_stack(source, channels=None, wavelengths=None):
    if isinstance(source, np.ndarray):
        stack = np.asarray(source, dtype=float)
        if stack.ndim != 2:
            raise ValueError(f'array source must be (n_channels, n_bins), got {stack.shape}')
        wl = np.arange(stack.shape[0], dtype=float) if wavelengths is None else np.asarray(wavelengths, float)
        return wl, stack, None
    reader = _open(source)
    n_chan = int(getattr(reader, 'n_channels', 0))
    if n_chan < 1:
        raise ValueError(f'reader {reader!r} reports no channels')
    if channels is None:
        channels = list(range(n_chan))
    rows = [np.asarray(reader.summed_decay(channel=c), dtype=float) for c in channels]
    widths = {r.shape[0] for r in rows}
    if len(widths) != 1:
        raise ValueError(f'channels returned decays of differing lengths: {sorted(widths)}')
    stack = np.vstack(rows)
    tcspc_res = float(getattr(reader, 'tcspc_res', 0.0))
    if wavelengths is None:
        wl = np.asarray(channels, dtype=float)
    else:
        wl = np.asarray(wavelengths, dtype=float)
    return wl, stack, tcspc_res


def default_gaussian_irf(n_bins, tcspc_res, centre_frac=0.08, width_frac=0.02):
    from flimkit_mufle.synth import gaussian_irf
    t_ns = np.arange(n_bins, dtype=float) * tcspc_res * 1e9
    span = t_ns[-1] if t_ns[-1] > 0 else float(n_bins)
    return gaussian_irf(t_ns, centre_frac * span, max(width_frac * span, 1e-6))


def unmixFile(source, n_components, irf=None, channels=None, wavelengths=None, **fit_kwargs):
    from flimkit_mufle.model import fitMufle
    wl, stack, tcspc_res = spectral_stack(source, channels=channels, wavelengths=wavelengths)
    if tcspc_res is None:
        raise ValueError('array sources have no tcspc_res; call fitMufle directly with one')
    if irf is None:
        irf = default_gaussian_irf(stack.shape[1], tcspc_res)
    return fitMufle(stack, wl, tcspc_res, irf, n_components, **fit_kwargs)
