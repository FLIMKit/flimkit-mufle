# flimkit-mufle

Spectral-temporal fluorescence unmixing for FLIMKit, in the style of MuFLE
(Adams et al., IEEE TBME 2023; Biomed. Opt. Express 17(4):2176, 2026). It takes a
multi-channel time-resolved emission dataset, fits it with a small number of
components where each component is an emission spectrum plus a single fluorescence
lifetime, and returns the unmixed components.

## What it does

Input is a stack of decays, one per wavelength channel, shape `(n_channels, n_bins)`.
Each component is a smooth emission spectrum (cubic B-splines over wavelength)
multiplied by an IRF-reconvolved single-exponential decay. Lifetimes are fitted by
non-linear least squares. The spectral amplitudes are solved by non-negative least
squares inside each step (variable projection), so spectra come out non-negative
without extra constraints. Poisson weighting is used throughout.

## Install

```
pip install -e .        # from this folder
pip install -e .[gui]   # adds matplotlib for the results window
```

FLIMKit discovers it through the `flimkit.plugins` entry point, so there's nothing
else to wire up. On start it appears under Tools > Unmixing.

## In the GUI

Tools > Unmixing > Spectral-Temporal Unmixing... picks a multi-channel file, asks
how many components, runs the fit and shows the emission spectra and lifetimes.

Tools > Unmixing > Synthetic Unmixing Demo runs the whole thing on generated data
with a known answer, so you can check it works without a file.

## From code

```python
from flimkit_mufle import fitMufle, unmixFile, spectral_stack

result = unmixFile('scan.ptu', n_components=2)
print(result['taus'], result['chi2_reduced'])
```

`fitMufle(data, wavelengths, tcspc_res, irf, n_components, ...)` is the core. It
returns a dict with `taus`, `spectra`, `amplitudes`, `background`, `model`,
`residual` and `chi2_reduced`.

`spectral_stack(source)` pulls a per-channel decay stack from any FLIMKit reader,
or passes a numpy array straight through.

## Validated

On synthetic data with two components at 0.5 ns and 3.0 ns the fit recovers 0.498 ns
and 2.997 ns, spectral correlation 0.9999 and 1.0, reduced chi-square 1.05. See
`tests/`.

## Known limits

Channel indexing follows the reader. PTU channels are 0-based, Becker and Hickl are
1-based, so pass `channels=` if the defaults pick the wrong ones.

The IRF for a real file currently defaults to a Gaussian placeholder. Wiring it to
FLIMKit's IRF tools, and to a per-channel IRF, is the next step.

Wavelength calibration is not read from file metadata yet. Without it the channel
index is used as the axis.

It fits one summed spectrum at a time, a point or an ROI. Per-pixel spectral-temporal
maps would need the GPU path.
