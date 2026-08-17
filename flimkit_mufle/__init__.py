FLIMKIT_PLUGIN_API = 1

from flimkit_mufle.model import fitMufle
from flimkit_mufle.synth import syntheticSpectralTemporal
from flimkit_mufle.accessor import spectral_stack, unmixFile

registered = False
register_error = None
try:
    from flimkit_mufle import plugin as _plugin
    registered = True
except ImportError as exc:
    register_error = exc

__all__ = [
    'FLIMKIT_PLUGIN_API',
    'fitMufle',
    'syntheticSpectralTemporal',
    'spectral_stack',
    'unmixFile',
]
