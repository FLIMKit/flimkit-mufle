from flimkit.plugins import tool, startup


def _parent(app):
    return getattr(app, 'root', None) or app


@startup(id='mufle_banner', order=50)
def announce(app):
    print('[MuFLE] spectral-temporal unmixing plugin ready (Tools > Unmixing)')


@tool(id='mufle_unmix_file', label='Spectral-Temporal Unmixing...', menu='Tools/Unmixing', order=10)
def unmix_file(app):
    from tkinter import filedialog, simpledialog, messagebox
    from flimkit_mufle.accessor import unmixFile
    from flimkit_mufle.results import show_results

    parent = _parent(app)
    path = filedialog.askopenfilename(
        parent=parent, title='Select a multi-channel FLIM file')
    if not path:
        return
    n_components = simpledialog.askinteger(
        'MuFLE', 'Number of components to unmix', parent=parent,
        minvalue=1, maxvalue=6, initialvalue=2)
    if not n_components:
        return
    try:
        result = unmixFile(path, n_components)
    except Exception as exc:
        messagebox.showerror('MuFLE failed',
                             f'{type(exc).__name__}: {exc}', parent=parent)
        return
    show_results(parent, result, title=f'MuFLE - {path}')


@tool(id='mufle_demo', label='Synthetic Unmixing Demo', menu='Tools/Unmixing', order=20)
def demo(app):
    from tkinter import messagebox
    from flimkit_mufle.synth import syntheticSpectralTemporal
    from flimkit_mufle.model import fitMufle
    from flimkit_mufle.results import show_results

    parent = _parent(app)
    data, wl, tcspc_res, irf, truth = syntheticSpectralTemporal()
    try:
        result = fitMufle(data, wl, tcspc_res, irf, n_components=len(truth['taus']))
    except Exception as exc:
        messagebox.showerror('MuFLE failed',
                             f'{type(exc).__name__}: {exc}', parent=parent)
        return
    show_results(parent, result, truth=truth, title='MuFLE - synthetic demo')
