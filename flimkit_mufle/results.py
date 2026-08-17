import numpy as np


def _summary_lines(result, truth):
    lines = [f"chi2_red = {result['chi2_reduced']:.3f}   "
             f"success = {result['success']}"]
    taus = result['taus'] * 1e9
    amps = result['amplitudes']
    amp_total = amps.sum() if amps.sum() > 0 else 1.0
    for k in range(result['n_components']):
        frac = 100.0 * amps[k] / amp_total
        line = f'  component {k + 1}: tau = {taus[k]:.3f} ns, abundance = {frac:.1f}%'
        if truth is not None and k < len(truth['taus']):
            line += f"   (truth tau = {truth['taus'][k] * 1e9:.3f} ns)"
        lines.append(line)
    return '\n'.join(lines)


def show_results(parent, result, truth=None, title='MuFLE'):
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    win = tk.Toplevel(parent)
    win.title(title)

    fig = Figure(figsize=(7, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    wl = result['wavelengths']
    for k, spectrum in enumerate(result['spectra']):
        tau_ns = result['taus'][k] * 1e9
        ax.plot(wl, spectrum, label=f'component {k + 1} ({tau_ns:.2f} ns)')
    if truth is not None:
        for spectrum in truth['spectra']:
            ax.plot(truth['wavelengths'], spectrum, '--', color='0.6', linewidth=1)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('emission amplitude')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    text = tk.Text(win, height=result['n_components'] + 2, width=70)
    text.insert('1.0', _summary_lines(result, truth))
    text.configure(state='disabled')
    text.pack(fill='x')
    return win
