import lmfit
from lmfit import fit_report
import time
import matplotlib.pyplot as plt
from ease_core import Ease
from layer import Layer, CompositeLayer, Void
import numpy as np
import utils


def fit(params, wavelengths, angle, n_Si, n_SiO2, data, use_parallel):
    d_Si = np.inf
    d_SiO2 = float(params["d_SiO2"])
    layer_Si = CompositeLayer()
    layer1 = Layer(n_Si, d_Si / 2, 0.5)
    layer2 = Layer(n_Si, d_Si / 2, 0.5)
    layer_Si.add_layers([layer1, layer2])
    layer_SiO2 = Layer(n_SiO2, d_SiO2)

    ease = Ease([Void(wavelengths), layer_SiO2, layer_Si], angle, wavelengths)
    N, C, S = ease.solve_ellipsometry(
        return_ncs=True, return_tuple=True, use_parallel=use_parallel
    )
    NCS = np.concatenate((N, C, S), axis=0)
    return NCS - data


def fit_and_plot(fit, fit_method, wl, angle, n_Si, n_SiO2, ncs, use_parallel=False):
    params = lmfit.Parameters()
    params.add("d_SiO2", value=15.0, min=10.0, max=30.0)
    minner = lmfit.Minimizer(fit, params, fcn_args=(wl, angle, n_Si, n_SiO2, ncs, use_parallel))
    
    t_start = time.time()
    result = minner.minimize(method=fit_method)
    t_fit = time.time() - t_start
    
    NCS = ncs + result.residual

    NCS_reshaped = NCS.reshape(-1, len(wl))
    ncs_reshaped = ncs.reshape(-1, len(wl))
    
    N = NCS_reshaped[0]
    C = NCS_reshaped[1]
    S = NCS_reshaped[2]
    
    n = ncs_reshaped[0]
    c = ncs_reshaped[1]
    s = ncs_reshaped[2]
    
    print(lmfit.fit_report(result))
    print(f"[[Latency]]\n    time: {t_fit:.2f} seconds")
    
    plot_results(wl, n, N, "N")
    plot_results(wl, c, C, "C")
    plot_results(wl, s, S, "S")
    
    Psi, Delta = utils.NCS_to_psi_delta(N, C, S)
    psi, delta = utils.NCS_to_psi_delta(n, c, s)

    plot_results(wl, psi, Psi, "Psi")
    plot_results(wl, delta, Delta, "Delta")


    plt.show()


def plot_results(wl, original_data, fitted_data, label):
    plt.figure()
    plt.plot(wl, original_data, "bo", label="Original")
    plt.plot(wl, fitted_data, "r-", label="Fitted")
    plt.xlabel('Wavelength')
    plt.ylabel(label)
    plt.legend()