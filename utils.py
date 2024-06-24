import numpy as np
from joblib import Parallel, delayed
import psutil
from scipy import interpolate

_cached_num_free_cpus = None

def _get_free_cpus(threshold=50):
    global _cached_num_free_cpus
    if _cached_num_free_cpus is None:
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        free_cpus = [i for i, usage in enumerate(cpu_usage) if usage < threshold]
        _cached_num_free_cpus = len(free_cpus)
    return _cached_num_free_cpus

def read_ellipsometry_file(filename, angle_of_incidence, min_wavelength, max_wavelength):
    T = []
    wl = []
    P = []
    D = []
    with open(filename) as f:
        lines = f.readlines()
        for i in range(3, len(lines), 3):  # Start at line 3, step by 3 lines
            T = lines[i].split()
            wavelength = float(T[0])
            angle = float(T[1])
            psi = float(T[2])
            delta = float(T[3])
            

            if min_wavelength <= wavelength <= max_wavelength and np.isclose(angle, angle_of_incidence):
                wl.append(wavelength)
                P.append(psi)
                D.append(delta)
    wl = np.array(wl)
    P = np.array(P)
    D = np.array(D)
    D = D - 360 * (D > 180)
    return wl, P, D

def read_ellipsometry(filename, angle_of_incidence, min_wavelength, max_wavelength, return_ncs = False):
    wl, P, D = read_ellipsometry_file(filename, angle_of_incidence, min_wavelength, max_wavelength)
    N, C, S = psi_delta_to_NCS(np.deg2rad(P), np.deg2rad(D))
    if return_ncs: return wl, N, C, S
    return wl,P,D


def read_refractive_index(filename, wavelength_interpolation_range):
    e12 = np.genfromtxt(filename)
    hc = 1239
    e12[:, 0] = hc/e12[:, 0]
    if e12[1, 0] < e12[0, 0]:
        e12 = np.flipud(e12)
    
    wl = e12[:, 0].copy()
    e_r = e12[:, 1].copy()
    e_i = e12[:, 2].copy()
    

    f_r = interpolate.interp1d(wl, e_r, kind='cubic') # store corresponding e_r values for wl, i.e y in y=f(x)
    f_i = interpolate.interp1d(wl, e_i, kind='cubic')
    e_real_interp = f_r(wavelength_interpolation_range)
    e_imag_interp = f_i(wavelength_interpolation_range)
    
    e_interp = e_real_interp + 1j * e_imag_interp
    n_interp = np.sqrt(e_interp)

    if wavelength_interpolation_range.shape[0] != n_interp.shape[0]:
        raise ValueError("Mismatch in the dimensions of wavelength and refractive index arrays.")
    
    return np.column_stack((wavelength_interpolation_range, n_interp))




def _partial_ema(e, e_eff):
    """
    Helper function to compute part of the EMA sum for a given dielectric function and effective dielectric function.
    """
    return (e - e_eff) / (e + 2 * e_eff)

def ema(dielectric_functions, fractions, use_parallel=False):
    e_eff = dielectric_functions[0]
    dielectric_functions = np.array(dielectric_functions)
    fractions = np.array(fractions)

    if use_parallel:
        free_cpus = _get_free_cpus()
        if free_cpus == 0:
            use_parallel = False

    for _ in range(100):
        if use_parallel:
            num_workers = free_cpus
            results = Parallel(n_jobs=num_workers)(delayed(_partial_ema)(e, e_eff) for e in dielectric_functions)
        else:
            results = [_partial_ema(e, e_eff) for e in dielectric_functions]
        
        results = np.array(results)  # Convert to numpy array for broadcasting
        f_sum = np.sum(fractions[:, np.newaxis] * results, axis=0)
        e_eff_new = e_eff * (1 + 2 * f_sum / 3)

        if np.all(np.abs(e_eff_new - e_eff) < 1e-6):
            break
        e_eff = e_eff_new

    return e_eff


def psi_delta_to_NCS(psi, delta):
    psi = np.asarray(psi)
    delta = np.asarray(delta)
    
    N = np.cos(2 * psi)
    C = np.sin(2 * psi) * np.cos(delta)
    S = np.sin(2 * psi) * np.sin(delta)
    
    if N.size == 1:
        return N.item(), C.item(), S.item()
    else:
        return N, C, S
    


def NCS_to_psi_delta(N, C, S):
    N = np.asarray(N)
    C = np.asarray(C)
    S = np.asarray(S)
    
    psi = 0.5 * np.arctan2(S, C)
    delta = 0.5 * np.arcsin(S / np.sin(2 * psi))
    
    if psi.size == 1:
        return psi.item(), delta.item()
    else:
        return psi, delta