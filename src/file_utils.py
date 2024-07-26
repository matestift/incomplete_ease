from scipy import interpolate
import numpy as np


def read_dat_txt(fname, wl_range):
    try:
        T = np.genfromtxt(fname)
        i_start = np.abs(T[:, 0] - wl_range[0]).argmin()
        i_stop = np.abs(T[:, 0] - wl_range[1]).argmin()
        return T[i_start:i_stop, 0], T[i_start:i_stop, 1], T[i_start:i_stop, 2]
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None, None, None


def read_nk(fname, wl_interp):
    try:
        e12 = np.genfromtxt(fname)
        f_r = interpolate.interp1d(e12[:, 0], e12[:, 1], kind="cubic")
        f_i = interpolate.interp1d(e12[:, 0], e12[:, 2], kind="cubic")
        return f_r(wl_interp) + 1j * f_i(wl_interp)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None


def read_dat_file(fname, AOI, wl_min, wl_max):
    """
    wl - wavelength
    wl_min - minimum wavelength
    wl_max - maximum wavelength
    AOI - incidence angle
    P - psi
    D - delta
    """
    T = []
    wl = []
    P = []
    D = []
    with open(fname) as f:
        i = -1
        for line in f:
            i = i + 1
            T = line.split()
            if (
                i > 3
                and (i % 3) == 0
                and int(float(T[0])) > wl_min
                and int(float(T[0])) < wl_max
                and int(float(T[1])) == AOI
            ):
                wl.append(float(T[0]))
                P.append(float(T[2]))
                D.append(float(T[3]))
    return np.array(wl), np.array(P), np.array(D)

def readeps(fname, wl_interp):
    """
    e12: matrix of wl, e1 and e2
    e: complex dielectric function
    n: complex refractive index
    readeps: function of reading optical properties from reference file
    h*c/q: planck's constant * speed of light / charge of electron
    q*V=h*nu=h*c/lambda > V=(h*c/q)/lambda=1239[J*nm/C]/lambda
    """
    e12 = np.genfromtxt(fname)
    hc = 1239
    e12[:, 0] = hc / e12[:, 0]  # E (eV) = 1239.8 / wavelength(nm)
    if e12[1, 0] < e12[0, 0]:  # if values are decreasing flip order about 0 axis
        e12 = np.flipud(e12)
    N_wl = np.size(wl_interp, 0)
    wl = e12[:, 0].copy()
    e_r = e12[:, 1].copy()
    e_i = e12[:, 2].copy()

    e = np.zeros(N_wl, dtype="complex")
    n = np.zeros(N_wl, dtype="complex")

    f_r = interpolate.interp1d(wl, e_r, kind="cubic")
    f_i = interpolate.interp1d(wl, e_i, kind="cubic")
    e.real = f_r(wl_interp)
    e.imag = f_i(wl_interp)
    n = e**0.5
    return n