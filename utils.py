import matplotlib.pyplot as plt
import os
from scipy import interpolate
import numpy as np

def read_dat_txt(fname, wl_range):
    try:
        T = np.genfromtxt(fname)
        i_start = np.abs(T[:,0]-wl_range[0]).argmin()
        i_stop = np.abs(T[:,0]-wl_range[1]).argmin()
        return T[i_start:i_stop,0], T[i_start:i_stop,1], T[i_start:i_stop,2]
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None, None, None

def read_nk(fname, wl_interp):
    try:
        e12 = np.genfromtxt(fname)
        f_r = interpolate.interp1d(e12[:, 0], e12[:, 1], kind='cubic')
        f_i = interpolate.interp1d(e12[:, 0], e12[:, 2], kind='cubic')
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
            if i > 3 and (i % 3) == 0 and int(float(T[0])) > wl_min and int(float(T[0])) < wl_max and int(float(T[1])) == AOI:
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
    e12[:, 0] = hc/e12[:, 0] # E (eV) = 1239.8 / wavelength(nm)  
    if e12[1, 0] < e12[0, 0]: #if values are decreasing flip order about 0 axis
        e12 = np.flipud(e12)
    N_wl = np.size(wl_interp, 0)
    wl = e12[:, 0].copy()
    e_r = e12[:, 1].copy()
    e_i = e12[:, 2].copy()
    
    e = np.zeros(N_wl, dtype="complex")
    n = np.zeros(N_wl, dtype="complex")
    
    f_r = interpolate.interp1d(wl, e_r, kind='cubic')
    f_i = interpolate.interp1d(wl, e_i, kind='cubic')
    e.real = f_r(wl_interp)
    e.imag = f_i(wl_interp)
    n = e**.5
    return n


def ema_model_func(e1, e2, fv):
    """
    DIELECTRIC FUNCTION FROM THE EFFECTIVE MEDIUM APPROXIMATION
    fv: volume fraction ratio
    """
    p = (e1 / e2) ** .5
    b = ((3 * fv - 1) * (1 / p - p) + p) / 4
    z = b + (b ** 2 + .5) ** .5
    e = z * (e1 * e2) ** .5
    return e



def psi_delta_to_NCS(psi, delta):
    psi = np.asarray(psi)
    delta = np.asarray(delta)
    
    N_t = np.cos(2 * psi)
    C_t = np.sin(2 * psi) * np.cos(delta)
    S_t = np.sin(2 * psi) * np.sin(delta)
    
    if N_t.size == 1:
        return N_t.item(), C_t.item(), S_t.item()
    else:
        return N_t, C_t, S_t


def plot_results(wl, N_m, C_m, S_m, N_c, C_c, S_c):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Ellipsometry")

    axes[0].plot(wl, N_m, "bo", wl, N_c, "r-")
    axes[0].set_xlabel(r"wavelength (nm)")
    axes[0].set_ylabel("N")
    axes[0].grid(True)
    
    axes[1].plot(wl, C_m, "bo", wl, C_c, "r-")
    axes[1].set_xlabel(r"wavelength (nm)")
    axes[1].set_ylabel("C")
    axes[1].grid(True)
    
    axes[2].plot(wl, S_m, "bo", wl, S_c, "r-")
    axes[2].set_xlabel(r"wavelength (nm)")
    axes[2].set_ylabel("S")
    axes[2].grid(True)
    
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt_file = os.path.join(os.getcwd(), 'ellipsometry.png')
    plt.savefig(plt_file)
    


def plot_layer_structure(material_layers, params, substrate):
    fig, ax = plt.subplots(figsize=(7, 8))
    y_offset = 0
    layer_rectangles = []
    layer_labels = []
    layer_descriptions = []

    substrate_thickness = 50  # Arbitrary value for illustration
    substrate_label = f"Substrate ({substrate})"
    substrate_rect = plt.Rectangle((0, y_offset), 1, substrate_thickness, edgecolor='black', facecolor='lightblue')
    layer_rectangles.append(substrate_rect)
    layer_labels.append("Substrate")
    layer_descriptions.append(substrate_label)
    y_offset += substrate_thickness

    for layer in reversed(material_layers):
        if layer['type'] == 'composite':
            material1_name = layer['material1']
            material2_name = layer['material2']
            composite_name = f"{material1_name}_{material2_name}_{layer['name']}"
            d_param_name = f"d_{composite_name}"
            fv_param_name = f"fv_{composite_name}"
            thickness = params[d_param_name].value
            fv = params[fv_param_name].value
            d_fit_status = "fit completed" if params[d_param_name].vary else "fixed"
            fv_fit_status = "fit completed" if params[fv_param_name].vary else "fixed"
            label = f"{layer['name']} (Composite)"
            description = (f"{material1_name}/{material2_name}\n"
                           f"d: {thickness:.2f} nm ({d_fit_status})\n"
                           f"fv: {fv:.2f} ({fv_fit_status})")
        elif layer['type'] == 'single':
            material_name = layer['material']
            d_param_name = f"d_{material_name}_{layer['name']}"
            thickness = params[d_param_name].value
            d_fit_status = "fit completed" if params[d_param_name].vary else "fixed"
            label = f"{layer['name']} (Single)"
            description = (f"{material_name}\n"
                           f"d: {thickness:.2f} nm ({d_fit_status})")

        rect = plt.Rectangle((0, y_offset), 1, thickness, edgecolor='black', facecolor='lightgray')
        layer_rectangles.append(rect)
        layer_labels.append(label)
        layer_descriptions.append(description)
        y_offset += thickness

    total_height = y_offset

    for rect in layer_rectangles:
        ax.add_patch(rect)

    for rect, label in zip(layer_rectangles, layer_labels):
        ax.text(0.5, rect.get_y() + rect.get_height() / 2, label, ha='center', va='center', fontsize=10)

    angle_value = params["angle"].value
    angle_status = "fit completed" if params["angle"].vary else "fixed"
    angle_description = f"Angle: {angle_value:.2f}° ({angle_status})"
    layer_labels.append("Angle")
    layer_descriptions.append(angle_description)

    plt.xlim(0, 1)
    plt.ylim(0, total_height + 10)
    plt.axis('off')
    plt.title('Layer Structure')

    legend_labels = [plt.Line2D([0], [0], color='black', lw=4, label=f"{label}\n{description}") for label, description in reversed(list(zip(layer_labels, layer_descriptions)))]
    legend = ax.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=False, labelspacing=1.2) 

    plt.tight_layout()
    plt.savefig('layer_structure_with_legend.png', bbox_inches='tight')
    
