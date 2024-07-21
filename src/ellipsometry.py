import psutil
import numpy as np
from tmm import ellips
from joblib import Parallel, delayed

_cached_num_free_cpus = None


def _get_free_cpus(threshold=50):
    global _cached_num_free_cpus
    if _cached_num_free_cpus is None:
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        free_cpus = [
            i for i, usage in enumerate(cpu_usage) if usage < threshold
        ]  # <50% usage is free
        _cached_num_free_cpus = len(free_cpus)
    return _cached_num_free_cpus


def compute_ellips(params, i, wl, n_layers, d_layers):
    angle = np.deg2rad(params["angle"])
    PD = ellips(n_layers, d_layers, angle, wl[i])
    return PD


def compute_ellips_parallel(args):
    params, i, wl, material_layers, materials, substrate = args
    n_layers = [1]  # n_Air
    d_layers = [np.inf]  # Semi-infinite Air

    for layer in material_layers:
        if layer["type"] == "composite":
            material1_name = layer["material1"]
            material2_name = layer["material2"]
            e_material1 = materials[material1_name]["e"][i]
            e_material2 = (
                materials[material2_name]["e"]
                if material2_name == "void"
                else materials[material2_name]["e"][i]
            )
            composite_name = f"{material1_name}_{material2_name}_{layer['name']}"
            fv = float(params[f"fv_{composite_name}"])
            e_eff = ema_model_func(e_material1, e_material2, fv) ** 0.5
        elif layer["type"] == "single":
            material_name = layer["material"]
            e_eff = materials[material_name]["e"][i] ** 0.5

        d_param_name = (
            f"d_{composite_name}"
            if layer["type"] == "composite"
            else f"d_{material_name}_{layer['name']}"
        )
        d = float(params[d_param_name])
        n_layers.append(e_eff)
        d_layers.append(d)

    n_layers.append(materials[substrate]["n"][i])  # Substrate
    d_layers.append(np.inf)  # Semi-infinite substrate

    return compute_ellips(params, i, wl, n_layers, d_layers)


def psi_delta(
    params, wl, NCS_m, material_layers, materials, substrate, use_parallel=False
):
    free_cpus = _get_free_cpus()
    if use_parallel and free_cpus > 1:
        args = [
            (params, i, wl, material_layers, materials, substrate)
            for i in range(len(wl))
        ]
        PD = Parallel(n_jobs=free_cpus)(
            delayed(compute_ellips_parallel)(arg) for arg in args
        )
    else:
        PD = []
        for i in range(len(wl)):
            PD.append(
                compute_ellips_parallel(
                    (params, i, wl, material_layers, materials, substrate)
                )
            )

    P_t = np.array([pd["psi"] for pd in PD])
    D_t = np.pi - np.array([pd["Delta"] for pd in PD])
    N_t, C_t, S_t = psi_delta_to_NCS(P_t, D_t)
    return np.concatenate((N_t, C_t, S_t), axis=0) - NCS_m


def set_fit_parameters(params, fit_parameters):
    for name in params:
        if name in fit_parameters:
            params[name].vary = True
        else:
            params[name].vary = False


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
