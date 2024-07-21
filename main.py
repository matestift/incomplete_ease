import matplotlib.pyplot as plt
import psutil
import numpy as np
import lmfit
from utils import read_dat_txt, ema_model_func, psi_delta_to_NCS, readeps
from tmm import ellips
from time import time, strftime, localtime
import json
import os
from joblib import Parallel, delayed
from utils import plot_layer_structure, plot_results


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
        PD = Parallel(n_jobs=-1)(delayed(compute_ellips_parallel)(arg) for arg in args)
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


def load_configuration(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def set_fit_parameters(params, fit_parameters):
    for name in params:
        if name in fit_parameters:
            params[name].vary = True
        else:
            params[name].vary = False


def main(config):
    try:
        wl_range = config["wl_range"]
        wl, P_m, D_m = read_dat_txt(config["data_file"], wl_range)

        P_m = np.deg2rad(P_m)
        D_m = np.deg2rad(D_m)
        N_m, C_m, S_m = psi_delta_to_NCS(P_m, D_m)

        materials = {}
        for material_name, material_config in config["materials"].items():
            if material_name == "void":
                n_material = material_config["n"]
                e_material = material_config["e"]
            else:
                try:
                    n_material = readeps(material_config["file"], wl)
                    e_material = n_material**2
                except FileNotFoundError as e:
                    print(f"Error: {e}. Please check the file path.")
                    return
            materials[material_name] = {"n": n_material, "e": e_material}

        params = lmfit.Parameters()
        for layer in config["material_layers"]:
            if layer["type"] == "composite":
                material1_name = layer["material1"]
                material2_name = layer["material2"]
                composite_name = f"{material1_name}_{material2_name}_{layer['name']}"
                d_param_name = f"d_{composite_name}"
                fv_param_name = f"fv_{composite_name}"
                params.add(d_param_name, **layer["d"])
                params.add(fv_param_name, **layer["fv"])
            elif layer["type"] == "single":
                material_name = layer["material"]
                d_param_name = f"d_{material_name}_{layer['name']}"
                params.add(d_param_name, **layer["d"])

        params.add("angle", **config["angle"])

        set_fit_parameters(params, config["fit_parameters"])

        use_parallel = config.get("use_parallel", False)

        NCS_m = np.concatenate((N_m, C_m, S_m), axis=0)
        minner = lmfit.Minimizer(
            psi_delta,
            params,
            fcn_args=(
                wl,
                NCS_m,
                config["material_layers"],
                materials,
                config["substrate"],
                use_parallel,
            ),
        )

        start_time = localtime()
        t_start = strftime("%Y-%m-%d %H:%M:%S", start_time)
        t0 = time()
        result = minner.minimize(method=config["fit_method"])
        t1 = time()
        t_end = strftime("%Y-%m-%d %H:%M:%S", localtime())
        t = t1 - t0

        fit_report_path = os.path.join(os.getcwd(), "fit_report.txt")
        with open(fit_report_path, "w") as f:
            f.write(lmfit.fit_report(result))
            f.write(f"\n[[Latency]]\n    time: {t:.6f} seconds\n")
            f.write(f"    fit started: {t_start}\n")
            f.write(f"    fit ended: {t_end}\n")

        print(f"Fit report saved to {fit_report_path}")

        NCS_c = NCS_m + result.residual
        NCS3 = len(wl)
        N_c, C_c, S_c = NCS_c[:NCS3], NCS_c[NCS3 : 2 * NCS3], NCS_c[2 * NCS3 :]
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    try:
        plot_results(wl, N_m, C_m, S_m, N_c, C_c, S_c)
        plot_layer_structure(
            config["material_layers"], result.params, config["substrate"]
        )
        plt.show()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping the program.")
    except Exception as e:
        print(f"An error occurred while plotting the results: {e}")


if __name__ == "__main__":
    try:
        config = load_configuration("config.json")
        main(config)
    except Exception as e:
        print(
            f"An error occurred while loading the configuration or running the main function: {e}"
        )
