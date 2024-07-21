import matplotlib.pyplot as plt
import numpy as np
import lmfit
from time import time, strftime, localtime
import os
from src.plot import plot_layer_structure, plot_results
from src.config import load_configuration, validate_config
from src.ellipsometry import set_fit_parameters, psi_delta, psi_delta_to_NCS
from src.file_utils import read_dat_txt, readeps


def generate_execution_dir_name(start_time, material_layers, substrate):
    timestamp = strftime("%Y%m%d_%H%M%S", start_time)
    materials = "_".join(
        [
            (
                f"{layer['material1']}_{layer['material2']}"
                if layer["type"] == "composite"
                else layer["material"]
            )
            for layer in material_layers
        ]
    )
    return f"{timestamp}_{materials}_substrate_{substrate}"


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
                n_material = readeps(material_config["file"], wl)
                e_material = n_material**2
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

        NCS_c = NCS_m + result.residual
        NCS3 = len(wl)
        N_c, C_c, S_c = NCS_c[:NCS3], NCS_c[NCS3 : 2 * NCS3], NCS_c[2 * NCS3 :]
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    try:
        parent_dir = "results"
        os.makedirs(parent_dir, exist_ok=True)
        execution_dir = os.path.join(
            parent_dir,
            generate_execution_dir_name(
                start_time, config["material_layers"], config["substrate"]
            ),
        )
        os.makedirs(execution_dir, exist_ok=True)

        fit_report_path = os.path.join(execution_dir, "fit_report.txt")
        with open(fit_report_path, "w") as f:
            f.write(lmfit.fit_report(result))
            f.write(f"\n[[Latency]]\n    time: {t:.6f} seconds\n")
            f.write(f"    fit started: {t_start}\n")
            f.write(f"    fit ended: {t_end}\n")

        plot_results(wl, N_m, C_m, S_m, N_c, C_c, S_c)
        plt.savefig(os.path.join(execution_dir, "ellipsometry_results.png"))

        plot_layer_structure(
            config["material_layers"], result.params, config["substrate"]
        )
        plt.savefig(os.path.join(execution_dir, "layer_structure.png"))

        print(f"Results saved to {execution_dir}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping the program.")
    except Exception as e:
        print(f"An error occurred while plotting the results: {e}")


if __name__ == "__main__":
    try:
        config = load_configuration("config.json")
        validate_config(config)
        main(config)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file path.")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(
            f"An error occurred while loading the configuration or running the main function: {e}"
        )
