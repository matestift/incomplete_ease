import sys
import os
import platform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lmfit
from src.ellipsometry import psi_delta_to_NCS, set_fit_parameters, psi_delta
from src.file_utils import readeps, read_dat_txt
from src.config import load_configuration
import matplotlib.pyplot as plt
from time import time
import numpy as np

def print_system_info():
    print("System Information:")
    print(f"System: {platform.system()}")
    print(f"Node Name: {platform.node()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    print()

def run_experiments(config, num_runs=1):
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

        input_sizes = np.linspace(100, len(wl), num=num_runs, dtype=int)
        parallel_times = []
        sequential_times = []

        for size in input_sizes:
            wl_subset = wl[:size]

            # Parallel execution
            use_parallel = True
            minner = lmfit.Minimizer(
                psi_delta,
                params,
                fcn_args=(
                    wl_subset,
                    np.concatenate((N_m[:size], C_m[:size], S_m[:size]), axis=0),
                    config["material_layers"],
                    materials,
                    config["substrate"],
                    use_parallel,
                ),
            )
            t0_parallel = time()
            result_parallel = minner.minimize(method=config["fit_method"])
            t1_parallel = time()
            parallel_times.append(t1_parallel - t0_parallel)

            # Sequential execution
            use_parallel = False
            minner = lmfit.Minimizer(
                psi_delta,
                params,
                fcn_args=(
                    wl_subset,
                    np.concatenate((N_m[:size], C_m[:size], S_m[:size]), axis=0),
                    config["material_layers"],
                    materials,
                    config["substrate"],
                    use_parallel,
                ),
            )
            t0_sequential = time()
            result_sequential = minner.minimize(method=config["fit_method"])
            t1_sequential = time()
            sequential_times.append(t1_sequential - t0_sequential)

        return input_sizes, parallel_times, sequential_times

    except Exception as e:
        print(f"An error occurred during experiments: {e}")

def plot_time_complexity(input_sizes, parallel_times, sequential_times):
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, parallel_times, label="Parallel Execution")
    plt.plot(input_sizes, sequential_times, label="Sequential Execution")
    plt.xlabel("Input Size (number of wavelengths)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Time Complexity: Parallel vs Sequential Execution")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_complexity2.png")
    plt.show()

def main(config):
    try:
        input_sizes, parallel_times, sequential_times = run_experiments(config)
        plot_time_complexity(input_sizes, parallel_times, sequential_times)
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    try:
        print_system_info()
        config = load_configuration("./config.json")
        main(config)
    except Exception as e:
        print(
            f"An error occurred while loading the configuration or running the main function: {e}"
        )
