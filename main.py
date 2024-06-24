import numpy as np
import utils
from fitter import fit_and_plot, fit

# config.yml
"""
fit_settings:
  method: differential_evolution
  use_parallel: false

ellipsometry_method: rcwa  # Choose between rcwa or tmm

data_files:
  ellipsometry_data: datfiles/SiO2_20nm1.dat
  SiO2_data: matfiles/SiO2_JAW2.mat
  Si_data: matfiles/Si_JAW2.mat

wl_range:
  min: 200
  max: 800

angle:
  value: 70 # Initial guess for angle
  fit_angle: false # Set to true if angle should be fitted

layers:
  - Void
  - Si:
      n: Si_data
      d: inf
  - SiO2:
      n: SiO2_data
      d: 20
      fit_d: true # Indicate that thickness d should be fitted

composite_layer:
  - SiO2_Si_composite:
      layers:
        - Si:
            n: Si_data
            d: 10
            fit_d: true # Indicate that thickness d should be fitted
            fraction: 0.3 # Initial guess for fraction
            fit_fraction: true # Indicate that fraction should be fitted
        - SiO2:
            n: SiO2_data
            d: 10
            fit_d: true # Indicate that thickness d should be fitted
            fraction: 0.7 # Initial guess for fraction
            fit_fraction: true # Indicate that fraction should be fitted

export_options:
  - export_fit_to_file: true # Set to true to export fit results to a file
  - default_export_filename: "fit_results.txt" # Default filename if not specified
  - display_options: 
      - psi_delta # Display fit as Psi and Delta
      - ncs # Display fit as N, C, S
"""


#define new helper classes that contain extra flags that contains information about whether a parameter or class should be found by fitting or the values is to be supplied



def main():
    wl_range = [200, 800]
    angle = 70
    wl, n, c, s = utils.read_ellipsometry(
        "datfiles/SiO2_20nm1.dat", angle, wl_range[0], wl_range[1], return_ncs=True
    )
    ncs = np.concatenate((n, c, s), axis=0)

    n_Si = utils.read_refractive_index("matfiles/Si_JAW2.mat", wl)
    n_SiO2 = utils.read_refractive_index("matfiles/SiO2_JAW2.mat", wl)
    fit_method = "differential_evolution"
    fit_and_plot(fit, fit_method, wl, angle, n_Si, n_SiO2, ncs, use_parallel=False)


if __name__ == "__main__":
    main()
