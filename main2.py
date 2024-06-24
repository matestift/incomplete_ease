import utils
import numpy as np
import lmfit
from lmfit import fit_report
import time
import matplotlib.pyplot as plt
from ease_core import Ease
from layer import Layer, CompositeLayer, Void
from utils import read_ellipsometry, NCS_to_psi_delta

# Define constants for keys
METHOD = "method"
USE_PARALLEL = "use_parallel"
ELLIPSOMETRY_DATA = "ellipsometry_data"
DATA_FILE = "data_file"
MIN = "min"
MAX = "max"
VALUE = "value"
FIT = "fit"
THICKNESS = "thickness"
FRACTION = "fraction"
FIT_FRACTION = "fit_fraction"
FIT_THICKNESS = "fit_thickness"

# Configuration using variables for referencing
fit_settings = {
    METHOD: "differential_evolution",
    USE_PARALLEL: False
}

data_files = {
    ELLIPSOMETRY_DATA: "datfiles/SiO2_20nm1.dat",
    "SiO2_data": "matfiles/SiO2_JAW2.mat",
    "Si_data": "matfiles/Si_JAW2.mat",
}

wl_range = {
    MIN: 200,
    MAX: 800
}

angle = {
    VALUE: 70,
    FIT: {MIN: 60, MAX: 80}
}

# Define layers with direct references
incident_layer = {
    "name": "Incident Layer",
    "material": "void",
    "thickness": "inf"
}

si_o2_layer = {
    "name": "SiO2 Layer",
    "material": "SiO2_data",
    "thickness": 20,
    FIT_THICKNESS: {MIN: 10, MAX: 30}
}

si_composite_layers = [
    {"name": "Si Layer 1", "material": "Si_data", "fraction": 0.5, FIT_FRACTION: {MIN: 0.0, MAX: 1.0}},
    {"name": "Si Layer 2", "material": "Si_data", "fraction": 0.5, FIT_FRACTION: {MIN: 0.0, MAX: 1.0}}
]

composite_layer = {
    "name": "Si_Si_composite",
    "composite_layers": si_composite_layers
}

transmission_layer = {
    "name": "Transmission Layer",
    "material": "Si_Si_composite",
    "thickness": "inf"
}

layers = [incident_layer, si_o2_layer, transmission_layer]

export_options = {
    "export_fit_to_file": True,
    "default_export_filename": "fit_results.txt",
    "display_options": ["psi_delta", "ncs"]  # can be "psi_delta", "ncs", or both
}

config = {
    "fit_settings": fit_settings,
    "ellipsometry_method": "rcwa",
    "data_files": data_files,
    "wl_range": wl_range,
    "angle": angle,
    "layers": layers,
    "export_options": export_options
}

class SettingsError(Exception):
    pass

class Settings:
    def __init__(self, config):
        try:
            self.fit_settings = config["fit_settings"]
            self.ellipsometry_method = config["ellipsometry_method"]
            self.data_files = config["data_files"]
            self.wl_range = config["wl_range"]
            self.angle = config["angle"]
            self.layers = config["layers"]
            self.export_options = config["export_options"]
        except KeyError as e:
            raise SettingsError(f"Missing required configuration key: {e}")

    def get_wl_range(self):
        try:
            return [self.wl_range[MIN], self.wl_range[MAX]]
        except KeyError as e:
            raise SettingsError(f"Missing wavelength range configuration: {e}")

    def get_angle(self):
        try:
            return self.angle[VALUE]
        except KeyError as e:
            raise SettingsError(f"Missing angle configuration: {e}")

    def should_fit_angle(self):
        return FIT in self.angle

    def get_angle_bounds(self):
        if self.should_fit_angle():
            try:
                return self.angle[FIT][MIN], self.angle[FIT][MAX]
            except KeyError as e:
                raise SettingsError(f"Missing angle fit bounds configuration: {e}")
        return None

    def use_parallel(self):
        return self.fit_settings[USE_PARALLEL]

    def get_fit_method(self):
        return self.fit_settings[METHOD]

def create_layer(layer_info, data_files, wl, layer_map):
    try:
        # Check if the layer is a composite layer
        if "composite_layers" in layer_info:
            return create_composite_layer(layer_info, data_files, wl, layer_map)

        material_key = layer_info["material"]
        print(f"Creating layer with material: {material_key}")
        
        if material_key in layer_map:
            # Use existing layer from the map
            n_data = layer_map[material_key].refractive_index
            d = np.inf
        elif material_key == "void":
            n_data = np.column_stack((wl, np.ones_like(wl)))
            d = np.inf
        elif material_key in data_files:
            print(f"Reading refractive index for material: {material_key}")
            n_data = utils.read_refractive_index(data_files[material_key], wl)
            d = layer_info.get("thickness", np.inf)
        else:
            raise SettingsError(f"Missing data file for material: {material_key}")

        fit_thickness = "fit_thickness" in layer_info
        thickness_bounds = layer_info["fit_thickness"] if fit_thickness else None

        layer = Layer(n_data, d)
        layer.name = layer_info["name"]
        layer.fit_thickness = fit_thickness
        layer.thickness_bounds = thickness_bounds
        layer_map[layer_info["name"]] = layer
        print(f"Created layer: {layer.name}")

        return layer
    except KeyError as e:
        raise SettingsError(f"Missing required layer configuration key: {e}")
    except Exception as e:
        raise SettingsError(f"Unexpected error in create_layer: {e}")

def create_composite_layer(composite_info, data_files, wl, layer_map):
    try:
        print(f"Creating composite layer: {composite_info['name']}")
        
        composite_layer = CompositeLayer()
        sublayers = []
        
        for sublayer_info in composite_info["composite_layers"]:
            sublayer = create_layer(sublayer_info, data_files, wl, layer_map)
            sublayers.append(sublayer)
        
        composite_layer.add_layers(sublayers)
        composite_layer.name = composite_info["name"]
        layer_map[composite_info["name"]] = composite_layer
        print(f"Created composite layer: {composite_info['name']}")
        
        return composite_layer
    except Exception as e:
        raise SettingsError(f"Unexpected error in create_composite_layer: {e}")

def create_structure(settings, wl):
    print("Creating structure...")  # Debug print
    layer_map = {}
    structure = []

    # First pass: create and map all layers
    for layer_info in settings.layers:
        print(f"Processing layer: {layer_info['name']}")  # Debug print
        layer = create_layer(layer_info, settings.data_files, wl, layer_map)
        layer_map[layer_info["name"]] = layer

    print("Layer map keys:", layer_map.keys())  # Debug print

    # Second pass: construct the structure in the correct order
    for layer_info in settings.layers:
        layer_name = layer_info["name"]
        if layer_name in layer_map:
            structure.append(layer_map[layer_name])
        elif layer_info.get("material") == "void":
            structure.append(Void(wl))
        else:
            print(f"Current layer info: {layer_info}")  # Debug print
            print(f"Layer map: {layer_map}")  # Debug print layer map to inspect keys and values
            raise SettingsError(f"Missing layer configuration for material: {layer_info.get('material')}")

    return structure

def fit(params, wavelengths, angle, structure, data, use_parallel):
    try:
        ease = Ease(structure, params['angle'].value, wavelengths)
        N, C, S = ease.solve_ellipsometry(
            return_ncs=True, return_tuple=True, use_parallel=use_parallel
        )
        NCS = np.concatenate((N, C, S), axis=0)
        residual = NCS - data
        if np.any(np.isnan(residual)):
            print("NaN values found in residual")
        return residual
    except Exception as e:
        print(f"Error during fitting: {e}")
        return np.ones_like(data) * np.nan

def fit_and_plot(settings, fit, fit_method, wl, angle, structure, ncs, use_parallel=False):
    params = lmfit.Parameters()
    
    if settings.should_fit_angle():
        angle_bounds = settings.get_angle_bounds()
        params.add("angle", value=angle, min=angle_bounds[0], max=angle_bounds[1])
    else:
        params.add("angle", value=angle)

    for layer in structure:
        if isinstance(layer, Layer):
            if layer.fit_thickness:
                bounds = layer.thickness_bounds if layer.thickness_bounds else (0.0, 100.0)
                params.add(f"d_{layer.name}".replace(" ", "_"), value=layer.thickness, min=bounds[0], max=bounds[1])
        elif isinstance(layer, CompositeLayer):
            for sublayer in layer.layers:
                if sublayer.fit_thickness:
                    bounds = sublayer.thickness_bounds if sublayer.thickness_bounds else (0.0, 100.0)
                    params.add(f"d_{sublayer.name}".replace(" ", "_"), value=sublayer.thickness, min=bounds[0], max=bounds[1])
                if sublayer.fit_fraction:
                    bounds = sublayer.fraction_bounds if sublayer.fraction_bounds else (0.0, 1.0)
                    params.add(f"fraction_{sublayer.name}".replace(" ", "_"), value=sublayer.fraction, min=bounds[0], max=bounds[1])

    try:
        minner = lmfit.Minimizer(fit, params, fcn_args=(wl, angle, structure, ncs, use_parallel))
        result = minner.minimize(method=fit_method)
        NCS = ncs + result.residual
        NCS_reshaped = NCS.reshape(-1, len(wl))
        ncs_reshaped = ncs.reshape(-1, len(wl))
        N, C, S = NCS_reshaped[0], NCS_reshaped[1], NCS_reshaped[2]
        n, c, s = ncs_reshaped[0], ncs_reshaped[1], ncs_reshaped[2]

        print(fit_report(result))
        display_options = settings.export_options["display_options"]
        
        if "ncs" in display_options:
            plot_results(wl, n, N, "N")
            plot_results(wl, c, C, "C")
            plot_results(wl, s, S, "S")
        
        if "psi_delta" in display_options:
            psi_orig, delta_orig = NCS_to_psi_delta(n, c, s)
            psi_fit, delta_fit = NCS_to_psi_delta(N, C, S)
            plot_results(wl, psi_orig, psi_fit, "Psi")
            plot_results(wl, delta_orig, delta_fit, "Delta")

        plt.show()
    except Exception as e:
        print(f"Error during minimization: {e}")

def plot_results(wl, original_data, fitted_data, label):
    plt.figure()
    plt.plot(wl, original_data, "bo", label="Original")
    plt.plot(wl, fitted_data, "r-", label="Fitted")
    plt.xlabel('Wavelength')
    plt.ylabel(label)
    plt.legend()

def main():
    try:
        settings = Settings(config)
        print("Settings initialized successfully")

        wl_range = settings.get_wl_range()
        angle = settings.get_angle()
        fit_method = settings.get_fit_method()
        use_parallel = settings.use_parallel()

        wl, n, c, s = read_ellipsometry(settings.data_files[ELLIPSOMETRY_DATA], angle, wl_range[0], wl_range[1], return_ncs=True)
        ncs = np.concatenate((n, c, s), axis=0)
        print("Ellipsometry data read successfully")

        structure = create_structure(settings, wl)
        print("Structure created successfully")

        fit_and_plot(settings, fit, fit_method, wl, angle, structure, ncs, use_parallel)
        print("Fit and plot completed successfully")
    except SettingsError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
