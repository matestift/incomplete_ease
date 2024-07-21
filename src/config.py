import json
import os

def load_configuration(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def validate_config(config):
    required_keys = [
        "wl_range",
        "data_file",
        "material_layers",
        "materials",
        "substrate",
        "angle",
        "fit_method",
        "use_parallel",
        "fit_parameters",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    if not isinstance(config["wl_range"], list) or len(config["wl_range"]) != 2:
        raise ValueError("wl_range must be a list of two values [min, max]")

    if not os.path.isfile(config["data_file"]):
        raise FileNotFoundError(f"Data file not found: {config['data_file']}")

    for material_name, material_config in config["materials"].items():
        if material_name != "void":
            if not os.path.isfile(material_config["file"]):
                raise FileNotFoundError(
                    f"Material file not found: {material_config['file']}"
                )

    for layer in config["material_layers"]:
        if layer["type"] == "composite":
            if (
                "material1" not in layer
                or "material2" not in layer
                or "d" not in layer
                or "fv" not in layer
            ):
                raise ValueError(
                    "Composite layers must have 'material1', 'material2', 'd', and 'fv' fields"
                )
        elif layer["type"] == "single":
            if "material" not in layer or "d" not in layer:
                raise ValueError("Single layers must have 'material' and 'd' fields")

    if not isinstance(config["fit_parameters"], list):
        raise ValueError("fit_parameters must be a list of parameter names")

    valid_params = set()
    for layer in config["material_layers"]:
        if layer["type"] == "composite":
            composite_name = (
                f"{layer['material1']}_{layer['material2']}_{layer['name']}"
            )
            valid_params.add(f"fv_{composite_name}")
            valid_params.add(f"d_{composite_name}")
        elif layer["type"] == "single":
            valid_params.add(f"d_{layer['material']}_{layer['name']}")
    valid_params.add("angle")

    for param in config["fit_parameters"]:
        if param not in valid_params:
            raise ValueError(f"Invalid fit parameter: {param}")
