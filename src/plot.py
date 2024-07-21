import matplotlib.pyplot as plt
import os


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
    plt_file = os.path.join(os.getcwd(), "ellipsometry.png")
    plt.savefig(plt_file)


def plot_layer_structure(material_layers, params, substrate):
    fig, ax = plt.subplots(figsize=(7, 8))
    y_offset = 0
    layer_rectangles = []
    layer_labels = []
    layer_descriptions = []

    substrate_thickness = 50  # Arbitrary value for illustration
    substrate_label = f"Substrate ({substrate})"
    substrate_rect = plt.Rectangle(
        (0, y_offset), 1, substrate_thickness, edgecolor="black", facecolor="lightblue"
    )
    layer_rectangles.append(substrate_rect)
    layer_labels.append("Substrate")
    layer_descriptions.append(substrate_label)
    y_offset += substrate_thickness

    for layer in reversed(material_layers):
        if layer["type"] == "composite":
            material1_name = layer["material1"]
            material2_name = layer["material2"]
            composite_name = f"{material1_name}_{material2_name}_{layer['name']}"
            d_param_name = f"d_{composite_name}"
            fv_param_name = f"fv_{composite_name}"
            thickness = params[d_param_name].value
            fv = params[fv_param_name].value
            d_fit_status = "fit completed" if params[d_param_name].vary else "fixed"
            fv_fit_status = "fit completed" if params[fv_param_name].vary else "fixed"
            label = f"{layer['name']} (Composite)"
            description = (
                f"{material1_name}/{material2_name}\n"
                f"d: {thickness:.2f} nm ({d_fit_status})\n"
                f"fv: {fv:.2f} ({fv_fit_status})"
            )
        elif layer["type"] == "single":
            material_name = layer["material"]
            d_param_name = f"d_{material_name}_{layer['name']}"
            thickness = params[d_param_name].value
            d_fit_status = "fit completed" if params[d_param_name].vary else "fixed"
            label = f"{layer['name']} (Single)"
            description = f"{material_name}\n" f"d: {thickness:.2f} nm ({d_fit_status})"

        rect = plt.Rectangle(
            (0, y_offset), 1, thickness, edgecolor="black", facecolor="lightgray"
        )
        layer_rectangles.append(rect)
        layer_labels.append(label)
        layer_descriptions.append(description)
        y_offset += thickness

    total_height = y_offset

    for rect in layer_rectangles:
        ax.add_patch(rect)

    for rect, label in zip(layer_rectangles, layer_labels):
        ax.text(
            0.5,
            rect.get_y() + rect.get_height() / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
        )

    angle_value = params["angle"].value
    angle_status = "fit completed" if params["angle"].vary else "fixed"
    angle_description = f"Angle: {angle_value:.2f}° ({angle_status})"
    layer_labels.append("Angle")
    layer_descriptions.append(angle_description)

    plt.xlim(0, 1)
    plt.ylim(0, total_height + 10)
    plt.axis("off")
    plt.title("Layer Structure")

    legend_labels = [
        plt.Line2D([0], [0], color="black", lw=4, label=f"{label}\n{description}")
        for label, description in reversed(list(zip(layer_labels, layer_descriptions)))
    ]
    legend = ax.legend(
        handles=legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="small",
        frameon=False,
        labelspacing=1.2,
    )

    plt.tight_layout()
    plt.savefig("layer_structure_with_legend.png", bbox_inches="tight")
