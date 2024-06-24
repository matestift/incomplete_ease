import numpy as np
from utils import ema
from scipy import interpolate

class Layer:
    def __init__(self, refractive_index, thickness, fraction=1):
        self.refractive_index = refractive_index
        self.thickness = thickness
        self.fraction = fraction
    
    def get_refractive_index(self, wavelength):
        idx = (np.abs(self.refractive_index[:, 0] - wavelength)).argmin()
        return self.refractive_index[idx, 1]

class CompositeLayer(Layer):
    def __init__(self):
        super().__init__(0, 0)
        self.layers = []
        self.fractions = []
        self.thickness = 0

    def add_layers(self, layers):
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError("All elements in layers must be instances of the Layer class.")
        
        self.layers = layers
        self.fractions = [layer.fraction for layer in layers]

        self.calculate_properties()

    def calculate_properties(self):
        if not self.layers or len(self.layers) != len(self.fractions):
            raise ValueError("Mismatch in the number of layers and fractions or empty layers.")
        
        dielectric_functions = [layer.refractive_index[:, 1]**2 for layer in self.layers]
        
        effective_dielectric_function = ema(np.array(dielectric_functions), np.array(self.fractions))
        
        self.refractive_index = np.column_stack((self.layers[0].refractive_index[:, 0], np.sqrt(effective_dielectric_function)))
        
        self.thickness = sum(layer.thickness * layer.fraction for layer in self.layers)
    

    def effective_refractive_index(self, wavelength):
        return self.get_refractive_index(wavelength)

    def effective_thickness(self):
        return self.thickness

def Void(wavelengths):
    return Layer(np.column_stack((wavelengths, np.ones_like(wavelengths))), np.inf)
