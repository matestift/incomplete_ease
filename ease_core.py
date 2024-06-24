from tmm import ellips
from joblib import Parallel, delayed
import numpy as np
import psutil
import utils


_cached_num_free_cpus = None

def _get_free_cpus(threshold=50):
    global _cached_num_free_cpus
    if _cached_num_free_cpus is None:
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        free_cpus = [i for i, usage in enumerate(cpu_usage) if usage < threshold]  #<50% usage is free
        _cached_num_free_cpus = len(free_cpus)
    return _cached_num_free_cpus

class Ease:
    def __init__(self, layers, angle, wavelengths):
        self.layers = layers
        self.angle = angle
        self.wavelengths = wavelengths

    def _compute_ellipsometry(self, wavelength):
        n_layers = [layer.get_refractive_index(wavelength) for layer in self.layers]
        d_layers = [layer.thickness for layer in self.layers]
        result = ellips(n_layers, d_layers, np.deg2rad(self.angle), wavelength)
        psi = result['psi']
        delta = np.pi - result['Delta']
        return {
            'angle': self.angle,
            'wavelength': wavelength,
            'psi': psi,
            'delta': delta
        }

    def solve_ellipsometry(self, return_ncs=False, return_tuple=False, use_parallel=False):
        results = []
        psi_list = []
        delta_list = []

        # cache the free cpu result
        if use_parallel:
            free_cpus = _get_free_cpus()
            if free_cpus == 0:
                use_parallel = False

        if use_parallel:
            num_workers = free_cpus
            results = Parallel(n_jobs=num_workers)(delayed(self._compute_ellipsometry)(wl) for wl in self.wavelengths)
            psi_list = [result['psi'] for result in results]
            delta_list = [result['delta'] for result in results]
        else:
            for wavelength in self.wavelengths:
                result = self._compute_ellipsometry(wavelength)
                results.append(result)
                psi_list.append(result['psi'])
                delta_list.append(result['delta'])

        if return_ncs:
            N = []
            C = []
            S = []
            for result in results:
                n, c, s = utils.psi_delta_to_NCS(result['psi'], result['delta'])
                N.append(n)
                C.append(c)
                S.append(s)

            if return_tuple:
                return np.array(N), np.array(C), np.array(S)
            else:
                ncs_results = []
                for n, c, s, result in zip(N, C, S, results):
                    ncs_results.append({
                        'angle': self.angle,
                        'wavelength': result['wavelength'],
                        'N': n,
                        'C': c,
                        'S': s
                    })
                return ncs_results
        
        if return_tuple:
            return np.array(psi_list), np.array(delta_list)
        else:
            psi_delta_results = []
            for psi, delta, result in zip(psi_list, delta_list, results):
                psi_delta_results.append({
                    'angle': self.angle,
                    'wavelength': result['wavelength'],
                    'psi': psi,
                    'delta': delta
                })
            return psi_delta_results
