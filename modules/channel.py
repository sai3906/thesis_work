import numpy as np

def awgn(bits, snr_db):
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_variance) * (np.random.randn(len(bits)))
    return bits + noise
