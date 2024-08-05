import numpy as np
def awgn1(modulated_bits):
    #adding noise
    received_signal=np.array([])
    for i in range(0,len(modulated_bits)):
        a=modulated_bits[i].real + np.random.rayleigh(1,1)
        b=modulated_bits[i].imag + np.random.rayleigh(1,1)
        received_signal=np.append(received_signal, a + (1j * b))
    #print("channel op ",received_signal,"\n")
    return received_signal

def awgn(symbols, snr_db):
    """Add AWGN noise to the signal given an SNR in dB."""
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_variance) * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise