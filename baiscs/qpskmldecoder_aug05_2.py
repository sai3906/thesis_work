import os
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sp

import modulation as mod
import channel as awgn
import demodulation as demod






def calculate_ber(msg_bits, demodulated_bits):
    return np.mean(msg_bits != demodulated_bits)


def simulate_ber_vs_snr(snr_db_range, msg_bits):
    ber = []
    for snr_db in snr_db_range:
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        # Modulate the bit sequence
        transmitted_signal =mod.qpsk(msg_bits)
        
        # in channel
        received_signal = awgn.awgn(transmitted_signal,snr_db)
        
        # Demodulate the received signal
        demodulated_bits =demod.qpsk(received_signal)
        
        # Calculate BER
        ber.append(calculate_ber(msg_bits, demodulated_bits))
        
    return ber

# Parameters
no_of_bits=50000
msg_bits=np.random.randint(0, 2, no_of_bits)

snr_db_range = np.arange(0,12, 1)  # SNR range from 0 dB to 20 dB

# Simulate and get BER for different SNR values
ber1= simulate_ber_vs_snr(snr_db_range, msg_bits)


#theoritical ber vs snr
snr=10**(snr_db_range / 10)
ber= sp.erfc(np.sqrt(snr))

# Plot SNR vs BER
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber1,  linestyle='-', color='b', label='simulated')
plt.semilogy(snr_db_range, ber,  linestyle='--', color='g', label='theoritical')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs Bit Error Rate for QPSK Modulation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

