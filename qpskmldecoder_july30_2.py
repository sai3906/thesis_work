import os
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
from matplotlib import pyplot as plt

import modulation as mod
import channel as awgn
import demodulation as demod





def calculate_ber(msg_bits, demodulated_bits):
    errors = np.sum(msg_bits != demodulated_bits)
    total_bits = len(msg_bits)
    return errors / total_bits




def simulate_ber_vs_snr(snr_db_range, msg_bits):
    ber = []
    for snr_db in snr_db_range:
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        
        # Modulate the bit sequence
        transmitted_signal = mod.qpsk_modulation(msg_bits)
        
        # Generate noise
        noise_power = 1 / (2 * snr_linear)
        #noise = (np.random.randn(len(transmitted_signal)) + 1j * np.random.randn(len(transmitted_signal))) * np.sqrt(noise_power)
        noise=awgn.awgn(transmitted_signal)* noise_power

        # Add noise to the signal
        received_signal = transmitted_signal + noise
        
        # Demodulate the received signal
        demodulated_bits = demod.qpsk_demodulation(received_signal)
        
        # Calculate BER
        ber.append(calculate_ber(msg_bits, demodulated_bits))
        
    return ber

# Parameters
no_of_bits=500000
msg_bits=np.random.choice([0,1],no_of_bits)
#print("source  ",msg_bits)

snr_db_range = np.arange(0, 21, 2)  # SNR range from 0 dB to 20 dB

# Simulate and get BER for different SNR values
ber = simulate_ber_vs_snr(snr_db_range, msg_bits)

# Plot SNR vs BER
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber,  linestyle='-', color='b')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs Bit Error Rate for QPSK Modulation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()