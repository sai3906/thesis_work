import os
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
import  matplotlib.pyplot as plt
import scipy.special as sp

import modulation as mod
import channel_1 as noise
import demodulation as demod






def calculate_ber(msg_bits, demodulated_bits):
    return np.mean(msg_bits != demodulated_bits)


def simulate_ber_vs_snr(snr_db_range, msg_bits):
    ber = []
    for snr_db in snr_db_range:
        # Modulate the bit sequence
        transmitted_signal =mod.qpsk(msg_bits)
        
        # in channel
        received_signal = noise.awgn(transmitted_signal,snr_db)
        
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
ber_qpsk_simulated= simulate_ber_vs_snr(snr_db_range, msg_bits)


#theoritical ber vs snr
snr=10**(snr_db_range / 10)
ber_qpsk_theory= 0.5*(sp.erfc(np.sqrt(0.5*snr)))

ber_bpsk_theory= 0.5*(sp.erfc(np.sqrt(2*snr)))

# Plot SNR vs BER
#plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_qpsk_simulated,  linestyle='-', color='b', label='simulated QPSK')
plt.semilogy(snr_db_range, ber_qpsk_theory,  linestyle='--', color='g', label='theoreitical QPSK')
plt.semilogy(snr_db_range, ber_bpsk_theory,  linestyle='--', color='r', label='theoreitical BPSK')
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER for QPSK Modulation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

