import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

def qpsk_mod(bits):
    mapping = {
        (0,0): 1 + 1j,
        (0,1): -1 + 1j,
        (1,0): 1 - 1j,
        (1,1): -1 - 1j
    }
    symbols = np.array([], dtype=complex)
    for i in range(len(bits)):
        for j in range(0, len(bits[i]), 2):
            bit_pair = (bits[i][j], bits[i][j+1])
            symbols = np.append(symbols, mapping[bit_pair])
    return symbols

def qpsk_demod(received_signal):
    symbols = {
        1 + 1j: (0, 0),
        -1 + 1j: (0, 1),
        1 - 1j: (1, 0),
        -1 - 1j: (1, 1)
    }
    demodulated_bits = []
    for i in received_signal:
        closest_symbol = min(symbols.keys(), key=lambda s: np.abs(i - s))
        demodulated_bits.extend(symbols[closest_symbol])
    return np.array(demodulated_bits)

def simulate(no_of_users, snr_db):
    no_of_bits = 10000
    msg_bits = np.random.randint(0, 2, (no_of_users, no_of_bits))
    
    symbols = qpsk_mod(msg_bits)
    
    symbols = symbols.reshape(no_of_users, int(no_of_bits / 2))
    symbols_sum = np.sum(symbols, axis=0)
    
    snr_lin = 10**(snr_db / 10)
    noise_power = 1 / snr_lin
    
    ber = []
    for noispower in noise_power:
        noise = np.sqrt(noispower / 2) * (np.random.randn(int(no_of_bits / 2)) + 1j * np.random.randn(int(no_of_bits / 2)))
        received_symbols = symbols_sum + noise
        
        recovered_bits = qpsk_demod(received_symbols).reshape(no_of_users, no_of_bits)
        
        ser = np.mean(msg_bits != recovered_bits)
        ber.append(ser / 2 if no_of_users == 1 else ser / 4)
    
    return ber

snr_db = np.arange(0, 12, 1)
no_of_users = 1
qpsk_ber = simulate(no_of_users, snr_db)

plt.semilogy(snr_db, qpsk_ber, label='QPSK BER')

ber_theory = 2 * sp.erfc(np.sqrt(10**(snr_db / 10)) / np.sqrt(2))
plt.semilogy(snr_db, ber_theory, label='Theoretical BER')

plt.grid(True, which='both')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.show()
