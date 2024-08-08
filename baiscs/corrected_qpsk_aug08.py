import os
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np
import  matplotlib.pyplot as plt
import scipy.special as sp

def qpskmod(symbols):
    modulated_bits = []
    for symbol in symbols:
        if symbol.real > 0 and symbol.imag > 0:
            modulated_bits.extend([0, 0])
        elif symbol.real < 0 and symbol.imag > 0:
            modulated_bits.extend([0, 1])
        elif symbol.real < 0 and symbol.imag < 0:
            modulated_bits.extend([1, 0])
        elif symbol.real > 0 and symbol.imag < 0:
            modulated_bits.extend([1, 1])
    return np.array(modulated_bits)


def awgn(bits, snr_db):
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_variance) * (np.random.randn(len(bits)))
    return bits + noise


def qpskdemod(bitstream):
    msg=[]
    for i in range(0,int(len(bitstream)),2):
        if(bitstream[i]>0 and bitstream[i+1]>0 ):
            msg.extend([1+1j])
        elif(bitstream[i]<0 and bitstream[i+1]>0 ):
            msg.extend([-1+1j])
        elif(bitstream[i]<0 and bitstream[i+1]<0 ):
            msg.extend([-1-1j])
        else:
            msg.extend([1-1j])
    return np.array(msg)


def calculate_ber(msg, recovered):
    # Ensure msg and recovered have the same length
    print(msg)
    if len(msg) != len(recovered):
        raise ValueError("msg and recovered must have the same length.")
    
    # Calculate bit errors
    bit_errors = np.sum(msg != recovered)
    print(bit_errors)
    # Calculate Bit Error Rate (BER)
    ber = bit_errors/len(msg)
    return ber

def simulate_ber_vs_snr(snr_db_range, msg_symbols):
    ber = []
    for snr_db in snr_db_range:
        # Modulate the bit sequence
        transmitted_bits =qpskmod(msg_symbols)
        
        # in channel
        received_bits = awgn(transmitted_bits,snr_db)
        
        # Demodulate the received signal
        demodulated_symbols =qpskdemod(received_bits)
        
        # Calculate BER
        ber.append(calculate_ber(msg_symbols, demodulated_symbols))
        
    return ber




# Parameters
no_of_symbols=5
symbols=np.array([1+1j,-1+1j,-1-1j,1-1j])
msg_symbols=np.random.choice(symbols,no_of_symbols)

snr_db_range = np.arange(0,2, 1)  # SNR range from 0 dB to 20 dB

# Simulate and get BER for different SNR values
ber_qpsk_simulated= simulate_ber_vs_snr(snr_db_range, msg_symbols)


#theoritical ber vs snr
snr=10**(snr_db_range / 10)
ber_qpsk_theory= 0.5*(sp.erfc(np.sqrt(0.5*snr)))


# Plot SNR vs BER
#plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_qpsk_simulated,  linestyle='-', color='b', label='simulated')
#plt.semilogy(snr_db_range, ber_qpsk_theory,  linestyle='--', color='g', label='theoreitical')
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER for QPSK Modulation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()