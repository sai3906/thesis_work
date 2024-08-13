import numpy as np
import modulation as mod
import channel as noise
import demodulation as demod

def calculate_ber(msg, rcovered):
    return np.mean(msg != rcovered)

def qpsk(snr_db_range, msg_bits):
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
def qpsk_sum(snr_db_range, msg_bits):
    ber = []
    #ser:=symbol error rate

    for snr_db in snr_db_range:
        # Modulation        symbols-->bitsy])
        transmitted_symbols =mod.qpsk_sum(msg_bits)
        
        #channel output = bits+noie
        received_symbols = noise.awgn(transmitted_symbols,snr_db)
        
        # Demodulate the received signal        noisy_bits-->symbols
        demodulated_symbols =demod.qpsk_sum(received_symbols)
        
        # Calculate BER
        ber.append(calculate_ber(msg_bits, demodulated_symbols))
        
    return ber

def bpsk(snr_db_range, msg_bits):
    ber = []
    #ser:=symbol error rate

    for snr_db in snr_db_range:
        # Modulation        symbols-->bitsy])
        transmitted_symbols =mod.bpsk(msg_bits)
        
        #channel output = bits+noie
        received_symbols = noise.awgn(transmitted_symbols,snr_db)
        
        # Demodulate the received signal        noisy_bits-->symbols
        demodulated_symbols =demod.bpsk(received_symbols)
        
        # Calculate BER
        ber.append(calculate_ber(msg_bits, demodulated_symbols))
        
    return ber