import modulation as mod
import numpy as np
def ber_bpsk_sum_with_fading(msg, recovered,rnd_seed):
    np.random.seed(rnd_seed)
    # Modulate the original message bits to get the transmitted symbols
    sym1 =mod.mod(msg)
    # Sum the transmitted symbols across all users
    combined_symbols = np.sum(sym1, axis=0)  # This is used to compare with the received symbols
    print("msg:",combined_symbols)
    print("rec:",np.array(recovered))
    # Calculate the Bit Error Rate (BER) by comparing the combined transmitted symbols with the received symbols
    ber = np.mean(combined_symbols != recovered)  # BER is calculated by averaging the number of symbol errors
    return ber
