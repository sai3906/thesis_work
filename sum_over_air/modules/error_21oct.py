import numpy as np
def error_calculation(msg, recovered,rnd_seed):
    np.random.seed(rnd_seed)
    # Sum the transmitted symbols across all users
    signal_sum = np.sum(msg, axis=0)  # This is used to compare with the received symbols
    # print("msg:",signal_sum)
    # print("rec:",recovered)
    # Calculate the Bit Error Rate (BER) by comparing the combined transmitted symbols with the received symbols
    error = signal_sum - recovered  # BER is calculated by averaging the number of symbol errors
    return error**2
