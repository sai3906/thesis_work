import numpy as np
def demod(received_signal,channel_coeff,snr_db):
    snr_lin = 10**(snr_db / 10)
    # print("received:\n",received_signal,"\n")
    a_opt=( np.sum(channel_coeff) )   /  ( (np.sum(channel_coeff**2)) + (1/snr_lin) )
    return received_signal*a_opt
