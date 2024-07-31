import numpy as np
def awgn(modulated_bits):
    #adding noise
    received_signal=np.array([])
    for i in range(0,len(modulated_bits)):
        a=modulated_bits[i].real + np.random.normal(0,1,1)
        b=modulated_bits[i].imag + np.random.normal(0,1,1)
        received_signal=np.append(received_signal, a+(1j * b))
    #print("channel op ",received_signal,"\n")
    return received_signal
