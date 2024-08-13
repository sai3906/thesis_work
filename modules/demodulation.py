import numpy as np

def bpsk(received_signal):
    symbols = {
        1: 0,
        -1:1
    }

    #demodulation
    demodulated_bits=np.array([])
    for i in received_signal:
        closest_symbol = min(symbols.keys(), key=lambda s: np.abs(i - s))
        demodulated_bits=np.append(demodulated_bits,(symbols[closest_symbol]))
    #    print(np.array(demodulated_bits))

    received_signal1=np.array([int(i)  for i in demodulated_bits])
    #converting into int just for appearance
    return received_signal1


def qpsk(received_signal):
    symbols = {
        1 + 1j: (0, 0),
        -1 + 1j: (0, 1),
        1 - 1j: (1, 0),
        -1 - 1j: (1, 1)
    }

    #demodulation
    demodulated_bits=np.array([])
    for i in received_signal:
        closest_symbol = min(symbols.keys(), key=lambda s: np.abs(i - s))
        demodulated_bits=np.append(demodulated_bits,(symbols[closest_symbol]))
    #    print(np.array(demodulated_bits))

    received_signal1=np.array([int(i)  for i in demodulated_bits])
    #converting into int just for appearance
    return received_signal1

def qpsk_sum(received_signal):
    symbols = {
        2 + 2j: (0, 0,0,0),
        2j: (0,0,0, 1),
        0: (0,0,1, 0),
        2: (0,0,1, 1),
        -2+2j:(0,1,0,0),
        -2:(0,1,0,1),
        -2-2j:(0,1,1,0),
        -2j:(0,1,1,1),
        2-2j:(1,0,0,0)
    }

    #demodulation
    demodulated_bits=np.array([])
    for i in received_signal:
        closest_symbol = min(symbols.keys(), key=lambda s: np.abs(i - s))
        demodulated_bits=np.append(demodulated_bits,(symbols[closest_symbol]))
    #    print(np.array(demodulated_bits))

    received_signal1=np.array([int(i)  for i in demodulated_bits])
    #converting into int just for appearance
    return received_signal1