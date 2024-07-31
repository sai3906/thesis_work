import numpy as np
def qpsk_modulation(bits):
    #mapping
    mapping = {
        (0,0): 1+1j,
        (0,1):-1+1j,
        (1,0):-1-1j,
        (1,1): 1-1j
    }

    #modulation
    modulated_bits=np.array([])
    for i in range(0,len(bits),2):
        bit_pair=(bits[i],bits[i+1])
        modulated_bits=np.append(modulated_bits,mapping[bit_pair])
    #print("mod",modulated_bits)
    return modulated_bits