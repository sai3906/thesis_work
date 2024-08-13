import numpy as np

def bpsk(bits):
    #mapping
    mapping = {
        0: 1,
        1:-1
    }

    #modulation
    modulated_bits=np.array([])
    for i in range(0,len(bits)):
        bit_pair=(bits[i])
        modulated_bits=np.append(modulated_bits,mapping[bit_pair])
    #print("mod",modulated_bits)
    return modulated_bits

def qpsk(bits):
    #mapping
    mapping = {
        (0,0): 1+1j,
        (0,1):-1+1j,
        (1,0):1-1j,
        (1,1): -1-1j
    }

    #modulation
    modulated_bits=np.array([])
    for i in range(0,len(bits),2):
        bit_pair=(bits[i],bits[i+1])
        modulated_bits=np.append(modulated_bits,mapping[bit_pair])
    #print("mod",modulated_bits)
    return modulated_bits/np.sqrt(2)

def qpsk_sum(bits):
    #mapping
    mapping = {
        (0,0,0,0): 2+2j,
        (0,0,0,1):2j,
        (0,0,1,0):0,
        (0,0,1,1):2,
        (0,1,0,0):-2+2j,
        (0,1,0,1):-2,
        (0,1,1,0):-2-2j,
        (0,1,1,1):-2j,
        (1,0,0,0):2-2j
    }

    #modulation
    symbols=np.array([])
    for i in range(0,len(bits),4):
        bit_pair=(bits[i],bits[i+1],bits[i+2],bits[i+3])
        symbols=np.append(symbols,mapping[bit_pair])
    #print("mod",modulated_bits)
    return symbols
