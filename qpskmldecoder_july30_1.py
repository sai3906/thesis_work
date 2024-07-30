import os
os.system('cls' if os.name == 'nt' else 'clear')

import numpy as np

bits=np.array([0,0,0,1,1,0])
print("msg  bits",bits)


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


#addin noise
received_signal=np.array([])
for i in range(0,len(modulated_bits)):
    a=modulated_bits[i].real + np.random.normal(0,1,1)
    b=modulated_bits[i].imag + np.random.normal(0,1,1)
    received_signal=np.append(received_signal,a+(1j * b))
#print("channel op ",received_signal,"\n")

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
print("rcvd bits",received_signal1)
