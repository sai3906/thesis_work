import os
os.system('cls' if os.name == 'nt' else 'clear')
#the above line is to clear the terminal for every run


import numpy as np
from matplotlib import pyplot as plt

code_length=5

no_of_transmitters_recievers=3

for i in range(no_of_transmitters_recievers):
    transmitter=np.random.rand(code_length)
    encoder=np.array([int(value) for value in (transmitter>0.5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value?
    #how to do lattice encoding ?

    channel_coeff=np.random.rayleigh(1,code_length**2).reshape(code_length,code_length)

    noise=np.random.normal(0,1)
    #reception
    decoder=channel_coeff@transmitter +noise
    receiver=np.array([int(value) for value in (decoder>5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value




    print(f"{i+1}")
    print("\nTranmission")
    print(f"tx{i+1} {transmitter}")
    print(f"enc{i+1} {encoder}")
    print("\nReception")
    print(f"dec{i+1} {decoder}")
    print(f"rx{i+1} {receiver}\n")

    print("________________________________________________________________________________________________________________")
