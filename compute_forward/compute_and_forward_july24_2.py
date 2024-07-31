import os
os.system('cls' if os.name == 'nt' else 'clear')
#the above line is to clear the terminal for every run


import numpy as np
from matplotlib import pyplot as plt

no_of_transmitters_receivers=5

no_of_simulations=3

for i in range(no_of_simulations):
    transmitter=np.random.rand(no_of_transmitters_receivers)
    encoder=np.array([int(value) for value in (transmitter>0.5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value?
    #how to do lattice encoding ?

    channel_gains=np.random.rayleigh(1,no_of_transmitters_receivers**2).reshape(no_of_transmitters_receivers,no_of_transmitters_receivers)

    noise=np.random.normal(0,1)
    #reception
    decoder_input=channel_gains@transmitter +noise
    decoder_output=np.array([int(value) for value in (decoder_input>5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value




    print(f"{i+1}")
    print("\nTranmission")
    print(f"message {transmitter}")
    print(f"encded_msg {encoder}")
    print("\nReception")
    print(f"decoder_i/p {decoder_input}")
    print(f"decoded_msg {decoder_output}\n")

    print("________________________________________________________________________________________________________________")
