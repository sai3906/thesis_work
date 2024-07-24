import numpy as np
from matplotlib import pyplot as plt

length=5
#transmission 

no_of_transmitters=4
for i in range(no_of_transmitters):
    transmitter=np.random.choice([0,1],length)
    encoder=np.array([int(value) for value in (transmitter>0.5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value?
    #how to do lattice encoding ?

    channel_coeff=np.random.rayleigh(1,length**2).reshape(length,length)

    noise1=np.random.normal(0,1)
    #reception
    decoder=channel_coeff@transmitter +noise1
    receiver=np.array([int(value) for value in (decoder>5)])
    #here encoder is deciding on arbitrary value 0.5
    #how can i decide this value

    print("\nTranmission")
    print(f"tx{i+1} {transmitter}")
    print(f"enc{i+1} {encoder}")
    print("\nReception")
    print(f"dec{i+1} {decoder}")
    print(f"rx{i+1} {receiver}\n")

    print("________________________________________________________________________________________________________________")




#transmitter2=np.random.choice([0,1],length)
#encoder2=np.array([int(value) for value in (transmitter2>0.5)])
#noise
#noise2=np.random.normal(0,1)
#decoder2=channel_coeff@transmitter2 +noise2
#receiver2=np.array([int(value) for value in (decoder2>5)])

#print("\nTranmission")
#print("tx2",transmitter2)
#print("enc2",encoder2)
#print("\nReception")
#print("dec2",decoder2)
#print("rx2",receiver2,"\n")
