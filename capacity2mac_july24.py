import numpy as np
from matplotlib import pyplot as plt

length=3

transmitter1=np.random.rand(length)
#encoder1=np.array([int(value) for value in (transmitter1>0)])

transmitter2=np.random.rand(length)
#encoder2=np.array([int(value) for value in (transmitter2>0)])

channel_coeff=np.random.rayleigh(1,length**2).reshape(length,length)

noise1=np.random.normal(0,1)
noise2=np.random.normal(0,1)

receiver1=channel_coeff@transmitter1 +noise1
receiver2=channel_coeff@transmitter2 +noise2


print("tx1",transmitter1)
print("rx1",receiver1,"\n")
print("tx2",transmitter2)
print("rx2",receiver2)
