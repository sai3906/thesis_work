import numpy as np
import  matplotlib.pyplot as plt
from scipy import special as sp

no_of_bits=10000
no_of_users=1
msg_bits=np.random.randint(0,2,(no_of_users,no_of_bits))


symbols=np.where(msg_bits==0,-1,1)


signal_sum=np.sum(symbols,axis=0)


snr_db=np.arange(0,12,1)
snr_lin=10**(snr_db/10)
noise_power=1/snr_lin  


ber=[]
for noispower in noise_power:
    noise=np.sqrt(noispower)*np.random.randn(len(signal_sum))
    received=signal_sum+noise
    
    if no_of_users == 2:
        recover_sym = np.where(received < -1, -2, np.where(received > 1, 2, 0))
    else:
        recover_sym = np.where(received < 0, -1, 1)
        recover_bits=np.where(recover_sym<0,0,1)
    
    ser=np.mean(msg_bits!=recover_bits)
    ber.append(ser if no_of_users==1 else ser/2)

plt.semilogy(snr_db,ber)
plt.grid(True, which='both')
plt.show()