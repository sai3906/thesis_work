import numpy as np
def demod(received_signal,channel_coeff):
    # print("received:\n",received_signal,"\n")
    a_opt=(np.sum(channel_coeff))/(np.sum(channel_coeff**2)+1)
    return received_signal*a_opt

# ch=np.array([[0.2,-0.3],[0.1,0.5]])
# dmd_syb=demod([-2,2],ch)
# print("demodulated symbols",dmd_syb)