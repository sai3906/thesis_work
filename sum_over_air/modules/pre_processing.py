import numpy as np
def pre_process(msg,cha_coeff1,snr_db):
    snr_lin = 10**(snr_db / 10)
    cha_coeff=np.array(cha_coeff1)
    a=( np.sum(cha_coeff) )   /  ( (np.sum(cha_coeff**2)) + (1/snr_lin) )
    b=1/(cha_coeff*a)
    # print("msg\n",msg)
    # print("cha coeff\n",cha_coeff)
    # print("a=\n",a)
    # print("b=\n",b)
    return msg*b

# res=pre_process([[1,2],[3,4]],[[0.1,0.2],[0.3,0.4]],1)
# print("res:",res)