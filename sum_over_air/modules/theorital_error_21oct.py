import channel as ch
import numpy as np
def theoritical_error_calc(snr_db,msg,cha_coeff,received):
    snr_lin = 10**(snr_db / 10)
    a_opt=( np.sum(cha_coeff) )   /  ( np.sum(cha_coeff**2) + (1/snr_lin) )
    msg_sum=np.sum(msg,axis=0)
    error_theory=np.mean(  (( a_opt*received) - msg_sum)**2   )   
    return error_theory