import numpy as np
def demod(received_signal,channel_coeff):
    print("received:\n",received_signal,"\n")
    possible_symbols=[(-1,-1),(-1,1),(1,-1),(1,1)]
    # print("channel_coeff\n",channel_coeff,"\n\n")
    recovered_sum=[]
    best_symbol = None  # To keep track of the symbol with the minimum distance
    for i in range(0,len(channel_coeff[0])):
        min_distance = float('inf')  # Initialize min_distance to a large value
        for j in range(0,len(possible_symbols)):
            # print(f"ch_coef[{i}]=",channel_coeff[:,i],"\tsymbol:",np.array(possible_symbols[j]))
            hi_xi=channel_coeff[:,i]*np.array(possible_symbols[j])
            sum_hi_xi=np.sum(hi_xi)
            # print("∑x_i*h_i=",sum_hi_xi,"\t","y=",received_signal[i])
            distance=np.abs(received_signal[i]-sum_hi_xi)**2
            # print("distance=",distance,"\n\n")
            # Update the minimum distance and corresponding symbol
            if distance < min_distance:
                min_distance = distance
                best_symbol =np.sum(possible_symbols[j])
        # print("min_distance:",min_distance,"best_symb:",best_symbol)    
        # print("---------------------------------------------------------------------")
        recovered_sum.append(np.sum(best_symbol))
    return recovered_sum

# ch=np.array([[1,-1],[0,1]])
# dmd_syb=demod([-2,2],ch)
# print("demodulated symbols",dmd_syb)