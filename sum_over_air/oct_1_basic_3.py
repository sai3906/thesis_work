import numpy as np
from matplotlib import pyplot as plt

no_of_sources=8
pmax=2
gain_threshold=0.1
snr_db=[1]

def source(no_of_sources):
     return np.array([i for i in range(1,no_of_sources+1)])



def multislot(msg):
    np.random.seed(23)
    return np.random.randn(len(msg))




"""
index=channel_coeff>gain_threshold
print("index:",index)

transmitted_1=msg[index]
print("transmitted1-->",transmitted_1)

channel_coeff1=channel_coeff[index]
print("channe_coeff-->",channel_coeff1)
"""
# pre_process=1
# np.minimum((np.sqrt(pmax)) ,  gain_threshold/channel_coeff1)
#print("pre_process",pre_process)

# post_process=1
# (np.sum(np.mean(pre_process*channel_coeff1))) / ( np.sum(np.var(pre_process*channel_coeff1)) + 2*(np.var(channel_coeff1)) )
#print("post process",post_process)

def channel(transmitted,channel_coeff1,snr_db):
    pre_process=1
    # Calculate the average power of the combined signal
    signal_power = np.mean(np.abs(transmitted)**2)  # Signal power calculation
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on the signal power and SNR
    noise_variance = signal_power / (2 * snr_linear)
    # Generate complex Gaussian noise with the calculated variance
    noise = np.sqrt(noise_variance) * channel_coeff1  # Generating noise with accordance with signal power

    # Add the noise to the combined symbols and return the result
    return transmitted + noise

# received=channel(msg,channel_coeff,snr_db)




def demod(transmitted,received_signal):
    possible_symbols=transmitted
    # print("\n\n\nmsg:",possible_symbols,"\nmsg_sum:",possible_symbols.sum(),"\n")
    # print("channel_coeff\n",channel_coeff)
    # print("received:\n",received_signal,"\n\n")
    recovered_sum=[]
    
    for r in received_signal:
        min_distance = float('inf')
        best_symbol = None  # To keep track of the symbol with the minimum distance
        for x in possible_symbols:
            # print("x=",x,"r=",r)
            distance = np.abs(r - x)
            # print("\t\t\t  d=",distance)
            if distance < min_distance:
               min_distance = distance
               best_symbol = x
        # print("min_distance:",min_distance,"best_symb:",best_symbol) 
        # print("---------------------------------------------------------------------")
        recovered_sum.append(best_symbol)       
    return np.array(recovered_sum)



def calculate_error(msg,recoverd):
    print("msg}",msg,msg.sum())
    print("rvd}",recoverd,recoverd.sum())
    return  np.abs(msg.sum()-recoverd.sum())

def simulate(snr_db,no_of_sources):
    error=[]
    for snr in snr_db:
        transmitted=source(no_of_sources)
        # print("transmitted:",transmitted)
        channel_coeff=multislot(transmitted)
        # print("channel coeff:",channel_coeff)
        received=channel(transmitted,channel_coeff,snr)
        recoverd=demod(transmitted,received)
        error.append(calculate_error(transmitted,recoverd))
    return error

result=simulate(snr_db,no_of_sources)
print("error:",result)








































# while source.size>1:
#     source=source[channel_coeff<=gain_threshold]
#     print("source:",source)

#     channel_coeff=np.random.rayleigh(1,len(source))
#     #print("channel_coeff:",channel_coeff)

#     index=channel_coeff>gain_threshold
#     print("index:",index)
    
    
#     trasnmitted_2=source[index]
#     print("transmitted2-->",trasnmitted_2)

#     channel_coeff1=channel_coeff[index]
#     print("channel_coeff2-->",channel_coeff1)

#     pre_process=np.minimum((np.sqrt(pmax)) ,  gain_threshold/channel_coeff1)
#     #print("pre_process",pre_process)

#     post_process=(np.sum(np.mean(pre_process*channel_coeff1))) / ( np.sum(np.var(pre_process*channel_coeff1)) + 2*(np.var(channel_coeff1)) )
#     #print("post process",post_process)
        

        
#     receieved =   post_process * ( (trasnmitted_2 * channel_coeff1 * pre_process) + np.random.normal(0,1,len(trasnmitted_2)))
#     print("recieved-->",receieved)

#     print("\n")