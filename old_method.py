import numpy as np

pmax=2
no_of_sources=4
snr_db=2
threshold_prob=1
no_of_slots=10
channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
rnd_seed=None

def source1(no_of_users,rnd_seed):
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_values_per_node)
    # Each element is from  uniform distribution
    u=np.random.uniform(-1, 1, no_of_users)
    return u

def demod(received_signal,channel_coeff,snr_db):
    snr_lin = 10**(snr_db / 10)
    # print("received:\n",received_signal,"\n")
    a_opt=( np.sum(channel_coeff) )   /  ( (np.sum(channel_coeff**2)) + (1/snr_lin) )
    return received_signal*a_opt

source_main=source1(no_of_sources,rnd_seed)
source=source_main.copy()

recovered1=np.array([])
for i in range(no_of_slots):
    print(f"This is slot {i + 1}")

    # source_main=np.array([i for i in range(1,no_of_sources+1)])

    channel_coeff=np.random.randn(len(source))
    transmitted_1=source[channel_coeff>channel_threshold]
    non_transmitted=source[channel_coeff<=channel_threshold]
    channel_gains=channel_coeff[channel_coeff>channel_threshold]
    pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
    received= ((transmitted_1 * channel_gains * pre_process) + np.random.randn(len(transmitted_1)))
    recovered=demod(received,channel_gains,snr_db)
    recovered1=np.append(recovered1,recovered)
    # print("source:",source)
    print("tr1",transmitted_1)
    print("rc1",recovered)
    print("\n")
    source=non_transmitted
    # no_of_slots=0


    
    # no_of_slots=no_of_slots+1
    # source=source[channel_coeff<=channel_threshold]
    

    # channel_gains=np.random.rayleigh(1,len(source))
    # trasnmitted_1=source
    # pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
    # received=(trasnmitted_1 * channel_gains * pre_process) + np.random.normal(0,1,len(trasnmitted_1))
    # recovered=demod(received,channel_gains,snr_db)
    # recovered1=np.append(recovered1,recovered)
    
    # print("source:",source)
    # print(trasnmitted_1)
    # print(recovered)
    

    
    # channel_coeff=np.random.rayleigh(1,len(source))   
    # trasnmitted_1=source[channel_coeff>channel_threshold]
    # channel_gains=channel_coeff[channel_coeff>channel_threshold]
    # pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
    # received=(trasnmitted_1 * channel_gains * pre_process) + np.random.normal(0,1,len(trasnmitted_1))
    # recovered=demod(received,channel_gains,snr_db)
    # recovered1=np.append(recovered1,recovered)
    
post_process=1
print("Tx",source_main,":",np.sum(source_main))
print("RX", recovered1,np.sum(recovered1))