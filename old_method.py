import numpy as np

pmax=2
no_of_sources=4
snr_db=2
threshold_prob=0.9
no_of_slots=2
channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )


received=np.array([])

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


source_main=source1(no_of_sources,15)
source=source_main.copy()


# source_main=np.array([i for i in range(1,no_of_sources+1)])
# source=source_main.copy()

channel_coeff=np.random.randn(len(source))
transmitted_1=source[channel_coeff>channel_threshold]
channel_gains=channel_coeff[channel_coeff>channel_threshold]
pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
received= ((transmitted_1 * channel_gains * pre_process) + np.random.randn(len(transmitted_1)))
recovered=demod(received,channel_gains,snr_db)
print("source:",source)
print("tr1",transmitted_1)
print("rc1",received)
print("rcvrd",recovered)
print("\n")
no_of_slots=0


while(len(source)>0):
    no_of_slots=no_of_slots+1
    source=source[channel_coeff<=channel_threshold]
    
    if len(source)==1:
        channel_gains=np.random.rayleigh(1,len(source))
        trasnmitted_1=source
        pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
        received=(trasnmitted_1 * channel_gains * pre_process) + np.random.normal(0,1,len(trasnmitted_1))
        recovered=demod(received,channel_gains,snr_db)
        
        print("source:",source)
        print(trasnmitted_1)
        print(received) 
        print("rcvrd",recovered)
        break 
     
    if len(source)==0:
        break
    
    channel_coeff=np.random.rayleigh(1,len(source))   
    trasnmitted_1=source[channel_coeff>channel_threshold]
    channel_gains=channel_coeff[channel_coeff>channel_threshold]
    pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
    received=(trasnmitted_1 * channel_gains * pre_process) + np.random.normal(0,1,len(trasnmitted_1))
    recovered=demod(received,channel_gains,snr_db)
    
    print("source:",source)
    print(trasnmitted_1)
    print(received)
    print("rcvrd",recovered)
    print("\n")
post_process=((np.sum(np.mean(channel_gains*pre_process)))  /  ( (np.sum(np.sum(channel_gains*pre_process)))**2 +no_of_slots*np.var(channel_coeff) ))
print("Tx",source_main,":",np.sum(source_main))
print("RX", received,np.sum(received))
