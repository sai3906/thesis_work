# %% [markdown]
# Its working fine for snr vs mse

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
no_of_sources=5
threshold_prob=0.9999
threshold_prob_range= np.linspace(0.1, 0.99,100)

# %% [markdown]
# changed msg,channel coeff,channel threshold,noise
# 

# %% [markdown]
# change random seed below

# %%
def source1(no_of_users)->np.ndarray:
    rnd_seed=19
    # np.random.seed(rnd_seed)
    return np.random.uniform(0, 0.5, (no_of_users))
    # return np.array([0.1, 0.1, 0.1, 0.1, 0.1])

# %%
def pre_process(msg,cha_coeff1,power:float)->np.ndarray:
    print("\t\tx=",msg)
    print("\t\th=",cha_coeff1)
    cha_coeff=np.array(cha_coeff1)
    print("\t\t1/h=",1/cha_coeff,"\t\tsqrt(p)=",np.sqrt(power))
    b=np.minimum(1 / (cha_coeff),np.sqrt(power))
    print("\t\tb=",b)
    return msg*b


# %% [markdown]
# changed noise variance
# 

# %%
def awgn_mac_with_channel_coeff(symbols, snr_db,channel_coeff)->np.ndarray:
    print("\t\txb=",symbols)
    faded_symbols=symbols*channel_coeff# This corresponds to y = x_i * h_i
    print("\t\tcha=",channel_coeff)
    print("\t\txbh:",faded_symbols)
    combined_faded_symbols = np.sum(faded_symbols)# This corresponds to y = ∑x_i * h_i
    print("\t\t∑xbh=",combined_faded_symbols)
    signal_power = np.abs(combined_faded_symbols)**2  # Signal power calculation
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = signal_power / (snr_linear)
    noise =np.sqrt(noise_variance) * np.random.randn()# Noise generation
    print("\t\tn=",noise)
    return combined_faded_symbols + noise

# %%
def demod(received_signal,cha_coeff,power:float)->np.ndarray:
    print("\t\tr=",received_signal)
    b=np.minimum(1 / (cha_coeff),np.sqrt(power))
    a_opt=( np.sum(b*cha_coeff) )   /  ( (np.sum((b*cha_coeff)**2)) + (1) )
    return received_signal*a_opt
    # return 1

# %% [markdown]
# input parmaeters

# %%
snr=40
snr_range=np.arange(0,snr+1)

available_power=4

no_of_slots=2
slots_range=np.arange(1,no_of_slots+1)
iterations=400

# %%
mse=[]
for snr in snr_range: 
    channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
    # channel_threshold=0.9
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("parameter::",snr,"|","chan threshold",channel_threshold,"|","power",available_power)
    error=[]
    for j in range(iterations):
        print(" iter ",j+1,100*"*")
        source_main=source1(no_of_sources)
        print("main source:",source_main,source_main.sum())
        source=source_main.copy()
        
        recovered=np.array([])
        # np.random.seed(j+snr)
        
        for i in slots_range:
            if i<len(slots_range):
                print(f"\tslot {i}")
                channel_coeff=np.random.randn(len(source))
                # channel_coeff = np.array([0.2, 0.2, 1, 1, 1])[:len(source)]
                print("\tcha coeff=",channel_coeff,"\n")
                
                eligible_msgs=source[np.abs(channel_coeff)>channel_threshold]
                
                non_transmitted=source[np.abs(channel_coeff)<=channel_threshold]
                
                channel_gains=channel_coeff[np.abs(channel_coeff)>channel_threshold]
                
                transmitted=pre_process(eligible_msgs,channel_gains,available_power)
                received= awgn_mac_with_channel_coeff(transmitted, snr,channel_gains)
                
                demod_signal=demod(received,channel_gains,available_power)
                recovered=np.append(recovered,demod_signal)
                print("\t\tr`=",demod_signal)
                source=non_transmitted
                print("\t\t---------------------------------------------")           
            
            # Check if it's the last slot
            if i == len(slots_range):
                print(f"\tslot {i}")
                channel_gains=np.random.randn(len(non_transmitted))
                # channel_gains = np.array([0.4, 0.4,1])[:len(non_transmitted)]
                transmitted=pre_process(non_transmitted,channel_gains,available_power)
                received= awgn_mac_with_channel_coeff(transmitted, snr,channel_gains)
                demod_signal=demod(received,channel_gains,available_power)
                recovered=np.append(recovered,demod_signal)
                print("\t\tr`=",demod_signal)
                print("\t\t---------------------------------------------")
        print("______________________________________________________________________")
        print("source sum",source_main.sum())
        print("recovered",recovered.sum())
        error.append(np.mean((source_main.sum() - recovered.sum()) ** 2))
        print("error",error)
    mse.append(np.mean(error))
    print("mse",mse)
    
print("\nSNR:",snr_range)
print("MSE:",mse)
plt.plot(snr_range,mse)
plt.show()

# %% [markdown]
# this is averaging over msg,channel coefficients and noise since random seed is not being used anywhere
#these good results are due to available power is too high hence b=1/h is always less than sqrt(p) hence b is always 1/h   

# %%
