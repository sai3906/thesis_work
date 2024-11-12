import numpy as np

pmax=2
no_of_sources=4
snr_db=2
threshold_prob=0.999
no_of_slots=4
channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
rnd_seed=14

def source1(no_of_users,rnd_seed):
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_values_per_node)
    # Each element is from  uniform distribution
    u=np.random.uniform(-1, 1, no_of_users)
    return u

def awgn_mac_with_channel_coeff(symbols, snr_db,channel_coeff,rnd_seed):
    print("symols:\n",symbols)
    #cahnnel coefficienst
    np.random.seed(rnd_seed)
    print("chann coeff:\n",channel_coeff)
    faded_symbols=symbols*channel_coeff# This corresponds to y = x_i * h_i
    print("faded symbols:\n",faded_symbols)
    # Print combined symbols for debugging (commented out)
    # combined_faded_symbols = np.sum(faded_symbols)# This corresponds to y = ∑x_i * h_i
    # print("combined_symbols:\n",combined_faded_symbols)   
    # Calculate the average power of the combined signal
    if faded_symbols.size > 0:
        signal_power = np.mean(np.abs(faded_symbols)**2)
    else:
        signal_power = 0
    #Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on the signal power and SNR
    noise_variance = signal_power / (2 * snr_linear)
    # Generate complex Gaussian noise with the calculated variance
    noise = np.sqrt(noise_variance) * np.random.randn(len(faded_symbols))  # Generating noise with accordance with signal power
    print("noise:\n",noise)
    return faded_symbols + noise


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
    # pre_process=np.minimum((np.sqrt(pmax)) ,  channel_threshold/channel_gains)
    received= awgn_mac_with_channel_coeff(transmitted_1, snr_db,channel_gains,rnd_seed)
    recovered=demod(received,channel_gains,snr_db)
    recovered1=np.append(recovered1,recovered)
    # print("source:",source)
    print("transmitted",transmitted_1)
    print("recieved",received)
    print("recovered",recovered)
    print("\n")
    source=non_transmitted
    
post_process=1
print("Tx",source_main,":",np.sum(source_main))
print("RX", recovered1,np.sum(recovered1))