import numpy as np
import matplotlib.pyplot as plt
print("\n")

no_of_sources=5

no_of_slots=3
slots_range=np.arange(1,no_of_slots+1)

snr=200
snr_range=np.arange(1,snr+1)

threshold_prob=0.9999
threshold_prob_range= np.linspace(0.1, 0.99,100)

rnd_seed=21
iterations=1



def source1(no_of_users,rnd_seed)->np.ndarray:
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_values_per_node)
    # Each element is from  uniform distribution
    return np.random.uniform(-1, 1, (no_of_users))

def pre_process(msg,cha_coeff1,chan_thres,snr_db)->np.ndarray:
    p=2
    snr_lin = 10**(snr_db / 10)
    cha_coeff=np.array(cha_coeff1)
    a=(np.sqrt(p)* np.sum(cha_coeff) )   /  (p* (np.sum(cha_coeff**2)) + (1/snr_lin) )
    b=1/(cha_coeff)
    b=np.minimum(1 / (cha_coeff),np.sqrt(p))
    return msg*b

def awgn_mac_with_channel_coeff(symbols, snr_db,channel_coeff,rnd_seed)->np.ndarray:
    # Sum the columns of the symbols array to combine signals from different users
    
    faded_symbols=symbols*channel_coeff# This corresponds to y = x_i * h_i
    # print("faded symbols:\n",faded_symbols)
    # Print combined symbols for debugging (commented out)
    combined_faded_symbols = np.sum(faded_symbols)# This corresponds to y = ∑x_i * h_i
    # print(f"Type of combined_faded_symbols: {type(combined_faded_symbols)}, Shape: {np.shape(combined_faded_symbols)}")

    # print("combined_symbols:\n",combined_faded_symbols)   
    # Calculate the average power of the combined signal
    signal_power = np.mean(np.abs(combined_faded_symbols)**2)  # Signal power calculation
    #Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on the signal power and SNR
    noise_variance = signal_power / (2 * snr_linear)
    # Generate complex Gaussian noise with the calculated variance
    noise = np.sqrt(noise_variance) * np.random.randn()  # Generating noise with accordance with signal power
    # print("noise:\n",noise)
    #multiplying with channel coeff
    # channel_coeff=np.random.randn(len(combined_symbols))
    # Add the noise to the combined symbols and return the result
    return combined_faded_symbols + noise

def demod(received_signal,channel_coeff,snr_db)->np.ndarray:
    # print("recieved",received)
    # print("recovered",recovered)
    # print("\n")r_db):
    snr_lin = 10**(snr_db / 10)
    # print("received:\n",received_signal,"\n")
    b=1/(channel_coeff)
    a_opt=( np.sum(b*channel_coeff) )   /  ( (np.sum((b*channel_coeff)**2)) + (1) )
    return received_signal*a_opt

mse=[]
for no_of_slots in slots_range:
    channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
    print("parameter::",no_of_slots,"|","chan threshold",channel_threshold)
    print("\n")
    error=[]
    for j in range(iterations):
        print("iter",j+1)
        source_main=source1(no_of_sources,rnd_seed)
        print("source:\n",source_main,"\n")
        source=source_main.copy()
        recovered=np.array([])
        for i in slots_range:
            if i<len(slots_range):
                print(f"slot {i}")
                print(f"source:{i}\n",source)
                channel_coeff=np.random.randn(len(source))
                print("chan coeff:\n",channel_coeff)
                transmitted=source[np.abs(channel_coeff)>channel_threshold]
                print(f"src \n{transmitted}" )
                # print("elilble msgs:\n",transmitted)
                non_transmitted=source[np.abs(channel_coeff)<=channel_threshold]
                print("non transmitted:\n",non_transmitted)
                channel_gains=channel_coeff[np.abs(channel_coeff)>channel_threshold]
                print("elilble chann gains:\n",channel_gains)
                # pre_process(transmitted,channel_gains,snr_db)
                received= awgn_mac_with_channel_coeff(transmitted, snr,channel_gains,rnd_seed)
                print("rc1",received)
                demod_signal=demod(received,channel_gains,snr)
                print("rec",demod_signal)
                recovered=np.append(recovered,demod_signal)
                print("---------------------------------------------")
                source=non_transmitted

            
            # Check if it's the last slot
            if i == len(slots_range):
                print(f"This is slot {i}")
                print("source:",non_transmitted)
                print(f"src {non_transmitted}" )
                channel_gains=np.random.randn(len(non_transmitted))
                print("chan coeff:\n",channel_gains)
                pre_process_signal=1
                # pre_process(transmitted,channel_gains,snr_db)
                received= awgn_mac_with_channel_coeff(non_transmitted, snr,channel_gains,rnd_seed)
                print("rc1",received)
                demod_signal=demod(received,channel_gains,snr)
                print("rec",demod_signal)
                recovered=np.append(recovered,demod_signal)
                print("---------------------------------------------")           
        print("***********************************************************")                                
        print("source",np.sum(source_main))
        print("recovr",sum(recovered))
        error.append((sum(source_main)-sum(recovered))**2)
    print("==============================================================")
    # print("error:",error)
    mse.append(np.mean(error))

print("mse:",mse)
# x=np.arange(0,snr_db)
plt.plot(slots_range,mse)
# plt.xlabel("snr_db")
plt.ylabel("mse")
plt.show()
