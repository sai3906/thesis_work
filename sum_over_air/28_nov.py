import numpy as np
import matplotlib.pyplot as plt

pmax=0.5

no_of_slots=40
slots_range=np.arange(2,no_of_slots+1)

snr=21
snr_range=np.arange(1,snr)

threshold_prob=0.999
threshold_prob_range= np.linspace(0, 1, 11)

no_of_sources=60
rnd_seed=21
iterations=1000



def source1(no_of_users)->np.ndarray:
    return np.random.uniform(-1, 1, (no_of_users))

def pre_process(msg,cha_coeff1,snr_db)->np.ndarray:
    snr_lin = 10**(snr_db / 10)
    cha_coeff=np.array(cha_coeff1)
    a=(np.sqrt(pmax)* np.sum(cha_coeff) )   /  (pmax* (np.sum(cha_coeff**2)) + (1/snr_lin) )
    b=1/(cha_coeff)
    b=np.minimum(1 / (cha_coeff),np.sqrt(pmax))
    print("b=",b)
    return msg*b

def awgn_mac_with_channel_coeff(symbols, snr_db,channel_coeff)->np.ndarray:
    # Sum the columns of the symbols array to combine signals from different users
    #cahnnel coefficienst

    # print("chann coeff:\n",channel_coeff)
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
    a_opt=( np.sum(b*channel_coeff) )   /  ( (np.sum((b*channel_coeff)**2)) + (no_of_slots) )
    return received_signal*a_opt

mse=[]
for threshold_prob  in threshold_prob_range:
    np.random.seed(rnd_seed)
    source_main=source1(no_of_sources)
    print("source:\n",source_main,"\n")
    channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
    print("channel threshold:: ",channel_threshold)
    print(f"snr={snr}")
    error=[]
    for j in range(iterations):
        print("***********************************************************") #-----------------                               
        print("iter",j+1,"\n")
        source=source_main.copy()
        recovered=np.array([])
        for i in slots_range:
            if i<len(slots_range):
                print(f"This is slot {i}")
                print("source:",source)#---------------------------
                # source_main=np.array([i for i in range(1,no_of_sources+1)])
                channel_coeff=np.random.randn(len(source))
                print("chan coeff:",channel_coeff)#----------------------
                transmitted=source[np.abs(channel_coeff)>channel_threshold]
                print(f"src {transmitted}" )#-----------------
                # print("elilble msgs:\n",transmitted)
                non_transmitted=source[np.abs(channel_coeff)<=channel_threshold]
                print("non transmitted:",non_transmitted)#---------------------
                channel_gains=channel_coeff[np.abs(channel_coeff)>channel_threshold]
                # print("elilble chann gains:\n",channel_gains)
                # pre_process(transmitted,channel_gains,snr_db)
                received= awgn_mac_with_channel_coeff(transmitted, snr,channel_gains)
                # print("rc1",received)#----------------------
                demod_signal=demod(received,channel_gains,snr)
                # print("rec",demod_signal)#---------------------
                recovered=np.append(recovered,demod_signal)
                print("---------------------------------------------")
                source=non_transmitted
                # print("untr",source)
            # Check if it's the last slot
            if i == len(slots_range):
                print(f"This is slot {i}")#----------------
                print("source:",non_transmitted)
                print(f"src {non_transmitted}" )#-------------------
                channel_gains=np.random.randn(len(non_transmitted))
                print("chan coeff:\n",channel_coeff)
                pre_process_signal=1
                # pre_process(transmitted,channel_gains,snr_db)
                received= awgn_mac_with_channel_coeff(non_transmitted, snr,channel_gains)
                # print("rc1",received)#---------------------------
                demod_signal=demod(received,channel_gains,snr)
                # print("rec",demod_signal)#---------------------
                recovered=np.append(recovered,demod_signal)          
        print("source_sum",sum(source_main))
        print("recovr_sum",sum(recovered))
        error.append((sum(source_main)-sum(recovered))**2)
    print("error:",error)
    print("==============================================================")
    mse.append(np.mean(error))
print("mse:",mse)
# x=np.arange(0,snr_db)
plt.plot(threshold_prob_range,mse)
plt.xlabel("Threshold Probability")
plt.ylabel("mse")
plt.show()

"""
For every snr it is averaging for given number of iterations
"""