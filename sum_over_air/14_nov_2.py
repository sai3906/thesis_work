import numpy as np
import matplotlib.pyplot as plt

# This is for no of slots vs mse
pmax=2
slots=np.arange(1,20)
snr=np.arange(4,5)
rnd_seed=None
no_of_slots=40
no_of_sources=60
iterations=50

threshold_prob=0.999

def source1(no_of_users,rnd_seed)->np.ndarray:
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_values_per_node)
    # Each element is from  uniform distribution
    return np.random.uniform(-1, 1, (no_of_users))

def pre_process(msg,cha_coeff1,snr_db)->np.ndarray:
    snr_lin = 10**(snr_db / 10)
    cha_coeff=np.array(cha_coeff1)
    a=( np.sum(cha_coeff) )   /  ( (np.sum(cha_coeff**2)) + (1/snr_lin) )
    b=1/(cha_coeff*a)
    return msg*b



def awgn_mac_with_channel_coeff(symbols, snr_db,channel_coeff,rnd_seed)->np.ndarray:
    # Sum the columns of the symbols array to combine signals from different users
    #cahnnel coefficienst
    np.random.seed(rnd_seed)
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
    a_opt=( np.sum(channel_coeff) )   /  ( (np.sum(channel_coeff**2)) + (1/snr_lin) )
    return received_signal*a_opt

mse=[]
for no_of_slots in slots:
    channel_threshold=np.sqrt( 2*np.log(  (1/ (1-(1-threshold_prob )**(1/no_of_slots))  )  ) )
    print("snr..",snr+1)
    error=[]
    for j in range(iterations):
        source_main=source1(no_of_sources,rnd_seed)
        print("source:\n",source_main)
        print("\n")
        source=source_main.copy()
        recovered=np.array([])
        for i in range(no_of_slots):
            # print(f"This is slot {i + 1}")
            print("source:",source)
            # source_main=np.array([i for i in range(1,no_of_sources+1)])
            channel_coeff=np.random.randn(len(source))
            # print("chan coeff:\n",channel_coeff)
            transmitted=source[channel_coeff>channel_threshold]
            print(f"src {transmitted}" )
            # print("elilble msgs:\n",transmitted)
            non_transmitted=source[channel_coeff<=channel_threshold]
            print("non transmitted:\n",non_transmitted)
            channel_gains=channel_coeff[channel_coeff>channel_threshold]
            # print("elilble chann gains:\n",channel_gains)
            pre_process_signal=1
            # pre_process(transmitted,channel_gains,snr_db)
            received= awgn_mac_with_channel_coeff(transmitted, snr,channel_gains,rnd_seed)
            print("rc1",received)
            demod_signal=demod(received,channel_gains,snr)
            print("rec",demod_signal)
            recovered=np.append(recovered,demod_signal)
            print("---------------------------------------------")
            source=non_transmitted
        print("main source",sum(source_main))
        print("recoverd",sum(recovered))
        error.append((sum(source_main)-sum(recovered))**2)
        print("error::",error)
        print("==============================================================")
    mse.append(np.mean(error))

print("mse=",mse)
# x=np.arange(0,snr_db)
plt.plot(slots,mse)
plt.show()