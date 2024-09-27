import numpy as np
import source as src
import modulation as md
def awgn_mac_with_channel_coeff(symbols, snr_db,no_of_users,no_of_bits,rnd_seed):
    # Sum the columns of the symbols array to combine signals from different users
    #cahnnel coefficienst
    np.random.seed(rnd_seed)
    channel_coeff=np.random.randn(no_of_users,no_of_bits)
    print("chann coeff:\n",channel_coeff)
    faded_symbols=symbols*channel_coeff# This corresponds to y = x_i * h_i
    # Print combined symbols for debugging (commented out)
    combined_faded_symbols = np.sum(faded_symbols, axis=0)# This corresponds to y = ∑x_i * h_i
    print("combined_symbols:\n",combined_faded_symbols)   
    # Calculate the average power of the combined signal
    signal_power = np.mean(np.abs(combined_faded_symbols)**2)  # Signal power calculation
    #Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on the signal power and SNR
    noise_variance = signal_power / (2 * snr_linear)
    # Generate complex Gaussian noise with the calculated variance
    noise = np.sqrt(noise_variance) * np.random.randn(len(combined_faded_symbols))  # Generating noise with accordance with signal power
    print("noise:\n",noise)
    #multiplying with channel coeff
    # channel_coeff=np.random.randn(len(combined_symbols))
    # Add the noise to the combined symbols and return the result
    return combined_faded_symbols + noise,channel_coeff

# ch_op=awgn_mac_with_channel_coeff([[-1],[1]],1,2,4)
# print(ch_op)