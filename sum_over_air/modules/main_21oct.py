import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
import  matplotlib.pyplot as plt
import source as  src
import channel as ch
import demodulation_21oct as dmd
import error_21oct as er
import theorital_error_21oct as th_err


rnd_seed=12
print("each row represent a user\neach column represents a time slot\n")
#Function to simulate the system for different SNR values
def simulate_mac_with_fading(snr_db_range,no_of_users,nvps,rnd_seed):
    
    mse_simulated = []  # Initialize an empty list to store Symbol Error Rate (SER) for each SNR value
    mse_theory=[]
    # Iterate over the range of SNR values
    for snr_db in snr_db_range:
        # print("snr=", snr_db)
         
        msg= src.source(no_of_users, nvps,rnd_seed) # Generating bit stream of size (no_of_users, no_of_bits)     
        
        # print("msg:\n",msg)         
        
        # Pass the transmitted symbols through the AWGN channel to get received symbols
        received_signal,channel_coeff = ch.awgn_mac_with_channel_coeff(msg, snr_db,no_of_users,nvps,rnd_seed)  # Received symbols: channel output = symbols + noise
        
        # print("received:\n",received_signal)       
        
        # Demodulate the received symbols to recover the transmitted symbols
        recovered_signal = dmd.demod(received_signal,channel_coeff,snr_db)  # Demodulation: noisy_symbols --> decoded symbols
        
        # print("demodulated:\n",recovered_signal)    
        
        theory_err=th_err.theoritical_error_calc(snr_db,msg,channel_coeff,received_signal)
        mse_theory.append(np.mean(theory_err))
        
        # linear_estimator_error.append(er.error_calculation(msg, recovered_signal,rnd_seed)) 
        estimator_error=er.error_calculation(msg, recovered_signal,rnd_seed)
        mse_simulated.append(np.mean(estimator_error))
        
        # print("error:",estimator_error)
        
        # Demodulate the received symbols to recover the transmitted symbol
        # print("===================================================================")
    return mse_simulated,mse_theory   # Return the list of SER values for the given range of SNR values


# Define the range of SNR values (in dB) for the simulation
snr_db_range =np.arange(0, 20, 1)  # Input for SNR range to plot
# Run the simulation to obtain SER for each SNR value

no_of_users = 2  # Define the number of users
no_of_values_per_source =40000  # Define the number of bits per user
# Generate the message bits as a binary matrix

mse_simulated,mse_theoritical=simulate_mac_with_fading(snr_db_range,no_of_users,no_of_values_per_source,rnd_seed)  
print("mse_simul:",mse_simulated)
print("\nmse_theor:",mse_theoritical)

# Convert SNR values from dB to linear scale
snr_lin = 10**(snr_db_range / 10)  # SNR in linear scale

# Plot the simulated BER values
plt.semilogy(snr_db_range, mse_simulated, linestyle='-', color="g", label='simulated')
plt.semilogy(snr_db_range, mse_theoritical, linestyle='-', color="r", label='theoritical')     
plt.xlabel("SNR(dB)")
plt.ylabel("MSE")
plt.legend()         
plt.title("MSE of linear estimator") 
plt.show()
