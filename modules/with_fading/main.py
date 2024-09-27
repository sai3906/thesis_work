import numpy as np
import  matplotlib.pyplot as plt
import source as  src
import modulation as  md
import channel as ch
import demodulation as dmd
import ber as ber
rnd_seed=None

#Function to simulate the system for different SNR values
def simulate_mac_with_fading(snr_db_range,no_of_users,no_of_bits,rnd_seed):
    
    ber_with_channel_coeff = []  # Initialize an empty list to store Symbol Error Rate (SER) for each SNR value
    # Iterate over the range of SNR values
    for snr_db in snr_db_range:
        # print("snr=", snr_db)
         
        msg_bits = src.source(no_of_users, no_of_bits,rnd_seed) # Generating bit stream of size (no_of_users, no_of_bits)     
        # print("msg_bits:\n",msg_bits)
        
        # Modulate the message bits to obtain transmitted symbols
        transmitted_symbols = md.mod(msg_bits)  # Modulation: bit stream --> symbols
        print("symbols:\n",transmitted_symbols)         
        
        # Pass the transmitted symbols through the AWGN channel to get received symbols
        received_symbols,faded_symbols = ch.awgn_mac_with_channel_coeff(transmitted_symbols, snr_db,no_of_users,no_of_bits,rnd_seed)  # Received symbols: channel output = symbols + noise
        # print("received\n",received_symbols)       
        
        # Demodulate the received symbols to recover the transmitted symbols
        demodulated_symbols = dmd.demod(received_symbols,faded_symbols)  # Demodulation: noisy_symbols --> decoded symbols
        # print("demodulated\n",demodulated_symbols)    
        
        #compare the demodulated symbols with  the original transmitted symbols to get the number of errors
        ber_with_channel_coeff.append(ber.ber_bpsk_sum_with_fading(msg_bits, demodulated_symbols,rnd_seed)) 
        # Demodulate the received symbols to recover the transmitted symbol
        
        return ber_with_channel_coeff   # Return the list of SER values for the given range of SNR values


# Define the range of SNR values (in dB) for the simulation
snr_db_range =np.arange(1, 12, 2)  # Input for SNR range to plot
# Run the simulation to obtain SER for each SNR value

no_of_users = 2  # Define the number of users
no_of_bits =4  # Define the number of bits per user
# Generate the message bits as a binary matrix

ber_bpsk_sum_with_fading=simulate_mac_with_fading(snr_db_range,no_of_users,no_of_bits,rnd_seed)
# Print the Bit Error Rate (BER) for each SNR value
# print(ber_bpsk_sum_with_channel_coeff) 
print("BER",ber_bpsk_sum_with_fading)

# Convert SNR values from dB to linear scale
snr_lin = 10**(snr_db_range / 10)  # SNR in linear scale


# Plot the simulated BER values
plt.semilogy(snr_db_range, ber_bpsk_sum_with_fading, linestyle='-', color="g", label='simulated')
#theoritical values
# plt.semilogy(snr_db_range,1/((1*snr_db_range)), linestyle='--', color="r", label='theoritical')
plt.xlabel("SNR linear")  
plt.ylabel("log(BER)")       
plt.legend()         
plt.title("BPSK Two Users") 
plt.grid(True)
plt.show()
