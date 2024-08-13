
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
import  matplotlib.pyplot as plt
import scipy.special as sp
import snrvsber as simulate


no_of_bits=35000
msg_bits1=np.random.randint(0, 2, no_of_bits)

def generate_2_bit_number():
    # Generate a random integer between 0 and 15 (inclusive) for true 4-bit numbers
    number = np.random.randint(0, 3)
    
    # Convert the number to its 4-bit binary representation
    binary_representation = format(number, '02b')
    
    return binary_representation
msg_bits_bpsk = []
for i in range(no_of_bits):
    a = generate_2_bit_number()
    msg_bits_bpsk.append(a)
# Convert msg_bits to a 2D numpy array, then flatten to 1D
msg_bits_array1 = np.array([list(bits) for bits in msg_bits_bpsk]).flatten()
# Convert the 1D array of binary bits from strings to integers
msg_bits_bpsk_sum = np.array([int(bit) for bit in msg_bits_array1])

def generate_2_bit_number():
    # Generate a random integer between 0 and 15 (inclusive) for true 4-bit numbers
    number = np.random.randint(0, 9)
    
    # Convert the number to its 4-bit binary representation
    binary_representation = format(number, '04b')
    
    return binary_representation
msg_bits2 = []
for i in range(no_of_bits):
    a = generate_2_bit_number()
    msg_bits2.append(a)
# Convert msg_bits to a 2D numpy array, then flatten to 1D
msg_bits_array2 = np.array([list(bits) for bits in msg_bits2]).flatten()
# Convert the 1D array of binary bits from strings to integers
msg_bits_qpsk_sum = np.array([int(bit) for bit in msg_bits_array2])

#------------------------------------------------------------------------------------------------------------------------------------#
snr_db_range = np.arange(0,9, 1)  # SNR range from 0 dB to 12 dB
#theoritical ber vs snr
snr=10**(snr_db_range / 10)

bpsk_theory=0.5*(sp.erfc(np.sqrt(snr)))
bpsk_sum_thoery=0.5*(sp.erfc(np.sqrt(snr)))

qpsk_theory= 0.25*(sp.erfc(np.sqrt(0.5*snr)))
qpsk_sum_theory= 0.25*(sp.erfc(np.sqrt(0.5*snr)))

#for bpsk,qpsk give input msg_bits1
#for bpsk_sum input is msg_bits_bpsk_sum
#for bpsk_sum input is msg_bits_qpsk_sum


bpsk_simulated= simulate.bpsk(snr_db_range, msg_bits1)
bpsk_sum_simulated= simulate.bpsk(snr_db_range, msg_bits_bpsk_sum)

qpsk_simulated=simulate.qpsk(snr_db_range,msg_bits1)
qpsk_sum_simulated=simulate.qpsk_sum(snr_db_range,msg_bits_qpsk_sum)


# Plot SNR vs BER

plt.semilogy(snr_db_range, bpsk_theory,  linestyle=':', color='g', label='bpsk theoreitical')
plt.semilogy(snr_db_range, bpsk_simulated,  linestyle='-', color='g', label='bpsk simulated')

plt.semilogy(snr_db_range, bpsk_sum_thoery,  linestyle='--', color='r', label='bpsk sum theory')
plt.semilogy(snr_db_range, bpsk_sum_simulated,  linestyle='-', color='r', label='bpsk sum simulated')

plt.semilogy(snr_db_range, qpsk_theory,  linestyle=':', color='m', label='qpsk_theoritical')
plt.semilogy(snr_db_range, qpsk_simulated,  linestyle='-', color='m ', label='qpsk simulated')

plt.semilogy(snr_db_range, qpsk_sum_theory,  linestyle='--', color="c", label='qpsk sum thoey')
plt.semilogy(snr_db_range, qpsk_sum_simulated,  linestyle='--', color='c', label='qpsk sum simulated')


plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER for QPSK Modulation')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

