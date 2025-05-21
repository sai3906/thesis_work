import numpy as np
import matplotlib.pyplot as plt

# Source generation function
def source1(no_of_users) -> np.ndarray:
    return np.random.uniform(-1, 1, (no_of_users))

# Preprocessing methods
def pre_process1(msg, cha_coeff, alpha, power):
    b = alpha / cha_coeff
    return msg * b

def pre_process2(msg, cha_coeff, alpha, power):
    b = alpha * (cha_coeff / np.abs(cha_coeff)) * np.minimum(1 / np.abs(cha_coeff), np.sqrt(power) / alpha)
    return msg * b

# Channel simulation with AWGN
def awgn_mac_with_channel_coeff(symbols, snr_db, channel_coeff):
    faded_symbols = symbols * channel_coeff
    signal_power = np.mean(np.abs(symbols) ** 2)
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    noise = np.sqrt(noise_variance) * np.random.randn(*faded_symbols.shape)
    combined_faded_symbols = np.sum(faded_symbols) + np.sum(noise)
    return combined_faded_symbols, noise_variance

# Demodulation methods
def demod1(received_signal, cha_coeff, power, alpha, noise_variance):
    b = alpha / cha_coeff
    a_opt = np.sum(b * cha_coeff) / (np.sum((b * cha_coeff) ** 2) + noise_variance)
    return received_signal * a_opt

def demod2(received_signal, cha_coeff, power, alpha, noise_variance):
    b = alpha * (cha_coeff / np.abs(cha_coeff)) * np.minimum(1 / np.abs(cha_coeff), np.sqrt(power) / alpha)
    a_opt = np.sum(b * cha_coeff) / (np.sum((b * cha_coeff) ** 2) + noise_variance)
    return received_signal * a_opt

# Main simulation parameters
plotting = 0
comments = 0

no_of_sources = 5
no_of_slots = 4
threshold_prob = 0.9999
snr_range = np.arange(0, 51)
available_power = 2.5
iterations = 1000

mse = []

# Run the slot-based simulation
for snr in snr_range:
    rnd_seed = 3
    np.random.seed(rnd_seed)
    alpha = np.sqrt(2 * np.log(1 / (1 - (1 - threshold_prob) ** (1 / no_of_slots))))
    channel_threshold = alpha / np.sqrt(available_power)

    error = []
    for j in range(iterations):
        source_main = source1(no_of_sources)
        source = source_main.copy()
        recovered_appending = np.array([])

        for i in range(1, no_of_slots + 1):
            if i < no_of_slots:
                channel_coeff = np.random.randn(len(source))
                eligible_msgs = source[np.abs(channel_coeff) > channel_threshold]
                if len(eligible_msgs) == 0:
                    continue
                non_transmitted = source[np.abs(channel_coeff) <= channel_threshold]
                channel_gains = channel_coeff[np.abs(channel_coeff) > channel_threshold]
                transmitted = pre_process1(eligible_msgs, channel_gains, alpha, available_power)
                received, noise_var = awgn_mac_with_channel_coeff(transmitted, snr, channel_gains)
                recovered_signal = demod1(received, channel_gains, available_power, alpha, noise_var)
                recovered_appending = np.append(recovered_appending, recovered_signal)
                source = non_transmitted

            elif i == no_of_slots:
                if len(source) == 0:
                    continue
                channel_gains = np.random.randn(len(source))
                transmitted = pre_process2(source, channel_gains, alpha, available_power)
                received, noise_var = awgn_mac_with_channel_coeff(transmitted, snr, channel_gains)
                recovered_signal = demod2(received, channel_gains, available_power, alpha, noise_var)
                recovered_appending = np.append(recovered_appending, recovered_signal)

        error.append(np.mean((source_main.sum() - recovered_appending.sum()) ** 2))
    mse.append(np.mean(error))

# print("\nSNR:", snr_range)
# print("MSE:", mse)

# MMSE Estimation Comparison
def compute_mmse(B, H, x, snr_db):
    n = B.shape[0]
    BH = B * H
    signal = BH @ x
    P_signal = np.sum(x**2) / n
    snr_linear = 10 ** (snr_db / 10)
    P_noise = P_signal / snr_linear
    z = np.random.normal(0, np.sqrt(P_noise), size=(n, 1))
    y = signal + z
    target = np.sum(x)
    estimate = np.sum(y)
    mmse = (estimate - target) ** 2
    return mmse

def norm(arr):
    return np.sum(arr * arr, axis=0)

# Gradient Descent Optimization
mse_gd = []
snrs = np.arange(0, 51)

for snr_db in snrs:
    np.random.seed(3)
    lambda_reg = 1.0
    eta = 0.01
    eta_lambda = 0.1
    tol = 1e-6
    max_iters = 100       #this is for the no of steps to reach the optimal value
    iterations = 1000     #this is for taking average

    H = np.random.randn(no_of_slots, no_of_sources)
    x = np.random.uniform(-1, 1, size=(no_of_sources, 1))
    B = np.random.randn(no_of_slots, no_of_sources)

    for iter in range(max_iters):
        BH = B * H
        y = BH @ x + np.random.randn(no_of_slots, 1) - np.sum(x)
        grad = 2 * (y @ x.T) * H + 2 * lambda_reg * B
        B_new = B - eta * grad
        colwise_norms = norm(B_new)
        B_new1 = B_new.copy()

        if any(colwise_norms > available_power):
            for i in range(B_new.shape[1]):
                if colwise_norms[i] > available_power:
                    scaling_factor = np.sqrt(available_power / colwise_norms[i])
                    B_new1[:, i] *= scaling_factor
            total_col_sum = np.sum(colwise_norms)
            lambda_reg += eta_lambda * (total_col_sum - available_power)
            lambda_reg = max(0, lambda_reg)

        if np.linalg.norm(B_new1 - B, ord='fro') < tol:
            break

        B = B_new1

    error = []
    for _ in range(iterations):
        err = compute_mmse(B, H, x, snr_db)
        error.append(err)
    mse_gd.append(np.mean(error))

# Plotting Results
plt.plot(snr_range, mse, label='Slot-based Scheme')
plt.plot(snrs, mse_gd, label='Gradient Projection Scheme')
plt.xlabel('SNR (dB)')
plt.ylabel('MMSE')
plt.title('MMSE vs SNR')
plt.legend()
plt.grid(True)
plt.show()
