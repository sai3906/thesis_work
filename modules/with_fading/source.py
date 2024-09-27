import numpy as np
import matplotlib.pyplot as plt
def source(no_of_users, no_of_bits,rnd_seed):
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_bits)
    # Each element is either 0 or 1
    return np.random.randint(0, 2, (no_of_users, no_of_bits))
