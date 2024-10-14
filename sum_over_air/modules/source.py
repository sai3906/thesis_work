import numpy as np
import matplotlib.pyplot as plt
def source(no_of_users, no_of_slots,rnd_seed):
    np.random.seed(rnd_seed)
    # Generate a random binary matrix with shape (no_of_users, no_of_values_per_node)
    # Each element is from  uniform distribution
    return np.random.uniform(-1, 1, (no_of_users, no_of_slots))
