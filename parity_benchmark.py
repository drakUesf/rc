import numpy as np

# parity benchmanrk function
def parity_function(input_sequence, n):
    parity_values = []
    for t in range(n, len(input_sequence) + 1):
        parity_values.append(np.bitwise_xor.reduce(input_sequence[t - n:t]))
    return np.array(parity_values)