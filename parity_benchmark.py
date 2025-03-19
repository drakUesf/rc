import numpy as np

# parity benchmanrk function
def parity_function(input_sequence, n):
    parity_values = []
    for i in range(n - 1, len(input_sequence)):
        product = 1
        for j in range(n):
            product  = product * input_sequence[i - j] 
        parity_values.append(product)
    return np.array(parity_values)