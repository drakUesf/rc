import matplotlib.pyplot as plt
import numpy as np

from rc import model, reservoir
from parity_benchmark import parity_function

def parity_benchmark_test(n_sample, n_order):
    
    input = np.random.choice([-1, 1], size=(n_sample))

    train = int(0.75 * n_sample)
    test = int(0.25 * n_sample)
    
    input_train = input[:train]
    input_test = input[-test:]

    target_train = parity_function(input_train, n_order)
    target_test = parity_function(input_test, n_order)

    reservoir_states = reservoir(input)

    train_model = model(reservoir_states[50:] , target_train[(50-n_order+1):])

    reservoir_states_test = reservoir_states[-test:]

    # Predict on test set
    predicted_output = np.sign(train_model.predict(reservoir_states_test[:len(target_test)]))


    # Calculate success rate
    success_rate = np.mean(predicted_output == target_test)
    print(f"Success Rate on Unseen Data: {success_rate * 100:.2f}%")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(target_test, label="Target Output", color='g', linestyle='dashed')
    plt.plot(predicted_output, label="Predicted Output", color='b')
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.title(f"Result (Success Rate: {success_rate * 100:.2f}%)")
    plt.show()


if __name__ == '__main__':

    n_sample = 500
    n_order = 2

    parity_benchmark_test(n_sample, n_order)


