import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt


class time_delay_RC():
    
    #set the parameter
    def __init__(self, f_0 = 228e3, beta = 1.9e23, a = 0.5, A_0 = 82, f_d = 115100, tau = 0.04, N = 400, Q = 100):

        self.f_0 = f_0
        self.beta = beta
        self.a = a
        self.A_0 = A_0
        self.f_d = f_d
        self.tau = tau     
        self.N = N

        self.omega_0 = 2 * np.pi * self.f_0
        self.Q = Q

        self.gamma = self.omega_0 / self.Q
        self.alpha = self.omega_0 ** 2
       
        self.theta = float(self.tau / self.N)

        self.fs = 1 / self.theta

    
    def envelope_detection(self, signal, fs, method="hilbert", cutoff_freq=50):
        if method == "hilbert":
            analytic_signal = hilbert(signal)
            return np.abs(analytic_signal)
        elif method == "lowpass":
            nyquist = 0.5 * fs
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(5, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, np.abs(signal))
        else:
            raise ValueError("Method must be 'hilbert' or 'lowpass'")

    # generate the reservoir states
    def generate_reservoir_states(self, input_data):

        n_sample = len(input_data)# length of input

        time = np.linspace(0, self.tau * n_sample, n_sample) #

        reservoir_states = np.zeros((n_sample, self.N)) # initial reservoir state

        feedback = np.zeros(self.N) #initial feedback

        mask = np.random.choice([0.45, 0.70], size = self.N)#initial mask

        u_min = 0.60
        u_max = 0.75

        state = [0.5, 0.0]

        for i in range(n_sample):
            print(f'timestep {i + 1}/{n_sample}', end='\r')

            t = time[i]

            normalized_input = u_min + (u_max - u_min) * (input_data[i] + 1) / 2 #normalize the input

            t_span = [0, 0 + self.tau]

            t_eval = np.linspace(t_span[0], t_span[1], 100) 

            for j in range(self.N):

                #input force
                F = lambda t: (self.A_0 * (normalized_input * mask[j] + self.a * feedback[j] + 1) * np.cos(2 * np.pi * self.f_d * t)) ** 2               
                
                #duffing equation
                def duffing_oscillator(t, state):
                    x, v = state
                    dvdt = - self.gamma * v - self.alpha * x - self.beta * x**3 + F(t)    
                    return [v, dvdt]
                
                #solve the equation
                solution = solve_ivp(duffing_oscillator, t_span = t_span, y0 = state, method='RK45', t_eval = t_eval)
                
                #solution = odeint(duffing_oscillator, state, t_span, atol=1e-6, rtol=1e-6)

                x = np.mean(solution.y[0][-50:])
                v = np.mean(solution.y[1][-50:])

                state = [x , v]

                reservoir_state = self.envelope_detection([x], self.fs, method="hilbert")
    
                reservoir_states[i, j] = reservoir_state

                if i < n_sample - 1:
                    feedback[j] = reservoir_state

        return reservoir_states
    
    
    def train(self, reservoir, target):

        model = Ridge(alpha=1e-5)
        model.fit(reservoir[:len(target)], target)

        return model
    

def reservoir(input):

    instance = time_delay_RC()

    reservoir = instance.generate_reservoir_states(input)
    
    return reservoir

def model(input, target):

    instance = time_delay_RC()

    model = instance.train(input, target)

    return model























