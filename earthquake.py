"""
This script is for generation of earthquake data. The function is able to generate earthquake data based on the given parameters, and the function is to be called in the main script.

Author: Shuaiwen Cui
Date: May 14, 2024

Log:
-  May 14, 2024: initial version

"""

import numpy as np
import matplotlib.pyplot as plt

def spec_density(omega, omegag, zetag, S0):
    """
    input parameters:
    
    omega: float, frequency
    
    omegag: float, central frequency
    
    zetag: float, damping ratio of the ground
    
    S0: float, strength factor
    
    return:
    
    spec_density_value: float, spectral density value at the given frequency
    
    """
    
    spec_density_value = S0 * ((2 * zetag * omegag * omega)**2 + omegag**4) / ((omegag**2 - omega**2)**2 + (2 * zetag * omegag * omega)**2)
    
    
    return spec_density_value

def earthquake_gen(signal_length, dt, eq_strength, eq_duration, omegag, zetag, fac_time, pos_time1, pos_time2):
    """
    Description: This function is to generate earthquake data based on the given parameters.
    
    Parameters:
    
    signal_length: int, the length of the signal to be generated
    
    dt: float, time step
    
    eq_strength: float, the strength factor of the spectrum
    
    eq_duration: float, the duration of the earthquake
    
    omegag: float, central frequency
    
    zetag: float, damping ratio of the ground
    
    fac_time: float, factor for time modulation
    
    pos_time1: float, positive time 1
    
    pos_time2: float, positive time 2
    
    Returns:
    
    earthquake: numpy array, the generated earthquake data
    
    """
    
    # initial data generation
    earthquake = np.zeros(signal_length)
    
    # number of terms in the Fourier series
    num_terms = 200
    
    # limits of the random variable
    vl = - np.pi
    vu = np.pi
    
    # earthquake duration
    eq_duration = int(np.floor(eq_duration / dt) * dt)
    
    # time vector
    t_vec = np.arange(dt, eq_duration + dt, dt)
    
    # upper limit of frequency - by Nyquist theorem
    wu = 1 / dt / 2
    dw = wu / num_terms
    
    # frequency vector
    w_vec = np.arange(dw, wu + dw, dw)
    
    # time modulating function
    gt = fac_time * (np.exp(-pos_time1 * t_vec) - np.exp(-pos_time2 * t_vec))
    
    # variable
    theta = vl + (vu - vl) * np.random.randn(1)
    
    # generate the random variables - initialize the variables
    Xk = np.zeros((1, num_terms))
    Yk = np.zeros((1, num_terms))
    
    # generate the random variables for the Fourier series
    for k in range(num_terms):
        Xk[0, k] = np.sqrt(2) * np.cos(k * theta + np.pi / 4)
        Yk[0, k] = np.sqrt(2) * np.sin(k * theta + np.pi / 4)
        
    # to make the random variables more random
    Xk = np.apply_along_axis(np.random.permutation, 1, Xk)
    Yk = np.apply_along_axis(np.random.permutation, 1, Yk)
    
    k = np.arange(1, num_terms + 1)
    
    # initialize the ground acceleration
    Ag = np.zeros((1, len(t_vec)))
    
    # generate the ground acceleration
    for i in range(len(t_vec)):
        # generate each term in the Fourier series
        Ag_terms = (np.sqrt(2 * spec_density(dw * k, omegag, zetag, eq_strength) * dw) *
                 (np.cos(dw * k * i * dt) * Xk + np.sin(dw * k * i * dt) * Yk))
        # sum up all the terms to get the ground acceleration at the given time
        Ag_series = np.sum(Ag_terms) # --- so far, the frequency domain requirements are met
        # apply the time modulation to get the final ground acceleration
        Ag[0, i] = Ag_series * gt[i] # --- so far, the time domain requirements are met too
    
    # pick a random starting point, and assign the generated earthquake data to the final earthquake data
    scale_factor = 0.7
    start_point = np.random.randint(0, np.floor(scale_factor*signal_length))
    
    # find the smaller value of the two: signal_length - start_point and length of the generated earthquake data
    smaller_ul = min(signal_length - start_point, len(Ag[0, :]))
    for i in range(smaller_ul):
        earthquake[start_point - 1 + i] = Ag[0, i]
        
    # use PGA to normalize and rescale the earthquake data
    PGA = max(abs(earthquake))
    
    rescale_factor = np.random.rand(1)*0.9*9.81 + 0.1*9.81
    
    earthquake = earthquake / PGA * rescale_factor
    
    return earthquake

if __name__ == '__main__':
    signal_length = 6000
    dt = 0.01
    eq_strength = 0.6
    eq_duration = 20.0
    omegag = 15
    zetag = 0.6
    fac_time = 12.21
    pos_time1 = 0.1
    pos_time2 = 0.5
    
    # generate the earthquake data
    earthquake = earthquake_gen(signal_length, dt, eq_strength, eq_duration, omegag, zetag, fac_time, pos_time1, pos_time2)
    
    print(earthquake.shape)
    
    # plot the earthquake data
    plt.figure()
    plt.plot(earthquake)
    plt.xlabel('Time')
    plt.ylabel('Ground Acceleration')
    plt.title('Generated Earthquake Data')
    plt.show()
    
