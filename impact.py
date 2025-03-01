"""
This script is for generation of impact data. The function is able to generate ambient vibration data based on the given parameters, and the function is to be called in the main script.

Author: Shuaiwen Cui
Date: May 14, 2024

Log:
-  May 14, 2024: initial version

"""

import numpy as np

def impact_gen(signal_length, signal_intensity = 10, signal_std = 5):
    """
    Description: This function is to generate impact data based on the given parameters.
    
    Parameters:
    
    signal_length: int, the length of the signal to be generated
    
    signal_intensity: float, the intensity of the signal to be generated. For impact, it is the amplitude of the signal. (absolute value)
    
    Returns:
    
    impact: numpy array, the generated impact data
    
    """
    # generate the sign, either 1 or -1
    sign = np.random.choice([-1, 1])
    
    # generate the impact amplitude, centered at signal intensity, with standard deviation of signal_std
    impact_amplitude = 0.5 * signal_intensity + np.random.normal(signal_intensity, signal_std)
    
    # generate the impact data
    impact = np.zeros(signal_length)
    
    # generate the impact time
    impact_time = np.random.randint(0, signal_length)
    
    # generate the impact data
    impact[impact_time] = sign * impact_amplitude
    
    return impact

# testing
if __name__ == '__main__':
    impact = impact_gen(100)
    print(impact)