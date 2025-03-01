"""
This script is for generation of ambient vibration data. The function is able to generate ambient vibration data based on the given parameters, and the function is to be called in the main script.

Author: Shuaiwen Cui
Date: May 13, 2024

Log:
-  May 13, 2024: initial version

"""

import numpy as np

def ambient_vibration_gen(signal_length, signal_intensity):
    """
    Description: This function is to generate ambient vibration data based on the given parameters.
    
    Parameters:
    
    signal_length: int, the length of the signal to be generated
    
    signal_intensity: float, the intensity of the signal to be generated. For the algorithm used in this function, the signal intensity is used as the standard deviation of the normal distribution.
    
    Returns:
    
    ambient_vibration: numpy array, the generated ambient vibration data
    
    """
    
    # Generate ambient vibration data, all positive values
    ambient_vibration = np.random.normal(0, signal_intensity, signal_length)
    
    return ambient_vibration
    
    
    
    