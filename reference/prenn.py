"""
This script is to preprocess the raw data signal before feeding it into the neural network. The function is able to preprocess the raw data signal based on the given parameters, and the function is to be called in the main script.

The key procedures include:

- time-frequency transformation to get PSD
- downsampling the time domain data to reduce the data size
- combining the PSD and time domain data to get the data for the neural network

Author: Shuaiwen Cui
Date: May 18, 2024

Log:
-  May 18, 2024: initial version

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # for signal processing
from scipy.signal import hilbert  # for signal processing


def prenn(signal_input, dt, nperseg, output_length):
    '''
    
    INPUTS PARAMETERS:
    
    signal_input: numpy array, the raw data signal, each row is a channel, each column is a time point
    
    dt: float, the time interval between two consecutive points
    
    nperseg: int, the number of data points used in each segment for the PSD calculation
    
    output_length: int, the desired output length of the preprocessed data
    
    OUTPUTS:
    
    signal_for_nn: numpy array, the preprocessed data for the neural network, each row is a channel, each column is a time point
    
    
    '''
    
    # ensure the output size is positive
    if output_length <= 0:
        raise ValueError('Output length must be positive')
    if not isinstance(output_length, int):
        raise ValueError('Output length must be an integer')
    
    # print input signal size
    # print('signal_input.shape = ', signal_input.shape)
    
    # ADAPTATION PROCESS
    input_length = signal_input.shape[1]
    
    # FREQUENCY DOMAIN PROCESS
    freq, psd = signal.welch(signal_input, fs=1/dt, nperseg=nperseg)
    
    tmp_psd_len = psd.shape[1]
    freq = freq.reshape(-1, tmp_psd_len)
    
    # convert the power spectrum density to dB
    psd = 10 * np.log10(psd)
    
    # normalize the power spectrum density
    tmp_psd = (psd - np.mean(psd)) / np.std(psd)

    # replace the nan value with 0
    tmp_psd[np.isnan(tmp_psd)] = 0
    
    psd = tmp_psd  
    
    psd = np.reshape(psd, (1, -1))  # reshape the power spectrum density
    
    # len_psd
    len_psd = int(output_length / 2)
    
    # resample the frequency to the desired length
    # freq = signal.resample(freq, len_psd, axis=1)
    
    # resample the power spectrum density to the desired length
    psd = signal.resample(psd, len_psd, axis=1)
    
    # TIME DOMAIN PROCESS
    
    # len_ts
    len_ts = output_length - len_psd
    
    # hilbert transform to get the envelope
    hilbert_signal = np.abs(hilbert(signal_input))
    
    # print('hilbert_signal.shape = ', hilbert_signal.shape)
    
    # resample the envelope to the desired length
    ts = signal.resample(hilbert_signal, len_ts, axis=1)
    
    # print('ts.shape = ', ts.shape)
    
    # normalize the time domain signal
    ts = (ts - np.mean(ts)) / np.std(ts)
    
    # print('max abs ts' , np.max(np.abs(ts)))
    
    # RETURNING DATA PREPARATION
    
    # return the preprocessed data
    signal_for_nn = np.concatenate((ts, psd), axis=1)
    
    # print('signal_for_nn.shape = ', signal_for_nn.shape)
    
    return signal_for_nn

if __name__ == '__main__':
    signal_test = np.random.rand(1, 1000)
    dt = 0.01
    nperseg = 512
    output_length = 100
    signal_for_nn = prenn(signal_test, dt, nperseg, output_length)
    print(signal_for_nn.shape)
    
    # plot the preprocessed data
    plt.figure()
    plt.plot(signal_for_nn[0, :])
    plt.show()
 
    