"""
This script is for generation of strong wind data. The function is able to generate strong wind data based on the given parameters, and the function is to be called in the main script.

Author: Shuaiwen Cui
Date: May 15, 2024

Log:
-  May 15, 2024: initial version

Theory Briefing:

The wind load is the multiplication of the wind pressure and the wall area:

Lt(t) = Pt(t) * At                       --- (1)

Pt(t) is the wind pressure matrix, which can be viewed as the combination of the static and the dynamic wind pressure:

Pt(t) = Ps + Pd(t)                       --- (2)

>> I - Static Wind Pressure:

Ps is the static wind pressure, which is the product of the air density, the wind speed, and the drag coefficient:

Ps = w0 * uz * us * ur                   --- (3)

where:

- w0: basic wind pressure
- uz: height factor
- us: shape factor
- ur: reccurence factor

w0 = 0.5 * rho * v0^2                    --- (4)

- rho: air density, take 1.225 kg/m^3
- v0: basic wind speed, at the height 10m, take 25 m/s

uz = (H[i]/10)^(0.3)                     --- (5)

- H[i]: height of the i-th floor

us = 0.8 + 0.5 = 1.3                     --- (6)

>> II - Dynamic Wind Pressure:

Pd(t) is the dynamic wind pressure, which is estimated by the auto-regressive model, which has two major parts: the initialization and the iteration.

Auto-regressive model assumes that the current value is the linear combination of the previous values, plus the white noise.

In this case the AR degree is p, and the time step is \Delta t. (For illustration purpose, take p = 5, and \Delta t = 0.01s.)

By the AR model, Pd(t) can be expressed as:

Pd(t) = \Sigma_{k=1}^{p}(Psi_{k}P(t-k\Delta t)) + \sigma^{N}N(t)     --- (7)

In (7), the first term stands for the autoregressive part, and the second term stands for the white noise part.

We note Pd(t) as T1 + T2, where T1 is the autoregressive part, and T2 is the white noise part.

In T2, the N(t) follows the normal distribution with the mean 0 and the variance 1, quite easy to simulate. Yet, for \sigma^{N}, it is the standard deviation of the white noise, which is to be estimated by the given data, more specifically, the initialization part or user input. To calculate \sigma^{N}, we need to calculate the correlation matrix of the given data, and then the standard deviation can be estimated by the lower triangular matrix of the Cholesky decomposition of the correlation matrix, R. (R(0 \Delta t) )

Note: R stands for the spatial correlation, while \Psi stands for the temporal correlation.

\sigma^{N} = np.linalg.cholesky(R(0 \Delta t))     --- (8)

For T1, \Psi_{k} is the autoregressive coefficient, which can also be estimated by the given data, more specifically, the initialization part or user input. 

\Psi = np.linalg.inv(RR)@ RC                       --- (9)

where:

RR = [R(0 \Delta t), R(\Delta t), R(2\Delta t), ..., R((p-1)\Delta t); R(1 \Delta t), R(0 \Delta t), R(\Delta t), ..., R((p-2)\Delta t); ...; R((p-1) \Delta t), R((p-2) \Delta t), ..., R(0 \Delta t)]     --- (10)

RC = [R(0 \Delta t), R(\Delta t), R(2\Delta t), ..., R((p-1)\Delta t)]     --- (11)

\Psi = [Psi_{1}, Psi_{2}, ..., Psi_{p}] = np.linalg.inv(RR)@ RC     --- (12)

So now, the problem comes down to how to calculate the spatial correlation, R, with R we can also calculate Psi.

R(m \Delta t) = [R_{ij}(m \Delta t)]               --- (13)

R_{ij}(m \Delta t) = \int_{0}^{\infty} S_{ij}(fv)cos(2 * np.pi * fv *m\Delta t)df     --- (14)

where:

- S_{ij}(f) is the power spectral density array, Davenport spectrum is used in this case
- fv is the frequency vector, the range is from 0 to 1/(2\Delta t), by Nyquist theorem
- m is the index of time lag in AR model
- \Delta t is the time step

The Davenport spectrum is given by:

S_{ij}(f) = \rho_{ij} * \sigma(z_{i}) * \sigma(z_{j}) S_{d}(f)   --- (15)

where:

- \rho_{ij} is the correlation coefficient, which is given by:

\rho_{ij} = exp(-\frac{(z_i - z_j)^2}{Lz})^(0.5)     --- (16)

\sigma(z_{i}) is the standard deviation of the wind speed at the height z_{i}, which is given by:

\sigma(z) = uf(z) * uz(z) * us * ur * w0 / u    --- (17)

where:

- ur is the reccurence factor, take 1.1
- us is the shape factor, take 1.3
- uz is the height factor, which is given by

uz(z) = (z/10)^(0.3)     --- (18)

- uf is the ratio of the pulsating wind over the static wind, which is given by:

uf(z) = 0.5 * 35^(1.9 * (alpha - 0.16))*(z/10)^(0.3)

- alpha is the terrain factor, take 0.16
- u is the safety factor, take 2

S_{d}(f) is the power spectral density of the wind speed, which is given by:

S_{d}(f) = 4 * K_Davenport*\bar(v10^2)*x0^2/f/(1+x0^2)^(4/3)

x0 = 1200f/v10

"""

# DEPENDENCIES
import numpy as np
import matplotlib.pyplot as plt

# strong wind generation
def strong_wind_gen(signal_length, nDOF, dt):
    """
    Description: This function is to generate strong wind data based on the given parameters.
    
    Parameters:
    
    signal_length: int, the length of the signal to be generated
    
    nDOF: int, number of degrees of freedom
    
    dt: float, time step
    
    Returns:
    
    strong_wind: matrix, the generated strong wind data
    
    
    - first part for configuration and parameter calculation
    - second part for static wind load calculation
    - third part for dynamic wind load calculation
    
    
    """
    # CONFIGURATION
    print_flag = 0
    
    # PARAMETERS AND RELATED CALCULATIONS
    
    ## building width and height [configurable]
    building_width = 30
    storey_height = 4
    
    ## storey heights [auto calculated]
    storey_heights = np.ones(nDOF) * storey_height
    storey_heights[-1] = 0.5 * storey_heights[-1]
    
    if print_flag:
        print('Storey Heights: ', storey_heights)
    
    ## area that the wind is acting on [auto calculated]
    Area = building_width * storey_heights
    
    if print_flag:
        print('Wall Area: ', Area)
    
    ## building heights [auto calculated]
    H = np.arange(storey_height, storey_height*(nDOF+1), storey_height)
    
    if print_flag:
        print('Mass Lump Height: ', H)

    ## air density [configurable]
    rho = 1.225 
    
    ## basic wind speed [configurable]
    v0 = 25
    
    ## basic wind pressure <auto calculated>
    w0 = 0.5 * rho * v0**2
    
    if print_flag:
        print('Basic Wind Pressure: ', w0)
    
    ## shape factor [configurable]
    us = 0.8 + 0.5 
    
    ## recurrence factor [configurable]
    ur = 1.1
    
    ## safety factor [configurable]
    u = 2
    ## frequency division factor [configurable]
    f_division = 100 # alternatively, the user can use signal_length for division
    
    ## frequency vector - based Nyquist Theorem [auto calculated]
    fu = 1/(2*dt) 
    fv = np.linspace(0, fu, f_division+1)
    
    if print_flag:
        print('Upper limit of Frequency Vector: ', fu)
        print('Frequency Vector: ', fv)
    
    ## frequency resolution [auto calculated]
    df = fv[1] - fv[0]
    
    if print_flag:
        print('Frequency Resolution: ', df)
        
    ## uz - height factor [auto calculated]
    uz = (H/10)**0.3
    
    if print_flag:
        print('Height Factor: ', uz)
    
    ## uf - pulsating wind over static wind ratio [auto calculated]
    ## alpha - terrain factor [configurable]
    alpha = 0.16
    ## uf calculation [auto calculated]
    uf = 0.5 * 35**(1.9 * (alpha - 0.16)) * (H/10)**0.3
    
    ## sigmaz - standard deviation of wind speed [auto calculated]
    sigmaz = uf * uz * us * ur * w0 / u # nDOF x 1
    
    if print_flag:
        print('Standard Deviation of Wind Speed: ', sigmaz)
        
    ## x0 - in Sd(f) Calculation [auto calculated]
    x0 = 1200 * fv / v0
    
    ### K_Davenport [configurable]
    K_Davenport = 0.0025
    
    ### Sdf
    Sdf = 4 * K_Davenport * v0**2 * x0**2 / fv / (1 + x0**2)**(4/3)
    Sdf = np.nan_to_num(Sdf)
    
    if print_flag:
        print('Sdf: ', Sdf)
        # plt.plot(fv, Sdf)
        plt.figure()
        plt.plot(np.log(fv), np.log(Sdf))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Davenport Spectrum log-log Plot')
        plt.show()
        
    ## correlation coefficient [auto calculated]
    Rho_Mat = np.zeros((nDOF,nDOF))
    for i in range(nDOF):
        for j in range(nDOF):
            Rho_Mat[i,j] = np.exp(-(np.sqrt((H[i] - H[j])**2/60)))
            
    ## Auto Regressive Degree [configurable]
    p = 5
        
    # >> I - Static Wind Load Computation
    
    pressure_static = np.ones(nDOF)
    for i in range(nDOF):
        pressure_static[i] = w0 * uz[i] * us * ur 
    
    load_static = pressure_static * Area # nDOF x 1
    
    # after the dynamic part is calculated, this part can be added to the dynamic part to get the total wind load. Broadcast can be used.
       
    # >> II - Dynamic Wind Load Computation
    
    ## Calculate Sdf Matrix
    Sdf_Mat = np.zeros((nDOF,nDOF,f_division+1))
    
    for i in range(nDOF):
        for j in range(nDOF):
            Sdf_Mat[i,j,:] = Rho_Mat[i,j] * sigmaz[i] * sigmaz[j] * Sdf
    
    if print_flag:
        print('Sdf Matrix: ', Sdf_Mat)
        print('Sdf Matrix Shape: ', Sdf_Mat.shape)
    
    ## Calculate R Matrix - make a list of R, put the in a matrix, like [R(0 \Delta t), R(\Delta t), R(2\Delta t), ..., R((p-1)\Delta t)]
    
    R_List = np.zeros((nDOF, nDOF * p))
    for k in range (p):
        for i in range(nDOF):
            for j in range(nDOF):
                R_List[i,j + k*nDOF] = np.trapz(Sdf_Mat[i,j,:] * np.cos(2 * np.pi * fv * k * dt), fv)
    
    if print_flag:
        print('R Matrix: ', R_List)
        print('R Matrix Shape: ', R_List.shape)
    
    
    # for zero time lag, we can have R(0 \Delta t) as the first element of the list, which is noted as R0
    R0 = R_List[:,:nDOF]
    
    if print_flag:
        # check if R0 is symmetric
        if not np.allclose(R0, R0.T):
            print('R0 is not symmetric')
        else:
            print('R0 is symmetric')

        # check if R0 is positive definite
        if not np.all(np.linalg.eigvals(R0) > 0):
            print('R0 is not positive definite')
        else:
            print('R0 is positive definite')
    
    ## RR - the matrix in the denominator of Psi calculation
    RR = np.zeros((nDOF * p, nDOF * p))
    for i in range(p):
        for j in range(p):
            k = np.abs(i - j)
            RR[i*nDOF:(i+1)*nDOF,j*nDOF:(j+1)*nDOF] = R_List[:,k*nDOF:(k+1)*nDOF]                 
    
    if print_flag:
        print('RR Matrix: ', RR)
        print('RR Matrix Shape: ', RR.shape)
       
    ## RC - the matrix in the numerator of Psi calculation
    RC = R_List.transpose()
    
    if print_flag:
        print('RC Matrix: ', RC)
        print('RC Matrix Shape: ', RC.shape)
        
    ## Psi - the autoregressive coefficient matrix
    Psi = np.linalg.inv(RR) @ RC
    
    if print_flag:
        print('Psi Matrix: ', Psi)
        print('Psi Matrix Shape: ', Psi.shape)

    ## PSI
    PSI = np.zeros((nDOF, nDOF, p))
    
    ## Sigma N - the standard deviation of the white noise
    Sigma_N = np.linalg.cholesky(R0)
    
    if print_flag:
        print('Sigma N Matrix: ', Sigma_N)
        print('Sigma N Matrix Shape: ', Sigma_N.shape)
    
    ## White Noise Generation
    N = np.random.randn(nDOF, signal_length)
    
    Pd_WN = Sigma_N @ N
    
    ## Autoregressive Part Calculation
    Pd_AR = np.zeros((nDOF, signal_length))
    AR_Ini = np.random.randn(nDOF, p)
    
    # AR_Ini is the initialization part of the AR model, which is the first p points of the signal
    for i in range(p):
        Pd_AR[:,i] = AR_Ini[:,i]
    
    for i in range(p+1, signal_length):
        for j in range(p):
            Pd_AR[:,i] = Pd_AR[:,i] + PSI[:,:,j] @ Pd_AR[:,i-j]
        
    ## Dynamic Wind Load Calculation
    Pd = Pd_AR + Pd_WN
    
    # plot the first row of Pd
    if print_flag:
        # print shapes of Pd, Pd_AR, Pd_WN
        print('Pd Shape: ', Pd.shape)
        print('Pd_AR Shape: ', Pd_AR.shape)
        print('Pd_WN Shape: ', Pd_WN.shape)
        plt.figure()
        plt.plot(Pd[0,:])
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title('Pressure Plot')
        plt.show()
    
    ## time modulation
    ## center of the wind in time-domain
    miu = np.random.randint(np.floor(0.2*signal_length), np.floor(0.8*signal_length))
    
    ## standard deviation of the wind in time-domain
    sigma_wind = 8 + 5 * np.random.rand()
    
    ## scale factor
    scale_factor = 2 + 1 * np.random.rand()
    
    ## time series
    ts = np.linspace(0, signal_length*dt, signal_length)
    
    ## scale_wind 
    scale_wind = np.ones(signal_length)
    scale_wind = scale_wind + scale_factor * np.exp(-0.5 * ((ts - miu*dt) / sigma_wind)**2)
    
    if print_flag:
        print('Scale Wind: ', scale_wind)
        print('Scale Wind Shape: ', scale_wind.shape)
        
        # plot the time modulation
        plt.figure()
        plt.plot(scale_wind)
        plt.xlabel('Time')
        plt.ylabel('Scale Factor')
        plt.title('Time Modulation')
        plt.show()
    
    ## time modulation
    Pd = Pd * scale_wind.transpose()
    
    if print_flag:
        print('Pressure Matrix with Time Modulation: ', Pd)
        print('Pressure Matrix Shape with Time Modulation: ', Pd.shape)
    
    # print('area shape', Area.shape)
    
    # load_dynamic
    load_dynamic = np.zeros((nDOF, signal_length))
    for i in range(nDOF):
        load_dynamic[i,:] = Pd[i,:] * Area[i]
    
    if print_flag:
        print('Dynamic Load Shape: ', load_dynamic.shape)
        # plot the first row of load_dynamic
        plt.figure()
        plt.plot(load_dynamic[0,:])
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.title('Load Plot')
        plt.show()  
    
    # >> Integration
    strong_wind = load_static.reshape(-1,1) + load_dynamic
    
    if print_flag:
        print('Wind Load Shape: ', strong_wind.shape)
        # plot the first row of strong_wind
        plt.figure()
        plt.plot(strong_wind[0,:])
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.title('Load Plot')
        # add horizontal line for the static part
        plt.axhline(y=load_static[0], color='r', linestyle='--')
        # add horizontal line for the zero line
        plt.axhline(y=0, color='k', linestyle='--')
        plt.show()
    
    return strong_wind

if __name__ == '__main__':
    signal_length = 6000
    nDOF = 10
    dt = 0.01
    strong_wind = strong_wind_gen(signal_length, nDOF, dt)
    