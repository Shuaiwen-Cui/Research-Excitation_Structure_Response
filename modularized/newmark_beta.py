"""
This script is to calculate the structure response subjected to event excitation using the Newmark-beta method.
Author: Shuaiwen Cui
Date: May 16, 2024

Log:
-  May 16, 2024: initial version

Theory Foundation:

By the state-space representation, the equation of motion for the system subjected to the event excitation can be written as:

M * x'' + C * x' + K * x = GF(t)

which yields the following form:

x'' = INV_M * (- C * x' - K * x + GF(t))
    = - INV_M * C * x' - INV_M * K * x + INV_M * GF(t)

"""
import numpy as np
import matplotlib.pyplot as plt

def newmark_beta(signal_length, nDOF, M, C, K, F, dt, beta=0.25, gamma=0.5):
    """
    Newmark-beta method for solving MDOF systems
    Args:
    M: Mass matrix - nDOF x nDOF
    C: Damping matrix - nDOF x nDOF
    K: Stiffness matrix - nDOF x nDOF
    F: Load time history (each column corresponds to a time step) - nDOF x signal_length
    dt: Time step size
    beta: Newmark-beta parameter (default 0.25 for average acceleration)
    gamma: Newmark-beta parameter (default 0.5 for average acceleration)
    Returns:
    u: Displacement time history - nDOF x signal_length [Disabled]
    v: Velocity time history - nDOF x signal_length [Disabled]
    a: Acceleration time history - nDOF x signal_length [Enabled]
    """
    num_dof = nDOF  # Number of degrees of freedom
    num_steps = signal_length # Number of time steps

    # Initialize arrays for displacement, velocity, and acceleration
    u = np.zeros((num_dof, num_steps))
    v = np.zeros((num_dof, num_steps))
    a = np.zeros((num_dof, num_steps))

    # Newmark Beta method coefficients
    a0 = 1 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt / 2 * (gamma / beta - 2)
    a6 = dt * (1 - gamma)
    a7 = dt * gamma

    # Effective stiffness matrix
    K_eff = K + a0 * M + a1 * C
    
    # Initial accleration put as zero
    a[:, 0] = np.zeros(num_dof)

    # # Time integration using Newmark-beta method
    for i in range(1, num_steps):
        # Compute effective force
        F_eff = F[:, i] + M.dot(a0 * u[:, i-1] + a2 * v[:, i-1] + a3 * a[:, i-1]) \
                + C.dot(a1 * u[:, i-1] + a4 * v[:, i-1] + a5 * a[:, i-1])
        
        # Solve
        u[:, i] = np.linalg.inv(K_eff).dot(F_eff)
        a[:, i] = a0 * (u[:, i] - u[:, i-1]) - a2 * v[:, i-1] - a3 * a[:, i-1]
        v[:, i] = v[:, i-1] + a6 * a[:, i-1] + a7 * a[:, i]

    # for now only acceleration is returned
    return a

# Example usage
if __name__ == "__main__":
    # Define system parameters
    signal_length = 1001
    nDOF = 2 
    m1, m2 = 1.0, 1.0
    k1, k2 = 10000.0, 10000.0
    c1, c2 = 1.0, 1.0
    M = np.array([[m1, 0],
                  [0, m2]])  # Mass matrix
    C = np.array([[c1, 0],
                  [0, c2]])  # Damping matrix
    K = np.array([[k1 + k2, -k2],
                  [-k2, k2]])  # Stiffness matrix

    # Define load time history
    time = np.linspace(0, 10, signal_length)
    dt = time[1] - time[0]
    F = np.zeros((2, len(time)))
    # F[0, :] = 100 * np.sin(2 * np.pi * time)  # Example load on the first degree of freedom
    F[0,0] = 100
    F[1,0] = 100

    # Solve using Newmark-beta method
    a = newmark_beta(signal_length, nDOF, M, C, K, F, dt)
    print(type(a))
    print(a.shape)

    # Plot results
    plt.figure()
    plt.plot(time, a[0, :], label='Acceleration of DOF 1')
    plt.plot(time, a[1, :], label='Acceleration of DOF 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend()
    plt.show()
