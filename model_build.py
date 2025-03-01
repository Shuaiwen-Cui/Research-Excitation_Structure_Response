"""
This script is to build up the model for the event simulation.
Author: Shuaiwen Cui
Date: May 14, 2024

Log:
-  May 14, 2024: initial version

"""

import numpy as np
import sys

def model_build(nDOF = 10, me = 29.13, ke = 1190e3, zeta = 0.00097):
    '''
    Description: This function is to build up the model for the event simulation.
    
    Parameters:
    
    nDOF: int, the number of degrees of freedom
    
    me: float, the mass for each floor
    
    ke: float, the stiffness unit N/m
    
    zeta: float, the damping ratio - actual damp / critical damp
    
    there are default values for the parameters, and the user input will overwrite the default values when used.
    
    Returns:
    
    M: numpy array, the mass matrix
    
    K: numpy array, the stiffness matrix
    
    C: numpy array, the damping matrix
    
    '''
    
    # M - mass matrix - diagonal matrix with mass me for each floor
    M = np.zeros((nDOF, nDOF))
    for i in range(nDOF):
        M[i, i] = me

    # K - stiffness matrix
    K = np.zeros((nDOF, nDOF))
    for i in range(nDOF):
        K[i, i] = 2*ke
        if i > 0:
            K[i, i-1] = -ke
        if i < nDOF-1:
            K[i, i+1] = -ke

    K[nDOF-1, nDOF-1] = ke

    # C - damping matrix

    ## calculate the eigenvalues and eigenvectors of the system
    INV_M = np.linalg.inv(M)
    INV_M_K = np.dot(INV_M, K)
    eigenvalues, eigenvectors = np.linalg.eig(INV_M_K)

    ## characteristic frequency
    omega = np.sqrt(eigenvalues)
    omega = np.diag(omega)

    ## damping matrtix
    C = 2 * zeta * me * np.linalg.inv(np.transpose(eigenvectors)) @ omega @ np.linalg.inv(eigenvectors)
    
    # return
    return M, K, C

# testing
if __name__ == '__main__':
    M, K, C = model_build()
    print('Model built successfully!')
    print('Mass matrix: ', M)
    print('Stiffness matrix: ', K)
    print('Damping matrix: ', C)

