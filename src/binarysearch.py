import scipy as sp
from pennylane import numpy as np
from typing import Callable

#certify subroutine, returns 0 if C(x+delta) > eta/2, 1 if C(x-delta) < eta, random if both
def certify(x: float, delta: float, eta: float, acdf: Callable):
    if acdf(x) > 5*eta/8:
        if acdf(x - delta) < 7*eta/8:
            cert_output = np.random.randint(2)
        else:
            cert_output = 0
    else:
        cert_output = 1
    return cert_output

def invert_cdf(delta: float, eta: float, acdf: Callable):
    #initialize x_0 and x_1 values
    x_0, x_1 = -np.pi/3, np.pi/3
    while x_1 - x_0 > 2*delta:
        x = (x_0 + x_1)/2
        u = certify(x, 2*delta/3, eta, acdf)
        if u ==0:
            x_1 = x + 2*delta/3
        else:
            x_0 = x - 2*delta/3
    return (x_0 + x_1)/2
    
