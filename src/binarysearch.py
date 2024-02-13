import scipy as sp
from pennylane import numpy as np
from typing import Callable

def invert_cdf(delta: float, eta: float, acdf: Callable) -> float:
    #certify subroutine, returns 0 if acdf(x) < eta - epsilon, 1 if acdf(x) > epsilon
    def certify(x: float, eta: float, acdf: Callable) -> int:
        if acdf(x) < eta/2:
            cert_output = 0
        else:
            cert_output = 1
        return cert_output
    
    #initialize x_0 and x_1 values
    x_0, x_1 = -np.pi/2, np.pi/2
    while x_1 - x_0 > 2*delta:
        x = (x_0 + x_1)/2 #set x to be midpoint between x_0 and x_1
        u = certify(x, 2*delta/3, eta, acdf) #use certify subroutine to check if C(x+(2/3)delta) > eta/2 or C(x-(2/3)delta) < eta
        if u == 0:
            x_1 = x + 2*delta/3 #move new x_1 point closer to x if C(x+(2/3)delta) > eta/2
        else:
            x_0 = x - 2*delta/3 #move new x_0 point closer to x if C(x-(2/3)delta) < eta
    return (x_0 + x_1)/2
    
