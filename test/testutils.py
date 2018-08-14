import numpy as np

def normdiff(v1,v2) :
    return np.amax(np.fabs(v1-v2))
