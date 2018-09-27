# from https://github.com/numba/numba/issues/2554
from numba import double, guvectorize
import numpy as np

@guvectorize([
    (double[:], double[:]),
], '(n)->(n)', cache=True, forceobj=True)
def gufunc(x, out):
    out[:] = x + 1

if __name__ == "__main__":
    print(gufunc(np.array([1.0, 2.0, 3.0])))
