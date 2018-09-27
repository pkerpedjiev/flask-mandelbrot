from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import numpy as np

# from: https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en

@jit(int32(complex64, int32))
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i],maxiter)
        
def mandelbrot_set2(xmin,xmax,ymin,ymax,width=256,height=256,maxiter=100):
    print("hi")
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    print("yo")
    n3 = mandelbrot_numpy(c,maxiter)
    print("fro")
    return n3.T

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    ret = mandelbrot_set2(-0.74877,-0.74872,0.06505,0.06510,256,256,100)
    return "Hello World! {}".format([str(s) for s in ret])
