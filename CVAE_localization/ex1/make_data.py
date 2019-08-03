import numpy as np

np.random.seed(seed=42)

def states():
    x=np.arange(0.0,100.0,1.0) + np.random.normal(0.0,0.1)
    y=np.arange(0.0,100.0,1.0) + np.random.normal(0.0,0.1)
    return x,y

def observations(x,y,x_0,y_0):
    return [np.linalg.norm(np.array([xt,yt]) - np.array([x_0,y_0])) + np.random.normal(0.0,0.1) for _,(xt,yt) in enumerate(zip(x,y))]

def make_data():
   x,y = states()
   x_0 = 0.0
   y_0 = 0.0
   o = observations(x,y,x_0,y_0)
   input = [[xt,yt,ot] for _, (xt,yt,ot) in enumerate(zip(x,y,o))]
   output = [[xt,yt] for _,(xt,yt) in enumerate(zip(x[1:],y[1:]))]
   return input,output
