from models.pspnet import Dropout2d
import numpy as np
import megengine as mge


inp = np.random.randn(2, 3, 4, 4)
print(inp)
inp = mge.tensor(inp)
m = Dropout2d(p=0.5)
out = m(inp)
print(out.numpy()) # some channel will be zero
