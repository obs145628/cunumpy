import numpy as np
import time

import pyhack

N = 8000
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)

start = time.time()
c1 = np.dot(a, b)
dur = int((time.time() - start) * 1000)
print('Duration: {}ms'.format(dur))

pyhack.enable_wrapper()

np.dot(a[:2, :2], b[:2, :2]) #overhead when first call to cuda function

start = time.time()
c2 = np.dot(a, b)
dur = int((time.time() - start) * 1000)
print('Duration: {}ms'.format(dur))

print('Difference = {}'.format(np.linalg.norm(c1 - c2)))
